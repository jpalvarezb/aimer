"""DelegateAgent — the robust non-realtime agent behind ``delegate_task(goal)``.

The decoupling this completes: the realtime audio model stays conversational and emits ONE
tool call; this agent (gemini-3.5-flash over the async Interactions API) receives the goal
plus the pointer-referent history and orchestrates the actual work with host tools:

  - ``run_shell``        — zsh one-liners (files, git, CLIs)
  - ``run_applescript``  — app control via osascript
  - ``browser_*``        — a persistent Chromium (Phase 4, registered by the caller)
  - ``computer_use``     — the Week-7b screenshot+mouse+keyboard loop as general fallback,
                           serialized by an optional desktop mutex and run in a worker
                           thread (its policy uses the sync client + a blocking
                           ``screencapture``; the loop must never run on the event loop).
                           If the hosted ``GeminiComputerUsePolicy`` gets input-blocked by
                           Google's server-side classifier (hard exception or a soft
                           ``safety_decision: blocked`` done result), the SAME goal is
                           retried under the mutex with ``GeminiVisionLoopPolicy`` (plain
                           ``generateContent``, no server-side gate) before giving up.

The agent loop mirrors :class:`GeminiComputerUsePolicy`'s round-trip bookkeeping (function
calls -> execute -> ``function_result`` steps -> ``previous_interaction_id`` chaining), but
dispatches to its OWN tool handlers instead of OS primitives, and runs fully async
(``client.aio``) so many delegate tasks can share the loop without stalling audio.

Safety pause/resume: a handler may raise :class:`ConfirmationRequired` (the Phase-6
classifier does this for non-allowlisted commands). The loop then returns
``status="awaiting_confirmation"`` with the pending command; the live model relays the
question by voice, and ``resume(approved=True)`` re-invokes the same handler with the
confirmation bypassed, continuing the SAME interaction chain.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import threading
import warnings
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .computer import Computer, MacOSComputer, Policy, run_computer_use
from .computer_policy import (
    GeminiComputerUsePolicy,
    GeminiVisionLoopPolicy,
    is_policy_block,
    wrap_with_vision_loop_fallback,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"

# Handlers take the model's arguments dict and return a JSON-serializable result.
DelegateToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]

_SYSTEM = """\
You are the action agent for Aimer, a voice assistant on the user's macOS desktop. You \
receive one goal, plus pointer context describing what the user pointed at while speaking. \
Accomplish the goal with your tools, preferring the cheapest that does the job: run_shell \
for files/git/CLIs, run_applescript for controlling macOS apps, browser_* for web tasks, \
and computer_use (OS-level mouse+keyboard driven by a vision model) ONLY when nothing else \
fits. For reading or extracting information from web pages, prefer the headless browser_* \
tools — never launch a visible test-browser window. When the user should actually SEE a \
page themselves, run `open <url>` via run_shell to open it in their real default browser \
instead of driving a browser yourself. Before reaching for computer_use to control an app, \
try a read-only AppleScript query first (e.g. System Events get/count/exists of processes, \
windows, or UI elements) — it is faster and needs no vision model. If a task needs many \
similar shell steps in a row, ask the user once whether to proceed with the rest, rather \
than confirming each command individually. Ground deictic goals ("this", "that") in the \
pointer context. Do not ask questions — make the reasonable choice and note it. If a tool \
result says policy_block, or the same approach fails twice, STOP retrying it: try one \
genuinely different tool, or finish and report honestly what could not be done and why. \
After performing the goal's action(s), verify the outcome with a cheap read-back query (an \
AppleScript get/exists, or a shell read) before reporting success; if the read-back shows a \
mismatch (wrong title, missing content), fix it before finishing. Never claim success \
without having observed it. Note for macOS Notes: it derives a note's title from the first \
line of its body, so `make new note with properties {name:"X"}` alone does not stick — set \
the title as the first line of the body. Resolve relative dates ("today", "yesterday", "this \
week") yourself with the `date` command against the system clock; if the goal states an \
absolute date that contradicts the system clock, trust the system clock and the user's \
relative intent instead, note the discrepancy in your final summary, and do not go hunting \
for data matching the wrong date. Any scratch or temporary file you create must go under a \
temp directory (`mktemp -d` or `$TMPDIR`) — never the current working directory (cwd) or the \
user's project folders — and clean it up afterward when reasonable. Finish with a \
one-sentence summary of what you did.\
"""

# Parameter schemas for the built-in delegate tools (Interactions API function tools).
_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "run_shell": {
        "description": "Run a shell command on the user's Mac and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command line."}},
            "required": ["command"],
        },
    },
    "run_applescript": {
        "description": "Run an AppleScript (osascript) to control macOS applications.",
        "parameters": {
            "type": "object",
            "properties": {"script": {"type": "string", "description": "AppleScript source."}},
            "required": ["script"],
        },
    },
    "computer_use": {
        "description": (
            "Drive the desktop with mouse+keyboard guided by a vision model. Slow; use only "
            "when no other tool can do it."
        ),
        "parameters": {
            "type": "object",
            "properties": {"goal": {"type": "string", "description": "What to accomplish."}},
            "required": ["goal"],
        },
    },
}


class ConfirmationRequired(Exception):
    """Raised by a tool handler when the action needs the user's spoken approval."""

    def __init__(self, command: str, reason: str) -> None:
        super().__init__(reason)
        self.command = command
        self.reason = reason


class _ToolFeedback(Exception):
    """Internal signal: short-circuit a handler with a model-actionable, is_error tool
    result — NEVER a voice pause. Used by the read-before-write guardrail (denial) and the
    verdict-feedback loop (reshape-hint) so both mirror the existing policy_block guidance
    convention (an error + guidance payload) without going through ConfirmationRequired.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload.get("error", "tool feedback")))
        self.payload = payload


# Shell commands that only READ a file; their non-flag arguments register as "read this
# task" for the read-before-write guardrail. Deliberately conservative — extending this list
# extends what counts as a "read", so keep it to genuinely read-only inspection commands.
_READ_COMMANDS = frozenset({"cat", "head", "tail", "grep", "rg", "less", "more", "wc"})
_SHELL_CONTROL_TOKENS = frozenset({";", "&&", "||", "|", "&"})

# `>`/`>>` redirect target (stops at the next whitespace/control character).
_REDIRECT_RE = re.compile(r">{1,2}\s*([^\s|;&]+)")
_SED_INPLACE_RE = re.compile(r"\bsed\b[^\n]*-i\b")
_TEE_RE = re.compile(r"\btee\b")


def _extract_read_paths(command: str) -> list[str]:
    """Non-flag path arguments following a read-only command (cat/head/tail/grep/...)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    paths: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _READ_COMMANDS:
            i += 1
            while i < len(tokens) and tokens[i] not in _SHELL_CONTROL_TOKENS:
                if not tokens[i].startswith("-"):
                    paths.append(tokens[i])
                i += 1
            continue
        i += 1
    return paths


def _mutation_targets(command: str) -> list[str]:
    """Paths a shell command would overwrite: `>`/`>>` redirects, `sed -i <file>`, `tee <file>`."""
    targets = [m.group(1) for m in _REDIRECT_RE.finditer(command)]
    if _SED_INPLACE_RE.search(command):
        tokens = [t for t in command.split() if t != "sed" and not t.startswith("-")]
        if tokens:
            targets.append(tokens[-1])
    if _TEE_RE.search(command):
        # \btee\b also matches a path invocation (`/usr/bin/tee`), where no token is the
        # bare word "tee" — locate the token by suffix instead of tokens.index().
        tokens = command.split()
        for idx, tok in enumerate(tokens):
            if tok == "tee" or tok.endswith("/tee"):
                for candidate in tokens[idx + 1 :]:
                    if candidate.startswith("-"):
                        continue
                    targets.append(candidate)
                    break
                break
    return targets


def _reshape_guidance(kind: str, command: str, classifier: Any) -> str:
    """Rerouting hint for the verdict-feedback loop's first non-destructive 'confirm'.

    Only claims the script "runs autonomously" via run_applescript when
    classify_applescript would ACTUALLY allow it — a non-allowlisted app (e.g. Spotify)
    would just pause again after the reshape, so promising autonomy there is misleading.
    """
    if kind == "shell" and re.search(r"\bosascript\b", command):
        extract = getattr(classifier, "_extract_osascript_script", None)
        classify_as = getattr(classifier, "classify_applescript", None)
        script = extract(command) if callable(extract) else None
        if script is not None and callable(classify_as):
            if getattr(classify_as(script), "verdict", "confirm") == "allow":
                return (
                    "osascript via run_shell requires confirmation; the same script via "
                    "run_applescript runs autonomously — reshape and retry."
                )
        elif script is None:
            return (
                "osascript mixed with other shell syntax requires confirmation; if the "
                "AppleScript is the whole job, send just the script via run_applescript "
                "(autonomous only for allowlisted apps) — otherwise ask for confirmation."
            )
    return (
        "this command isn't allowlisted; reshape it to an allowlisted form or ask for confirmation."
    )


@dataclass(frozen=True)
class PendingConfirmation:
    """The paused action a user must approve (relayed by voice) before resume()."""

    call_id: str
    name: str
    command: str
    reason: str


@dataclass
class DelegateResult:
    """Outcome of a delegate run (or the pause point awaiting user confirmation)."""

    status: Literal["done", "awaiting_confirmation", "error"]
    note: str = ""
    interaction_id: str | None = None
    pending: PendingConfirmation | None = None


@dataclass
class _Paused:
    """Loop state frozen while a confirmation is pending."""

    call: Any
    args: dict[str, Any]
    results: list[dict[str, Any]]
    remaining: list[Any]


class _NullLock:
    """Stand-in when no desktop mutex is configured."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


@dataclass
class DelegateAgentConfig:
    """Construction knobs, grouped so TaskManager's agent_factory stays one call."""

    model: str = DEFAULT_MODEL
    api_key_env: str = "GEMINI_API_KEY"
    max_rounds: int = 24
    shell_timeout_s: float = 60.0
    computer_max_steps: int = 24
    computer_model: str = "gemini-3.5-flash"
    # Claude-Code automode analog: 'confirm' (default) pauses on every non-allowlisted /
    # destructive verdict; 'auto' runs benign/allowlisted actions autonomously, but
    # DEFAULT_CONFIRM_PATTERNS (destructive) verdicts ALWAYS escalate to the user in either
    # mode — see require_confirmation.
    safety_mode: Literal["confirm", "auto"] = "confirm"


class DelegateAgent:
    """One goal, one agent instance: stateful over an Interactions chain."""

    def __init__(
        self,
        *,
        config: DelegateAgentConfig | None = None,
        client: Any | None = None,
        tool_handlers: Mapping[str, DelegateToolHandler] | None = None,
        extra_tools: Mapping[str, dict[str, Any]] | None = None,
        desktop_mutex: asyncio.Lock | None = None,
        computer_factory: Callable[[], Computer] = MacOSComputer,
        policy_factory: Callable[[], Policy] | None = None,
        fallback_policy_factory: Callable[[], Policy] | None = None,
        classifier: Any | None = None,
        safety_mode: Literal["confirm", "auto"] | None = None,
    ) -> None:
        self._config = config or DelegateAgentConfig()
        # A directly-passed safety_mode kwarg wins over the config's (mirrors how tests and
        # the TaskManager factory construct agents) — this is the ONE effective mode, never
        # a second parallel gate.
        self._safety_mode: Literal["confirm", "auto"] = safety_mode or self._config.safety_mode
        if client is None:
            from google import genai  # noqa: PLC0415 — lazy; tests inject a fake

            client = genai.Client(api_key=os.environ.get(self._config.api_key_env) or None)
        self._client = client
        self._desktop_mutex: asyncio.Lock | _NullLock = desktop_mutex or _NullLock()
        self._computer_factory = computer_factory
        self._policy_factory = policy_factory or self._default_policy
        self._fallback_policy_factory = fallback_policy_factory or self._default_fallback_policy
        # The Phase-6 CommandSafetyClassifier seam; handlers consult it before executing.
        self._classifier = classifier
        self._handlers: dict[str, DelegateToolHandler] = {
            "run_shell": self._run_shell,
            "run_applescript": self._run_applescript,
            "computer_use": self._computer_use,
        }
        self._specs: dict[str, dict[str, Any]] = dict(_TOOL_SPECS)
        if extra_tools:
            self._specs.update(extra_tools)
        if tool_handlers:
            self._handlers.update(tool_handlers)
        self._interaction_id: str | None = None
        self._paused: _Paused | None = None
        self._bypassing = False
        # Live de-nagging fix (2d): once granted (resume(..., approve_all=True)), persists
        # for the rest of THIS task — subsequent require_confirmation() calls auto-approve
        # unless the reason is a destructive-pattern match, which always pauses individually.
        self._approve_all = False
        # Read-before-write guardrail (2b): resolved paths this task has read via a
        # cat/head/tail/grep/... invocation. Per-agent == per-task (agents are per-task via
        # agent_factory), so no separate task-registry state is needed.
        self._read_paths: set[str] = set()
        # Verdict-feedback loop (3): one reshape retry per logical action. Set when a
        # non-destructive 'confirm' issues a feedback error; cleared on the next 'allow'
        # verdict OR when the retry itself still confirms (falls through to a voice pause).
        self._feedback_retry_pending = False
        # Review fix: every exact command that already received its hint. Without this, an
        # allowed command interleaved between retries of the SAME command reset the pending
        # flag and the command was re-hinted forever (never pausing until max_rounds).
        self._hinted_commands: set[str] = set()

    def _default_policy(self) -> Policy:
        return GeminiComputerUsePolicy(
            model=self._config.computer_model, api_key_env=self._config.api_key_env
        )

    def _default_fallback_policy(self) -> Policy:
        return GeminiVisionLoopPolicy(
            model=self._config.computer_model, api_key_env=self._config.api_key_env
        )

    # -- public entry points ---------------------------------------------------------

    async def run(self, goal: str, context: str = "") -> DelegateResult:
        """Run the agent loop for ``goal`` until done / confirmation needed / max rounds."""
        text = goal if not context else f"{goal}\n\nPointer context: {context}"
        return await self._loop([{"type": "text", "text": text}], first=True)

    async def resume(self, approved: bool, approve_all: bool = False) -> DelegateResult:
        """Continue after a confirmation: re-run the paused action (approved) or decline.

        ``approve_all=True`` grants a persistent bypass for the REST of this task's
        non-destructive confirmations (see ``require_confirmation``); it does not affect
        the CURRENT paused action, which is still governed by ``approved`` alone.
        """
        paused = self._paused
        if paused is None:
            return DelegateResult("error", note="nothing awaiting confirmation")
        self._paused = None
        if approve_all:
            self._approve_all = True
        results = list(paused.results)
        if approved:
            self._bypassing = True
            try:
                results.append(await self._invoke(paused.call, paused.args))
            finally:
                self._bypassing = False
        else:
            results.append(
                self._result_step(
                    paused.call, {"error": "the user declined this action"}, is_error=True
                )
            )
        stop = await self._execute_calls(paused.remaining, results)
        if stop is not None:
            return stop
        return await self._loop(results, first=False)

    # -- the agent loop --------------------------------------------------------------

    async def _loop(self, input_steps: list[dict[str, Any]], *, first: bool) -> DelegateResult:
        all_call_names: list[str] = []
        for _round in range(self._config.max_rounds):
            interaction = await self._create(input_steps, first=first)
            first = False
            self._interaction_id = interaction.id
            calls = [
                s for s in interaction.steps or [] if getattr(s, "type", None) == "function_call"
            ]
            if not calls:
                return DelegateResult(
                    "done", note=_final_text(interaction), interaction_id=self._interaction_id
                )
            all_call_names.extend(str(getattr(c, "name", "?")) for c in calls)
            results: list[dict[str, Any]] = []
            stop = await self._execute_calls(calls, results)
            if stop is not None:
                return stop
            input_steps = results
        return await self._wrap_up(input_steps, all_call_names)

    async def _wrap_up(
        self, input_steps: list[dict[str, Any]], recent_call_names: list[str]
    ) -> DelegateResult:
        """Tool budget exhausted: ask for one honest, tool-free summary instead of erroring
        out blind. If the model still tries to call a tool on this final round, fall back to
        the plain "max rounds reached" error, naming the last few tool calls for
        diagnosability. If the wrap-up request itself fails, fall back the same way — we
        cannot get an honest summary either way."""
        wrap_text = (
            "Tool budget exhausted — no more tool calls; summarize honestly what was and was "
            "not accomplished and why."
        )
        combined = [*input_steps, {"type": "text", "text": wrap_text}]
        try:
            interaction = await self._create(combined, first=False)
        except Exception:  # noqa: BLE001 — no honest summary available; fall back below
            return DelegateResult(
                "error", note="max rounds reached", interaction_id=self._interaction_id
            )
        self._interaction_id = interaction.id
        calls = [s for s in interaction.steps or [] if getattr(s, "type", None) == "function_call"]
        if calls:
            recent_call_names.extend(str(getattr(c, "name", "?")) for c in calls)
            names = ", ".join(recent_call_names[-5:])
            return DelegateResult(
                "error",
                note=f"max rounds reached; model still tried to call tools ({names})",
                interaction_id=self._interaction_id,
            )
        note = _final_text(interaction)
        return DelegateResult(
            "done",
            note=f"{note} (incomplete — ran out of tool budget)",
            interaction_id=self._interaction_id,
        )

    async def _execute_calls(
        self, calls: list[Any], results: list[dict[str, Any]]
    ) -> DelegateResult | None:
        """Run each call, appending result steps; return a pause result on confirmation."""
        for index, call in enumerate(calls):
            args = dict(call.arguments or {})
            try:
                results.append(await self._invoke(call, args))
            except ConfirmationRequired as need:
                self._paused = _Paused(call, args, results, list(calls[index + 1 :]))
                return DelegateResult(
                    "awaiting_confirmation",
                    note=f"needs the user's confirmation: {need.reason} ({need.command})",
                    interaction_id=self._interaction_id,
                    pending=PendingConfirmation(
                        str(call.id), str(call.name), need.command, need.reason
                    ),
                )
        return None

    def _task_tag(self) -> str:
        return self._interaction_id or "-"

    def _log_invocation(self, call: Any, args: dict[str, Any]) -> None:
        """INFO-log every tool call with its key argument, for live-log diagnosability."""
        name = getattr(call, "name", "?")
        if name == "run_shell":
            detail = str(args.get("command", ""))
        elif name == "run_applescript":
            detail = " ".join(str(args.get("script", "")).split())
        elif name == "computer_use":
            detail = str(args.get("goal", ""))
        else:
            detail = repr(args)
        logger.info("[delegate task=%s] invoking %s: %s", self._task_tag(), name, detail[:200])

    def _log_error_result(self, call: Any, payload: Any) -> None:
        """INFO-log a tool result that came back as an error, so failures are diagnosable
        from the live log even when the model keeps going without surfacing them."""
        name = getattr(call, "name", "?")
        logger.info("[delegate task=%s] %s result error: %s", self._task_tag(), name, payload)

    async def _invoke(self, call: Any, args: dict[str, Any]) -> dict[str, Any]:
        self._log_invocation(call, args)
        handler = self._handlers.get(call.name)
        if handler is None:
            logger.warning("[delegate] model called unknown tool %r", call.name)
            return self._result_step(
                call, {"error": f"unsupported tool {call.name!r}"}, is_error=True
            )
        try:
            outcome = await handler(args)
        except ConfirmationRequired:
            raise
        except _ToolFeedback as feedback:
            self._log_error_result(call, feedback.payload)
            return self._result_step(call, feedback.payload, is_error=True)
        except Exception as exc:  # noqa: BLE001 — report tool failure to the model, keep going
            logger.warning("[delegate] tool %s failed: %s", call.name, exc)
            payload: dict[str, Any] = {"error": str(exc)}
            if is_policy_block(exc):
                # Observed live: Gemini's computer_use input filter hard-blocks e.g. Gmail
                # automation with a 400 "Input blocked". Retrying the same approach loops
                # forever — tell the model explicitly that this path is closed.
                payload["policy_block"] = True
                payload["guidance"] = (
                    "The provider refused this action for policy reasons. Do NOT retry this "
                    "approach. Either try ONE genuinely different tool (run_applescript, "
                    "browser_*) or finish now and report the limitation honestly."
                )
            return self._result_step(call, payload, is_error=True)
        is_error_outcome = isinstance(outcome, Mapping) and (
            "error" in outcome or outcome.get("status") == "error"
        )
        if is_error_outcome:
            self._log_error_result(call, outcome)
        return self._result_step(call, outcome)

    @staticmethod
    def _result_step(call: Any, payload: Any, *, is_error: bool = False) -> dict[str, Any]:
        import json  # noqa: PLC0415

        step: dict[str, Any] = {
            "type": "function_result",
            "name": call.name,
            "call_id": call.id,
            "result": [{"type": "text", "text": json.dumps(payload, default=str)}],
        }
        if is_error:
            step["is_error"] = True
        return step

    async def _create(self, input_steps: list[dict[str, Any]], *, first: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "input": input_steps,
            "tools": [
                {"type": "function", "name": name, **spec} for name, spec in self._specs.items()
            ],
        }
        if first:
            kwargs["system_instruction"] = _SYSTEM
        if self._interaction_id is not None:
            kwargs["previous_interaction_id"] = self._interaction_id
        # google-genai's async Interactions client warns on the first `.aio` access when the
        # aiohttp extra isn't installed ("... falling back to httpx"); httpx is our actual,
        # intended transport here, so this is expected, not actionable — silence just this
        # message at the source rather than letting it spam the log on every delegate task.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Async interactions client cannot use aiohttp",
                category=UserWarning,
            )
            interactions = self._client.aio.interactions
        return await interactions.create(**kwargs)

    # -- confirmation seam -----------------------------------------------------------

    def require_confirmation(self, command: str, reason: str) -> None:
        """Raise ConfirmationRequired unless approved (this resume, or a standing approve_all).

        Destructive-pattern reasons always pause individually, even under approve_all — see
        ``_is_destructive_reason``.
        """
        if self._bypassing:
            return
        if self._approve_all and not _is_destructive_reason(reason):
            return
        if self._safety_mode == "auto" and not _is_destructive_reason(reason):
            return
        raise ConfirmationRequired(command, reason)

    def _classify(self, kind: str, command: str) -> None:
        """Consult the classifier; non-destructive 'confirm' verdicts get ONE reshape retry
        (the verdict-feedback loop, deliverable 3) before falling through to a voice pause.

        Destructive verdicts, auto-mode bypasses, and a resumed (bypassing) re-invocation all
        skip the feedback loop entirely and route straight through require_confirmation —
        which is the single place that decides whether this actually pauses.
        """
        if self._classifier is None:
            return
        # AppleScript gets its own verdict path: multi-line tell blocks are normal there,
        # so the shell compound-command rule must not force a confirmation every time.
        classify = (
            getattr(self._classifier, "classify_applescript", None)
            if kind == "applescript"
            else None
        ) or self._classifier.classify
        decision = classify(command)
        if getattr(decision, "verdict", "allow") != "confirm":
            self._feedback_retry_pending = False
            return
        reason = f"{kind}: {decision.reason}"
        if self._bypassing or self._safety_mode == "auto" or _is_destructive_reason(reason):
            self._feedback_retry_pending = False
            self.require_confirmation(command, reason)
            return
        if not self._feedback_retry_pending and command not in self._hinted_commands:
            self._feedback_retry_pending = True
            self._hinted_commands.add(command)
            raise _ToolFeedback(
                {
                    "error": f"{reason} — requires confirmation",
                    "guidance": _reshape_guidance(kind, command, self._classifier),
                }
            )
        self._feedback_retry_pending = False
        self.require_confirmation(command, reason)

    def _check_read_before_write(self, command: str) -> None:
        """Read-before-write guardrail (2b), active in BOTH safety modes. A mutation
        targeting a file that EXISTS but hasn't been read this task is blocked with a
        model-recoverable denial — never a voice pause. New (non-existent) paths pass."""
        for target in _mutation_targets(command):
            resolved = os.path.abspath(os.path.expanduser(target))
            if resolved.startswith("/dev/"):
                # Device paths (e.g. `2>/dev/null`, `>/dev/null`) are never "read first" —
                # they aren't real files with content to inspect. Exempt them unconditionally.
                continue
            if os.path.exists(resolved) and resolved not in self._read_paths:
                raise _ToolFeedback(
                    {
                        "error": f"refusing to write {target!r} — it has not been read this task",
                        "guidance": (
                            f"you must read the file first, e.g. `cat {target}`, then retry "
                            "the write"
                        ),
                    }
                )

    def _register_read_paths(self, command: str) -> None:
        for path in _extract_read_paths(command):
            self._read_paths.add(os.path.abspath(os.path.expanduser(path)))

    # -- built-in tool handlers ------------------------------------------------------

    async def _run_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args.get("command") or "").strip()
        if not command:
            return {"error": "empty command"}
        self._check_read_before_write(command)
        self._register_read_paths(command)
        self._classify("shell", command)
        proc = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        return await self._communicate(proc)

    async def _run_applescript(self, args: dict[str, Any]) -> dict[str, Any]:
        script = str(args.get("script") or "").strip()
        if not script:
            return {"error": "empty script"}
        self._classify("applescript", script)
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await self._communicate(proc)

    async def _communicate(self, proc: asyncio.subprocess.Process) -> dict[str, Any]:
        timeout = self._config.shell_timeout_s
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {"status": "timeout", "error": f"timed out after {timeout}s"}
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace")[-4000:],
            "stderr": stderr.decode(errors="replace")[-2000:],
        }

    async def _computer_use(self, args: dict[str, Any]) -> dict[str, Any]:
        """The general fallback: whole-goal desktop drive, serialized + off the event loop.

        The mutex is held on the loop for the WHOLE nested run (two concurrent tasks must
        never interleave mouse events); the run itself goes to a worker thread because the
        policy uses the sync Interactions/generateContent client and ``screencapture``
        blocks. Block-detection and the hosted-tool -> vision-loop retry decision itself are
        delegated to :func:`wrap_with_vision_loop_fallback` — the SAME helper the live
        duplex model's ``computer_use`` tool handler uses, so there is one implementation of
        "hosted policy input-blocked -> retry with GeminiVisionLoopPolicy" instead of two.
        """
        goal = str(args.get("goal") or "").strip()
        if not goal:
            return {"error": "empty goal"}
        # asyncio.to_thread cannot cancel the thread; on cancellation (barge-in, shutdown)
        # set the stop flag so the executor quits at its next tick instead of continuing
        # to drive the real mouse as an orphan.
        stop = threading.Event()

        async def _run(policy: Policy) -> Any:
            return await asyncio.to_thread(
                run_computer_use,
                goal,
                self._computer_factory(),
                policy,
                self._config.computer_max_steps,
                stop.is_set,
            )

        async with self._desktop_mutex:
            try:
                result, via_fallback = await wrap_with_vision_loop_fallback(
                    _run, self._policy_factory, self._fallback_policy_factory
                )
            except asyncio.CancelledError:
                stop.set()
                raise
        payload: dict[str, Any] = {
            "status": "ok" if result.done else "incomplete",
            "steps": result.steps,
            "note": result.final_note,
        }
        if via_fallback:
            payload["via"] = "vision_loop_fallback"
        return payload


def _is_destructive_reason(reason: str) -> bool:
    """True when a confirmation reason came from a DEFAULT_CONFIRM_PATTERNS match.

    ``CommandSafetyClassifier`` always phrases those as "matches a destructive pattern
    (...)"; approve_all must never silently wave those through.
    """
    return "destructive pattern" in reason.lower()


def _final_text(interaction: Any) -> str:
    texts: list[str] = []
    for step in interaction.steps or []:
        if getattr(step, "type", None) == "model_output":
            for content in step.content or []:
                text = getattr(content, "text", None)
                if text:
                    texts.append(str(text))
    return " ".join(texts).strip() or "task complete"
