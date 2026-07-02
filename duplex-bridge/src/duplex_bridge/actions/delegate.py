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
                           ``screencapture``; the loop must never run on the event loop)

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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .computer import Computer, MacOSComputer, Policy, run_computer_use
from .computer_policy import GeminiComputerUsePolicy

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
fits. Ground deictic goals ("this", "that") in the pointer context. Do not ask questions — \
make the reasonable choice and note it. Finish with a one-sentence summary of what you did.\
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
        classifier: Any | None = None,
    ) -> None:
        self._config = config or DelegateAgentConfig()
        if client is None:
            from google import genai  # noqa: PLC0415 — lazy; tests inject a fake

            client = genai.Client(api_key=os.environ.get(self._config.api_key_env) or None)
        self._client = client
        self._desktop_mutex: asyncio.Lock | _NullLock = desktop_mutex or _NullLock()
        self._computer_factory = computer_factory
        self._policy_factory = policy_factory or self._default_policy
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

    def _default_policy(self) -> Policy:
        return GeminiComputerUsePolicy(
            model=self._config.computer_model, api_key_env=self._config.api_key_env
        )

    # -- public entry points ---------------------------------------------------------

    async def run(self, goal: str, context: str = "") -> DelegateResult:
        """Run the agent loop for ``goal`` until done / confirmation needed / max rounds."""
        text = goal if not context else f"{goal}\n\nPointer context: {context}"
        return await self._loop([{"type": "text", "text": text}], first=True)

    async def resume(self, approved: bool) -> DelegateResult:
        """Continue after a confirmation: re-run the paused action (approved) or decline."""
        paused = self._paused
        if paused is None:
            return DelegateResult("error", note="nothing awaiting confirmation")
        self._paused = None
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
            results: list[dict[str, Any]] = []
            stop = await self._execute_calls(calls, results)
            if stop is not None:
                return stop
            input_steps = results
        return DelegateResult(
            "error", note="max rounds reached", interaction_id=self._interaction_id
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

    async def _invoke(self, call: Any, args: dict[str, Any]) -> dict[str, Any]:
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
        except Exception as exc:  # noqa: BLE001 — report tool failure to the model, keep going
            logger.warning("[delegate] tool %s failed: %s", call.name, exc)
            return self._result_step(call, {"error": str(exc)}, is_error=True)
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
        return await self._client.aio.interactions.create(**kwargs)

    # -- confirmation seam -----------------------------------------------------------

    def require_confirmation(self, command: str, reason: str) -> None:
        """Raise ConfirmationRequired unless this invoke is a user-approved resume."""
        if not self._bypassing:
            raise ConfirmationRequired(command, reason)

    def _classify(self, kind: str, command: str) -> None:
        if self._classifier is None:
            return
        decision = self._classifier.classify(command)
        if getattr(decision, "verdict", "allow") == "confirm":
            self.require_confirmation(command, f"{kind}: {decision.reason}")

    # -- built-in tool handlers ------------------------------------------------------

    async def _run_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args.get("command") or "").strip()
        if not command:
            return {"error": "empty command"}
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
        policy uses the sync Interactions client and ``screencapture`` blocks.
        """
        goal = str(args.get("goal") or "").strip()
        if not goal:
            return {"error": "empty goal"}
        async with self._desktop_mutex:
            result = await asyncio.to_thread(
                run_computer_use,
                goal,
                self._computer_factory(),
                self._policy_factory(),
                self._config.computer_max_steps,
            )
        return {
            "status": "ok" if result.done else "incomplete",
            "steps": result.steps,
            "note": result.final_note,
        }


def _final_text(interaction: Any) -> str:
    texts: list[str] = []
    for step in interaction.steps or []:
        if getattr(step, "type", None) == "model_output":
            for content in step.content or []:
                text = getattr(content, "text", None)
                if text:
                    texts.append(str(text))
    return " ".join(texts).strip() or "task complete"
