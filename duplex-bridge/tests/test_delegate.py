"""DelegateAgent — deterministic tests over a fake async Interactions client.

Mirrors test_computer_policy.py's fake-client pattern (async variant): canned interactions,
recorded requests, no network. Covers handler dispatch + function_result round-trips,
interaction-id chaining, the nested computer_use fallback against FakeComputer, and the
confirmation pause/resume mechanic.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from duplex_bridge.actions.computer import Action, FakeComputer, Policy
from duplex_bridge.actions.delegate import (
    ConfirmationRequired,
    DelegateAgent,
    DelegateAgentConfig,
)
from duplex_bridge.actions.safety import CommandSafetyClassifier


def _call(name: str, call_id: str = "c1", **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", id=call_id, name=name, arguments=arguments)


def _interaction(interaction_id: str, *steps: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(id=interaction_id, steps=list(steps))


def _text_output(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="model_output", content=[SimpleNamespace(text=text)])


class _FakeAsyncClient:
    """Duck-typed genai.Client with async interactions.create; records every request."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        outer = self

        class _Interactions:
            async def create(self, **kwargs: Any) -> SimpleNamespace:
                outer.requests.append(kwargs)
                return outer.responses.pop(0)

        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.aio = SimpleNamespace(interactions=_Interactions())


def _agent(responses: list[SimpleNamespace], **kwargs: Any) -> DelegateAgent:
    return DelegateAgent(client=_FakeAsyncClient(responses), **kwargs)


async def test_dispatches_to_handler_and_round_trips_result() -> None:
    seen: list[dict[str, Any]] = []

    async def _fake_shell(args: dict[str, Any]) -> dict[str, Any]:
        seen.append(args)
        return {"status": "ok", "stdout": "/Users/jp"}

    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c9", command="pwd")),
            _interaction("i2", _text_output("You are in /Users/jp.")),
        ],
        tool_handlers={"run_shell": _fake_shell},
    )
    result = await agent.run("where am I?")

    assert result.status == "done"
    assert result.note == "You are in /Users/jp."
    assert seen == [{"command": "pwd"}]
    client: Any = agent._client
    first, second = client.requests
    assert first["system_instruction"]  # system prompt only on the first create
    assert "system_instruction" not in second
    assert second["previous_interaction_id"] == "i1"
    (step,) = second["input"]
    assert step["type"] == "function_result"
    assert step["call_id"] == "c9"
    assert '"/Users/jp"' in step["result"][0]["text"]


async def test_declared_tools_include_registered_handlers() -> None:
    agent = _agent([_interaction("i1", _text_output("hi"))])
    await agent.run("g")
    client: Any = agent._client
    names = {t["name"] for t in client.requests[0]["tools"]}
    assert {"run_shell", "run_applescript", "computer_use"} <= names
    assert all(t["type"] == "function" for t in client.requests[0]["tools"])


async def test_computer_use_fallback_drives_executor_with_fake_computer() -> None:
    """The nested computer_use handler reuses the Week-7b executor + policy unchanged."""
    script = iter([Action("click", x=10, y=20), Action("done", note="clicked it")])

    def _scripted_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        return next(script)

    fake_computer = FakeComputer()
    mutex = asyncio.Lock()
    agent = _agent(
        [
            _interaction("i1", _call("computer_use", goal="click the icon")),
            _interaction("i2", _text_output("done")),
        ],
        computer_factory=lambda: fake_computer,
        policy_factory=lambda: _scripted_policy,
        desktop_mutex=mutex,
    )
    result = await agent.run("click the icon")

    assert result.status == "done"
    assert fake_computer.calls == [Action("click", x=10, y=20)]
    assert not mutex.locked()


async def test_computer_use_fallback_runs_vision_loop_when_hosted_policy_is_input_blocked() -> None:
    """Hosted GeminiComputerUsePolicy raising an 'Input blocked' error retries the SAME
    goal via the generateContent vision-loop fallback, tagging the result 'via', with no
    policy_block in the tool result (the caller-visible failure mode is avoided)."""

    def _blocked_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        raise RuntimeError("Error code: 400 - {'error': {'message': 'Input blocked: nope'}}")

    fallback_script = iter([Action("click", x=1, y=1), Action("done", note="done via fallback")])

    def _fallback_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        return next(fallback_script)

    fake_computer = FakeComputer()
    mutex = asyncio.Lock()
    agent = _agent(
        [
            _interaction("i1", _call("computer_use", goal="open mail")),
            _interaction("i2", _text_output("done")),
        ],
        computer_factory=lambda: fake_computer,
        policy_factory=lambda: _blocked_policy,
        fallback_policy_factory=lambda: _fallback_policy,
        desktop_mutex=mutex,
    )
    result = await agent.run("open mail")

    assert result.status == "done"
    assert not mutex.locked()
    assert fake_computer.calls == [Action("click", x=1, y=1)]
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert "is_error" not in step
    text = step["result"][0]["text"]
    assert '"via": "vision_loop_fallback"' in text
    assert "policy_block" not in text


async def test_computer_use_fallback_runs_vision_loop_on_soft_safety_block() -> None:
    """A 'done' result whose note carries the soft safety_decision block text (no
    exception) must trigger the same fallback as the hard 'Input blocked' exception."""

    def _soft_blocked_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        return Action("done", note="blocked by the model's safety policy: payment page")

    def _fallback_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        return Action("done", note="done via fallback")

    agent = _agent(
        [
            _interaction("i1", _call("computer_use", goal="pay the bill")),
            _interaction("i2", _text_output("done")),
        ],
        computer_factory=FakeComputer,
        policy_factory=lambda: _soft_blocked_policy,
        fallback_policy_factory=lambda: _fallback_policy,
    )
    result = await agent.run("pay the bill")

    assert result.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    text = step["result"][0]["text"]
    assert '"via": "vision_loop_fallback"' in text
    assert "policy_block" not in text


async def test_computer_use_no_fallback_when_hosted_policy_succeeds() -> None:
    """The fallback factory must never even be constructed when the hosted policy works."""

    def _ok_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        return Action("done", note="all good")

    def _fallback_factory() -> Policy:
        raise AssertionError("fallback policy must not be constructed on the happy path")

    agent = _agent(
        [
            _interaction("i1", _call("computer_use", goal="click ok")),
            _interaction("i2", _text_output("done")),
        ],
        computer_factory=FakeComputer,
        policy_factory=lambda: _ok_policy,
        fallback_policy_factory=_fallback_factory,
    )
    result = await agent.run("click ok")

    assert result.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    text = step["result"][0]["text"]
    assert "via" not in text


async def test_computer_use_both_policies_blocked_reports_policy_block() -> None:
    """When the fallback ALSO fails with a block, its exception propagates unchanged so
    the existing _invoke policy_block handling still fires (no regression)."""

    def _blocked_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        raise RuntimeError("Error code: 400 - Input blocked: first attempt")

    def _also_blocked_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        raise RuntimeError("Error code: 400 - Input blocked: fallback too")

    agent = _agent(
        [
            _interaction("i1", _call("computer_use", goal="read private notes")),
            _interaction("i2", _text_output("reported the limitation")),
        ],
        computer_factory=FakeComputer,
        policy_factory=lambda: _blocked_policy,
        fallback_policy_factory=lambda: _also_blocked_policy,
    )
    result = await agent.run("read private notes")

    assert result.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert step["is_error"] is True
    text = step["result"][0]["text"]
    assert '"policy_block": true' in text


async def test_unknown_tool_reports_error_and_continues() -> None:
    agent = _agent(
        [
            _interaction("i1", _call("teleport", call_id="c3")),
            _interaction("i2", _text_output("fine, done another way")),
        ]
    )
    result = await agent.run("g")
    assert result.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert step["is_error"] is True
    assert "teleport" in step["result"][0]["text"]


async def test_handler_exception_reports_error_and_continues() -> None:
    async def _boom(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("browser crashed")

    agent = _agent(
        [
            _interaction("i1", _call("run_shell", command="ls")),
            _interaction("i2", _text_output("gave up gracefully")),
        ],
        tool_handlers={"run_shell": _boom},
    )
    result = await agent.run("g")
    assert result.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert step["is_error"] is True
    assert "browser crashed" in step["result"][0]["text"]


async def test_provider_policy_block_gets_do_not_retry_guidance() -> None:
    """A Gemini 'Input blocked' 400 must come back marked non-retryable (observed live:
    the delegate retried a blocked Gmail automation for minutes without this)."""

    async def _blocked(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "Error code: 400 - {'error': {'message': 'Input blocked: The user is requesting "
            "unauthorized access to private email content', 'code': 'invalid_request'}}"
        )

    agent = _agent(
        [
            _interaction("i1", _call("computer_use", goal="read gmail")),
            _interaction("i2", _text_output("reported the limitation")),
        ],
        tool_handlers={"computer_use": _blocked},
    )
    result = await agent.run("g")
    assert result.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert step["is_error"] is True
    assert '"policy_block": true' in step["result"][0]["text"]
    assert "Do NOT retry" in step["result"][0]["text"]


async def test_max_rounds_yields_error_status() -> None:
    responses = [
        _interaction(f"i{n}", _call("run_shell", call_id=f"c{n}", command="ls")) for n in range(3)
    ]

    async def _ok(args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok"}

    agent = _agent(
        responses,
        tool_handlers={"run_shell": _ok},
        config=DelegateAgentConfig(max_rounds=3),
    )
    result = await agent.run("g")
    assert result.status == "error"
    assert "max rounds" in result.note


async def test_confirmation_pauses_and_resume_approved_executes() -> None:
    """A handler raising ConfirmationRequired pauses the run; resume(True) re-invokes it
    with the bypass set and continues the SAME interaction chain."""
    executed: list[str] = []

    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c5", command="rm -rf /tmp/x")),
            _interaction("i2", _text_output("deleted")),
        ]
    )

    async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
        agent.require_confirmation(args["command"], "destructive delete")
        executed.append(args["command"])
        return {"status": "ok"}

    agent._handlers["run_shell"] = _guarded

    paused = await agent.run("delete the scratch dir")
    assert paused.status == "awaiting_confirmation"
    assert paused.pending is not None
    assert paused.pending.command == "rm -rf /tmp/x"
    assert executed == []  # never ran

    final = await agent.resume(approved=True)
    assert final.status == "done"
    assert executed == ["rm -rf /tmp/x"]
    client: Any = agent._client
    assert client.requests[1]["previous_interaction_id"] == "i1"  # same chain


async def test_resume_declined_sends_error_result_and_continues() -> None:
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c6", command="sudo reboot")),
            _interaction("i2", _text_output("okay, I won't")),
        ]
    )

    async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
        raise ConfirmationRequired(args["command"], "requires sudo")

    agent._handlers["run_shell"] = _guarded

    paused = await agent.run("reboot my machine")
    assert paused.status == "awaiting_confirmation"
    final = await agent.resume(approved=False)
    assert final.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert step["is_error"] is True
    assert "declined" in step["result"][0]["text"]


async def test_real_run_shell_handler_executes_commands() -> None:
    """The built-in run_shell handler really runs a (harmless) command."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", command="echo aimer-delegate")),
            _interaction("i2", _text_output("echoed")),
        ]
    )
    result = await agent.run("echo something")
    assert result.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert "aimer-delegate" in step["result"][0]["text"]
    assert '"status": "ok"' in step["result"][0]["text"]


async def test_classifier_pauses_builtin_run_shell_and_resume_executes() -> None:
    """Phase 6 integration: the real run_shell handler + CommandSafetyClassifier pause on a
    destructive command; approval resumes and actually executes it (harmless no-op rm)."""
    from duplex_bridge.actions.safety import CommandSafetyClassifier

    command = "rm -rf /nonexistent-aimer-safety-test"
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command=command)),
            _interaction("i2", _text_output("cleaned up")),
        ],
        classifier=CommandSafetyClassifier(),
    )
    paused = await agent.run("clean the scratch dir")
    assert paused.status == "awaiting_confirmation"
    assert paused.pending is not None and paused.pending.command == command
    assert "destructive" in paused.pending.reason

    final = await agent.resume(approved=True)
    assert final.status == "done"
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert '"status": "ok"' in step["result"][0]["text"]  # really executed after approval


async def test_classifier_allows_allowlisted_builtin_run_shell() -> None:
    from duplex_bridge.actions.safety import CommandSafetyClassifier

    agent = _agent(
        [
            _interaction("i1", _call("run_shell", command="echo safety-allow-path")),
            _interaction("i2", _text_output("echoed")),
        ],
        classifier=CommandSafetyClassifier(),
    )
    result = await agent.run("say hi")
    assert result.status == "done"  # never paused
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert "safety-allow-path" in step["result"][0]["text"]


def test_classify_routes_applescript_to_its_own_policy() -> None:
    """AppleScript verdicts come from classify_applescript: benign multi-line tell blocks
    run autonomously (the shell compound rule stalled every app action — live 2026-07-03),
    destructive scripts still pause, and shell keeps the compound rule."""
    import pytest
    from duplex_bridge.actions.safety import CommandSafetyClassifier

    agent = _agent([], classifier=CommandSafetyClassifier())
    benign = 'tell application "Notes"\n\tmake new note with properties {name:"x"}\nend tell'
    agent._classify("applescript", benign)  # must not raise

    with pytest.raises(ConfirmationRequired):
        agent._classify("applescript", 'tell application "Notes" to delete note 1')
    with pytest.raises(ConfirmationRequired):
        agent._classify("shell", "ls; rm -rf ~")  # compound rule still guards shell


# --- live-fix (2d): DelegateAgent.resume(approve_all=True) — persistent per-task bypass ----


async def test_resume_approve_all_bypasses_subsequent_non_destructive_confirmations() -> None:
    executed: list[str] = []
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="step 1")),
            _interaction("i2", _call("run_shell", call_id="c2", command="step 2")),
            _interaction("i3", _text_output("done")),
        ]
    )

    async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
        agent.require_confirmation(args["command"], "not on the allowlist")
        executed.append(args["command"])
        return {"status": "ok"}

    agent._handlers["run_shell"] = _guarded

    paused = await agent.run("do two things")
    assert paused.status == "awaiting_confirmation"

    final = await agent.resume(approved=True, approve_all=True)
    assert final.status == "done"
    # step 2's confirmation was auto-approved by the approve_all bypass, never paused again.
    assert executed == ["step 1", "step 2"]


async def test_resume_approve_all_still_pauses_for_destructive_reason() -> None:
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="rm -rf /a")),
            _interaction("i2", _call("run_shell", call_id="c2", command="rm -rf /b")),
            _interaction("i3", _text_output("done")),
        ]
    )

    async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
        agent.require_confirmation(args["command"], "matches a destructive pattern (rm)")
        return {"status": "ok"}

    agent._handlers["run_shell"] = _guarded

    paused = await agent.run("delete two things")
    assert paused.status == "awaiting_confirmation"

    still_paused = await agent.resume(approved=True, approve_all=True)
    assert still_paused.status == "awaiting_confirmation"
    assert still_paused.pending is not None
    assert still_paused.pending.command == "rm -rf /b"


async def test_resume_without_approve_all_does_not_set_a_bypass() -> None:
    """Plain resume(approved=True) (no approve_all) must NOT leave a bypass behind — the
    next non-destructive confirmation in the same task still pauses."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="step 1")),
            _interaction("i2", _call("run_shell", call_id="c2", command="step 2")),
            _interaction("i3", _text_output("done")),
        ]
    )

    async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
        agent.require_confirmation(args["command"], "not on the allowlist")
        return {"status": "ok"}

    agent._handlers["run_shell"] = _guarded

    await agent.run("do two things")
    still_paused = await agent.resume(approved=True)
    assert still_paused.status == "awaiting_confirmation"
    assert still_paused.pending is not None
    assert still_paused.pending.command == "step 2"


# --- live-fix (5a): _SYSTEM routing guidance ------------------------------------------------


def test_system_prompt_includes_new_routing_guidance() -> None:
    from duplex_bridge.actions.delegate import _SYSTEM

    lower = _SYSTEM.lower()
    # browser_* for reading, headless, never a visible test-browser window
    assert "browser_" in _SYSTEM
    assert "headless" in lower
    # `open <url>` via run_shell when the user should SEE the page
    assert "open <url>" in _SYSTEM
    # prefer read-only AppleScript queries before falling back to computer_use
    assert "read-only" in lower
    assert "applescript" in lower and "computer_use" in _SYSTEM
    # ask the user once before a long run of similar shell steps, not per-command
    assert "ask" in lower and "once" in lower


# --- live-fix (5b): silence the aiohttp-fallback UserWarning at Interactions client use ----


async def test_aiohttp_fallback_warning_is_suppressed_around_first_interactions_access(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """google-genai's async Interactions client warns 'cannot use aiohttp, falling back to
    httpx' the first time `.aio` is touched. Live finding: this fired on every delegate task,
    spamming the log. It must be caught with warnings.catch_warnings (not blanket-ignored
    elsewhere), so a fake client that emits it on `.aio` access must leave recwarn empty."""
    import warnings as warnings_module

    class _WarningEmittingClient:
        def __init__(self, responses: list[SimpleNamespace]) -> None:
            self._responses = responses
            self.requests: list[dict[str, Any]] = []
            self.aio_accesses = 0

        @property
        def aio(self) -> SimpleNamespace:
            self.aio_accesses += 1
            warnings_module.warn(
                "Async interactions client cannot use aiohttp, falling back to httpx",
                UserWarning,
                stacklevel=2,
            )
            outer = self

            class _Interactions:
                async def create(self, **kwargs: Any) -> SimpleNamespace:
                    outer.requests.append(kwargs)
                    return outer._responses.pop(0)

            return SimpleNamespace(interactions=_Interactions())

    client = _WarningEmittingClient([_interaction("i1", _text_output("done"))])
    agent = DelegateAgent(client=client)

    result = await agent.run("do something")

    assert result.status == "done"
    assert client.aio_accesses >= 1  # the warning-triggering path really executed
    assert len(recwarn) == 0  # suppressed at the source, not merely unobserved


# --- live-fix (4a): _computer_use routes through the shared fallback helper ----------------


async def test_computer_use_fallback_uses_the_shared_wrap_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """delegate._computer_use must delegate its block-detection/retry logic to the shared
    computer_policy.wrap_with_vision_loop_fallback helper (one implementation shared with
    __main__.py's live computer_use path), not a bespoke inline copy."""
    import duplex_bridge.actions.delegate as delegate_module

    calls: list[tuple[Any, Any]] = []

    async def _fake_wrap(
        run: Any, primary_factory: Any, fallback_factory: Any, **kwargs: Any
    ) -> Any:
        calls.append((primary_factory, fallback_factory))
        return await run(primary_factory)

    monkeypatch.setattr(
        delegate_module, "wrap_with_vision_loop_fallback", _fake_wrap, raising=False
    )

    def _ok_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        return Action("done", note="all good")

    agent = _agent(
        [
            _interaction("i1", _call("computer_use", goal="click ok")),
            _interaction("i2", _text_output("done")),
        ],
        computer_factory=FakeComputer,
        policy_factory=lambda: _ok_policy,
    )
    result = await agent.run("click ok")

    assert result.status == "done"
    assert len(calls) == 1  # the shared helper was actually invoked, not bypassed


# --- safety-overhaul (2): --delegate-safety {confirm,auto} — Claude-Code automode analog ---


async def test_auto_mode_executes_non_allowlisted_benign_command_without_confirmation() -> None:
    """DelegateAgent(safety_mode='auto') runs benign, non-allowlisted commands autonomously —
    only a DEFAULT_CONFIRM_PATTERNS (destructive) hit is ever allowed to pause the task."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="true")),
            _interaction("i2", _text_output("done")),
        ],
        classifier=CommandSafetyClassifier(),
        safety_mode="auto",
    )
    result = await agent.run("run a benign non-allowlisted command")
    assert result.status == "done"  # zero ConfirmationRequired pauses
    client: Any = agent._client
    (step,) = client.requests[1]["input"]
    assert "is_error" not in step
    assert '"status": "ok"' in step["result"][0]["text"]


async def test_auto_mode_still_escalates_destructive_commands() -> None:
    """Auto mode never silently runs a DEFAULT_CONFIRM_PATTERNS hit — destructive actions
    ALWAYS escalate to the user via the existing voice ConfirmationRequired flow, exactly
    like Claude Code's accept-edits mode still prompting on harmful bash."""
    agent = _agent(
        [
            _interaction(
                "i1", _call("run_shell", call_id="c1", command="rm -rf /tmp/aimer-safety-auto-x")
            ),
            _interaction("i2", _text_output("done")),
        ],
        classifier=CommandSafetyClassifier(),
        safety_mode="auto",
    )
    result = await agent.run("delete the scratch dir")
    assert result.status == "awaiting_confirmation"
    assert result.pending is not None
    assert result.pending.command == "rm -rf /tmp/aimer-safety-auto-x"
    assert "destructive" in result.pending.reason


# --- safety-overhaul (2): read-before-write guardrail — Claude-Code edit-requires-read rule -


async def test_read_before_write_guardrail_blocks_unread_mutation_and_recovers(
    tmp_path: Path,
) -> None:
    """A file-mutating shell command targeting a file the task has NOT previously read is
    blocked with an actionable, model-recoverable error (an is_error function_result, never
    a ConfirmationRequired pause). Reading the file earlier in the same task — or targeting a
    brand-new (non-existent) path — allows the mutation straight through. Active in BOTH
    safety modes (no classifier/safety_mode override — this is a separate guardrail)."""
    existing = tmp_path / "notes.txt"
    existing.write_text("original")
    fresh = tmp_path / "brand_new.txt"

    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command=f"echo new > {existing}")),
            _interaction("i2", _call("run_shell", call_id="c2", command=f"cat {existing}")),
            _interaction("i3", _call("run_shell", call_id="c3", command=f"echo new > {existing}")),
            _interaction("i4", _call("run_shell", call_id="c4", command=f"echo hi > {fresh}")),
            # verify-before-done (2b) requires a read-back of both post-c3/c4 mutations
            # before the task can finalize — a single read of both satisfies it.
            _interaction("i5", _call("run_shell", call_id="c5", command=f"cat {existing} {fresh}")),
            _interaction("i6", _text_output("done")),
        ]
    )

    result = await agent.run("edit notes.txt")
    assert result.status == "done"  # a recoverable tool error, never a confirmation pause

    client: Any = agent._client
    (blocked_step,) = client.requests[1]["input"]
    assert blocked_step["is_error"] is True
    assert "read the file first" in blocked_step["result"][0]["text"].lower()
    # (is_error + the guidance text above are the proof the block took effect — the guardrail
    # raises before ever invoking the subprocess for a blocked mutation, see
    # DelegateAgent._run_shell. A live disk read here would be checked only after the whole
    # run — including c3's later, spec-required post-read write of this same path — has
    # already completed, so it can't distinguish "c1 never ran" from "c1 ran then c3 reran it".)

    (read_step,) = client.requests[2]["input"]
    assert "is_error" not in read_step
    assert "original" in read_step["result"][0]["text"]

    (write_after_read_step,) = client.requests[3]["input"]
    assert "is_error" not in write_after_read_step
    assert existing.read_text().strip() == "new"  # this write really ran, post-read

    (write_new_file_step,) = client.requests[4]["input"]
    assert "is_error" not in write_new_file_step
    assert fresh.exists()
    assert fresh.read_text().strip() == "hi"  # a brand-new path needs no prior read


# --- safety-overhaul (3): verdict-feedback loop — one reshape retry before a voice pause ---


async def test_verdict_feedback_loop_reshape_then_success() -> None:
    """The first non-destructive 'confirm' verdict comes back as a reshape-hint tool error
    (no pause); a reshaped retry that classifies 'allow' just executes."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="python deploy.py --prod")),
            _interaction("i2", _call("run_shell", call_id="c2", command="echo redeploying")),
            _interaction("i3", _text_output("redeployed")),
        ],
        classifier=CommandSafetyClassifier(),
    )
    result = await agent.run("redeploy")
    assert result.status == "done"  # never paused for user confirmation
    client: Any = agent._client
    (first_step,) = client.requests[1]["input"]
    assert first_step["is_error"] is True
    text = first_step["result"][0]["text"].lower()
    assert "reshape" in text or "retry" in text  # actionable rerouting hint, not a bare denial
    (second_step,) = client.requests[2]["input"]
    assert "is_error" not in second_step
    assert "redeploying" in second_step["result"][0]["text"]


async def test_verdict_feedback_loop_second_consecutive_confirm_escalates() -> None:
    """Exactly ONE reshape retry per logical action: if the reshaped attempt ALSO classifies
    'confirm', it falls through to the existing ConfirmationRequired voice pause."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="python deploy.py --prod")),
            _interaction("i2", _call("run_shell", call_id="c2", command="node deploy.js --prod")),
        ],
        classifier=CommandSafetyClassifier(),
    )
    result = await agent.run("redeploy")
    assert result.status == "awaiting_confirmation"
    assert result.pending is not None
    assert result.pending.command == "node deploy.js --prod"
    client: Any = agent._client
    (first_step,) = client.requests[1]["input"]
    assert first_step["is_error"] is True  # round 1's confirm was a reshape hint, not a pause


@pytest.mark.parametrize("safety_mode", ["confirm", "auto"])
async def test_destructive_confirm_verdict_skips_feedback_loop(safety_mode: str) -> None:
    """A destructive-pattern 'confirm' verdict must go straight to ConfirmationRequired, with
    NO preceding reshape-error round, in EVERY safety mode — the feedback loop must never
    intercept destructive verdicts (safety must never be weakened)."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="rm -rf /tmp/x")),
            _interaction("i2", _text_output("done")),
        ],
        classifier=CommandSafetyClassifier(),
        safety_mode=safety_mode,
    )
    result = await agent.run("delete the scratch dir")
    assert result.status == "awaiting_confirmation"
    assert result.pending is not None
    assert result.pending.command == "rm -rf /tmp/x"
    client: Any = agent._client
    assert len(client.requests) == 1  # paused on round 1 — no reshape round happened first


async def test_verdict_feedback_loop_same_command_after_interleaved_allow_escalates() -> None:
    """Review fix: a command that already received its reshape hint must NOT be hinted again
    just because an allowed command ran in between — re-submitting the identical command
    pauses for the user instead of looping on hints until max_rounds."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="python deploy.py --prod")),
            _interaction("i2", _call("run_shell", call_id="c2", command="echo checking")),
            _interaction("i3", _call("run_shell", call_id="c3", command="python deploy.py --prod")),
        ],
        classifier=CommandSafetyClassifier(),
    )
    result = await agent.run("redeploy")
    assert result.status == "awaiting_confirmation"
    assert result.pending is not None
    assert result.pending.command == "python deploy.py --prod"


async def test_reshape_guidance_does_not_claim_autonomy_for_non_allowlisted_app() -> None:
    """Review fix: the run_applescript rerouting hint must only claim the script 'runs
    autonomously' when classify_applescript would actually allow it — scripting a
    non-allowlisted app (Spotify) would just pause again after the reshape."""
    agent = _agent(
        [
            _interaction(
                "i1",
                _call(
                    "run_shell",
                    call_id="c1",
                    command="osascript -e 'tell application \"Spotify\" to play'",
                ),
            ),
            _interaction("i2", _text_output("ok, asking first")),
        ],
        classifier=CommandSafetyClassifier(),
    )
    result = await agent.run("play music")
    assert result.status == "done"
    client: Any = agent._client
    (first_step,) = client.requests[1]["input"]
    assert first_step["is_error"] is True
    assert "runs autonomously" not in first_step["result"][0]["text"]


def test_mutation_targets_handles_pathed_tee() -> None:
    """Review fix: `/usr/bin/tee` matched \\btee\\b but tokens.index("tee") raised ValueError,
    silently skipping the read-before-write guardrail for path-invoked tee."""
    from duplex_bridge.actions.delegate import _mutation_targets

    assert _mutation_targets("echo x | /usr/bin/tee /tmp/out.txt") == ["/tmp/out.txt"]


# --- safety-overhaul: --delegate-safety CLI flag ---------------------------------------


def test_build_parser_accepts_delegate_safety_and_defaults_to_confirm() -> None:
    import duplex_bridge.__main__ as bridge_main

    parser = bridge_main.build_parser()

    args = parser.parse_args([])
    assert args.delegate_safety == "confirm"

    args = parser.parse_args(["--delegate-safety", "auto"])
    assert args.delegate_safety == "auto"

    with pytest.raises(SystemExit):
        parser.parse_args(["--delegate-safety", "bogus"])


async def test_delegate_safety_flag_threads_into_delegate_agent_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--delegate-safety threads through async_main's agent factory into the
    DelegateAgentConfig each delegated task's DelegateAgent is built with."""
    import argparse

    import duplex_bridge.__main__ as bridge_main

    class _FakeSession:
        def __init__(self, **_kwargs: Any) -> None: ...
        async def open(self) -> None: ...
        def on_tool_call(self, callback: Any) -> None: ...
        def on_tool_call_cancellation(self, callback: Any) -> None: ...
        def set_resume_context_provider(self, provider: Any) -> None: ...
        async def close(self) -> None: ...

    class _FakeServer:
        def __init__(self, **_kwargs: Any) -> None:
            self.port = 8765

        async def start(self) -> None: ...
        async def stop(self) -> None: ...

    class _FakeEvent:
        async def wait(self) -> None:
            return None

    class _RecordingDelegateAgent:
        last_config: Any = None

        def __init__(self, **kwargs: Any) -> None:
            _RecordingDelegateAgent.last_config = kwargs.get("config")

    captured_factory: dict[str, Any] = {}

    class _RecordingTaskManager:
        def __init__(self, agent_factory: Any, **_kwargs: Any) -> None:
            captured_factory["factory"] = agent_factory

        def pending_note(self) -> str:
            return ""

        async def confirm(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"status": "done"}

    class _RecordingDelegateBrowser:
        def __init__(self, **_kwargs: Any) -> None: ...
        def handlers_for_task(self, task_id: str) -> dict[str, Any]:
            return {}

        async def close_page(self, task_id: str) -> None: ...
        async def aclose(self) -> None: ...

    class _RecordingDispatcher:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.registered: dict[str, Any] = {}

        def register(self, name: str, handler: Any) -> None:
            self.registered[name] = handler

        def dispatch(self, *_args: Any, **_kwargs: Any) -> None: ...
        def cancel(self, *_args: Any, **_kwargs: Any) -> None: ...

    monkeypatch.setattr(bridge_main, "GeminiLiveSession", _FakeSession)
    monkeypatch.setattr(bridge_main, "WebSocketContextServer", _FakeServer)
    monkeypatch.setattr(bridge_main.asyncio, "Event", _FakeEvent)
    monkeypatch.setattr(bridge_main, "DelegateAgent", _RecordingDelegateAgent)
    monkeypatch.setattr(bridge_main, "TaskManager", _RecordingTaskManager)
    monkeypatch.setattr(bridge_main, "DelegateBrowser", _RecordingDelegateBrowser)
    monkeypatch.setattr(bridge_main, "ToolDispatcher", lambda *a, **kw: _RecordingDispatcher())

    namespace = argparse.Namespace(
        host="127.0.0.1",
        port=8765,
        gemini_model="test-model",
        api_key_env="GEMINI_API_KEY",
        no_audio=True,
        audio_backend="sounddevice",
        audio_activity_rms_threshold=300.0,
        vad_silence_ms=None,
        vad_start_sensitivity=None,
        turn_coverage=None,
        manual_vad=False,
        end_of_turn_silence_ms=400,
        onset_speech_ms=250,
        thinking_level=None,
        push_to_talk=False,
        ptt_key="cmd_r",
        escalate_full_frame=False,
        deixis_model="gemini-flash-lite-latest",
        no_deixis_resolver=True,
        computer_use_model="gemini-3.5-flash",
        computer_use_max_steps=24,
        delegate_safety="auto",
    )
    await bridge_main.async_main(namespace)

    factory = captured_factory["factory"]
    factory("task-1")
    config = _RecordingDelegateAgent.last_config
    assert config is not None
    assert config.safety_mode == "auto"


# --- goal-fidelity fix: delegate_task description, verification prompt, tool-call logging ----


def test_delegate_task_goal_description_requires_verbatim_details() -> None:
    """The live model must be told to carry every user-specified detail into the goal
    verbatim — the delegate agent never hears the user, so anything left out is lost.

    Regression: 2026-07-14 live smoke, the user asked for a titled note and the live model
    delegated goal="open Notes and create a new note" — the title was silently dropped.
    """
    from duplex_bridge.actions import TOOL_DECLARATIONS

    delegate_decl = next(d for d in TOOL_DECLARATIONS if d["name"] == "delegate_task")
    goal_desc = delegate_decl["parameters"]["properties"]["goal"]["description"]
    lower = goal_desc.lower()
    assert "verbatim" in lower
    assert "title" in lower


def test_system_prompt_includes_verification_and_notes_guidance() -> None:
    """The delegate agent's system prompt must require a read-back verification before
    reporting success, and must explicitly call out the macOS Notes first-line-title quirk
    (Notes derives the title from the first line of the body, not the `name` property alone).

    Regression: 2026-07-14 live smoke, the delegate claimed success after one unverified
    AppleScript call that did not actually set the note's title.
    """
    from duplex_bridge.actions.delegate import _SYSTEM

    lower = _SYSTEM.lower()
    assert "verify" in lower
    assert "read-back" in lower or "read back" in lower
    assert "never claim success" in lower
    assert "notes" in lower
    assert "first line" in lower


async def test_invoke_logs_run_shell_call_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """Every delegate tool invocation must be diagnosable from the live log — INFO, naming
    the tool and a snippet of what it actually ran (2026-07-14 smoke gave zero visibility)."""

    async def _fake_shell(args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "stdout": ""}

    agent = _agent(
        [
            _interaction("i1", _call("run_shell", command="echo distinctive-log-marker-42")),
            _interaction("i2", _text_output("done")),
        ],
        tool_handlers={"run_shell": _fake_shell},
    )
    caplog.set_level("INFO", logger="duplex_bridge.actions.delegate")
    result = await agent.run("say hello")
    assert result.status == "done"

    info_records = [
        r
        for r in caplog.records
        if r.name == "duplex_bridge.actions.delegate" and r.levelname == "INFO"
    ]
    assert any(
        "run_shell" in r.getMessage() and "distinctive-log-marker-42" in r.getMessage()
        for r in info_records
    )


async def test_invoke_logs_run_applescript_call_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A multi-line AppleScript is flattened to one line in the log so it stays readable."""

    async def _fake_applescript(args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "stdout": ""}

    script = (
        'tell application "Notes"\n'
        '  make new note with properties {name:"DISTINCTIVE-MARKER-99"}\n'
        "end tell"
    )
    agent = _agent(
        [
            _interaction("i1", _call("run_applescript", script=script)),
            _interaction("i2", _text_output("done")),
        ],
        tool_handlers={"run_applescript": _fake_applescript},
    )
    caplog.set_level("INFO", logger="duplex_bridge.actions.delegate")
    result = await agent.run("make a note")
    assert result.status == "done"

    info_records = [
        r
        for r in caplog.records
        if r.name == "duplex_bridge.actions.delegate" and r.levelname == "INFO"
    ]
    matching = [
        r
        for r in info_records
        if "run_applescript" in r.getMessage() and "DISTINCTIVE-MARKER-99" in r.getMessage()
    ]
    assert matching
    # flattened to one line for log readability — no embedded newline
    assert all("\n" not in r.getMessage() for r in matching)


async def test_invoke_logs_error_result(caplog: pytest.LogCaptureFixture) -> None:
    """A tool result that comes back as an error (including the internal _ToolFeedback
    short-circuit path) must be visible in the log at INFO or higher, not silently
    swallowed — only handler *exceptions* were logged before this fix."""
    from duplex_bridge.actions.delegate import _ToolFeedback

    async def _feedback_handler(args: dict[str, Any]) -> dict[str, Any]:
        raise _ToolFeedback({"error": "refusing to write 'x' — it has not been read this task"})

    agent = _agent(
        [
            _interaction("i1", _call("run_shell", command="echo hi > x")),
            _interaction("i2", _text_output("done")),
        ],
        tool_handlers={"run_shell": _feedback_handler},
    )
    caplog.set_level("INFO", logger="duplex_bridge.actions.delegate")
    result = await agent.run("write to x")
    assert result.status == "done"

    records = [
        r
        for r in caplog.records
        if r.name == "duplex_bridge.actions.delegate" and r.levelno >= logging.INFO
    ]
    assert any("run_shell" in r.getMessage() and "error" in r.getMessage().lower() for r in records)


# --- 2026-07-14 22:09 live-smoke fix B1: /dev/null read-before-write exemption -----------


def test_read_before_write_exempts_dev_null_python_stderr_redirect() -> None:
    """`python3 -c "..." 2>/dev/null` must never trigger the 'refusing to write' denial —
    /dev/null is a device, not a real file that can be "read first"."""
    agent = _agent([])
    agent._check_read_before_write('python3 -c "print(1)" 2>/dev/null')  # must not raise


def test_read_before_write_exempts_dev_null_stdout_redirect() -> None:
    agent = _agent([])
    agent._check_read_before_write("echo x >/dev/null")  # must not raise


def test_read_before_write_still_denies_unread_real_file(tmp_path: Path) -> None:
    """A real, existing, unread file under a redirect is still denied — the /dev/null
    exemption must not weaken the guardrail for anything else."""
    from duplex_bridge.actions.delegate import _ToolFeedback

    existing = tmp_path / "notes.txt"
    existing.write_text("original")
    agent = _agent([])
    with pytest.raises(_ToolFeedback):
        agent._check_read_before_write(f"echo new > {existing}")


# --- 2026-07-14 22:09 live-smoke fix B2: date fidelity -----------------------------------


def test_delegate_task_goal_description_pins_verbatim_relative_dates() -> None:
    """live smoke task-3: 'delete all notes created today' got delegated with a fabricated
    absolute date. The goal description must instruct the live model to pass relative time
    expressions verbatim, never converted to absolute dates."""
    from duplex_bridge.actions import TOOL_DECLARATIONS

    (delegate_spec,) = [t for t in TOOL_DECLARATIONS if t["name"] == "delegate_task"]
    description = delegate_spec["parameters"]["properties"]["goal"]["description"].lower()

    assert "today" in description or "relative" in description
    assert "verbatim" in description
    assert "absolute date" in description or "convert" in description


def test_system_pins_date_reconciliation_rule() -> None:
    """_SYSTEM must instruct the delegate to resolve relative dates with the system clock
    (the `date` command) and trust the clock over a contradicting absolute date in the goal,
    noting the discrepancy rather than hunting for data matching the wrong date."""
    from duplex_bridge.actions.delegate import _SYSTEM

    lowered = _SYSTEM.lower()
    assert "`date`" in lowered or "the date command" in lowered
    assert "system clock" in lowered
    assert "discrepancy" in lowered or "note" in lowered


# --- 2026-07-14 22:09 live-smoke fix B3: honest max-rounds wrap-up ------------------------


async def test_max_rounds_wrap_up_round_returns_honest_incomplete_summary() -> None:
    """When the round budget is nearly exhausted, the agent sends one final wrap-up
    instruction (no more tool calls; summarize honestly) instead of erroring out blind. If
    the model complies with a text-only response, the task completes with an honest,
    clearly-incomplete summary rather than the bare 'max rounds reached' error."""
    responses = [
        _interaction(f"i{n}", _call("run_shell", call_id=f"c{n}", command="ls")) for n in range(3)
    ] + [_interaction("i3", _text_output("Ran out of time; deleted 2 of 5 notes."))]

    async def _ok(args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok"}

    agent = _agent(
        responses,
        tool_handlers={"run_shell": _ok},
        config=DelegateAgentConfig(max_rounds=3),
    )
    result = await agent.run("g")

    client: Any = agent._client
    # One extra _create call beyond the 3 budgeted rounds carries the wrap-up instruction.
    assert len(client.requests) == 4
    final_input = str(client.requests[-1]["input"]).lower()
    assert "budget exhausted" in final_input or "no more tool calls" in final_input
    assert "summarize" in final_input or "honestly" in final_input

    # A normal completion, not the opaque error path — but clearly marked incomplete.
    assert result.status == "done"
    assert "budget" in result.note.lower() or "incomplete" in result.note.lower()


async def test_max_rounds_wrap_up_round_still_calls_tool_falls_back_to_error_with_names() -> None:
    """If the model still tries to call a tool on the wrap-up round, fall back to the error
    path — but the message must name the last few tool calls made, for diagnosability."""
    responses = [
        _interaction("i0", _call("run_shell", call_id="c0", command="ls")),
        _interaction("i1", _call("run_applescript", call_id="c1", script="tell app x")),
        _interaction("i2", _call("run_shell", call_id="c2", command="pwd")),
        _interaction("i3", _call("run_shell", call_id="c3", command="rm x")),
    ]

    async def _ok(args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok"}

    agent = _agent(
        responses,
        tool_handlers={"run_shell": _ok, "run_applescript": _ok},
        config=DelegateAgentConfig(max_rounds=3),
    )
    result = await agent.run("g")

    assert result.status == "error"
    assert "max rounds" in result.note.lower()
    assert "run_shell" in result.note
    assert "run_applescript" in result.note


# --- 2026-07-14 22:09 live-smoke fix B4: scratch files go under a temp directory ----------


def test_system_pins_scratch_file_tempdir_rule() -> None:
    """_SYSTEM must tell the delegate its shell runs in a private scratch directory and
    that relative paths land there, never in the user's project folders (2026-08-05 live
    smoke: the agent wrote ocr.swift into the repo root despite the old ask-nicely rule —
    the cwd is now ENFORCED by _run_shell, and the prompt must describe that reality)."""
    from duplex_bridge.actions.delegate import _SYSTEM

    lowered = _SYSTEM.lower()
    assert "scratch" in lowered
    assert "relative paths" in lowered
    assert "project folders" in lowered


# --- scratch-cwd enforcement (2026-08-05 live smoke) ----------------------------------------
#
# The delegate wrote ocr.swift/screenshot.png into the bridge's repo-root cwd despite the
# prompt forbidding it. Prompt-only guardrails don't bind; _run_shell now executes every
# command with cwd set to a per-task private scratch directory.


async def test_run_shell_executes_in_private_scratch_cwd(tmp_path: Path) -> None:
    """Relative writes land in a per-task aimer-delegate-* scratch dir, not the process cwd."""
    agent = _agent([])
    pwd = await agent._run_shell({"command": "pwd"})
    assert pwd["status"] == "ok"
    scratch = pwd["stdout"].strip()
    assert "aimer-delegate-" in scratch
    assert Path(scratch).resolve() != Path.cwd().resolve()

    write = await agent._run_shell({"command": "echo hi > relative.txt"})
    assert write["status"] == "ok"
    assert (Path(scratch) / "relative.txt").exists()
    assert not (Path.cwd() / "relative.txt").exists()


async def test_run_shell_scratch_cwd_is_stable_within_a_task() -> None:
    """Consecutive commands share one scratch dir, so multi-step file work composes."""
    agent = _agent([])
    first = await agent._run_shell({"command": "pwd"})
    second = await agent._run_shell({"command": "pwd"})
    assert first["stdout"] == second["stdout"]


async def test_run_shell_scratch_cwd_differs_between_tasks() -> None:
    """Each agent (== one task) gets its own scratch dir — no cross-task file collisions."""
    a = await _agent([])._run_shell({"command": "pwd"})
    b = await _agent([])._run_shell({"command": "pwd"})
    assert a["stdout"] != b["stdout"]


async def test_read_before_write_guardrail_resolves_relative_paths_against_scratch() -> None:
    """The guardrail must judge relative paths where the command actually runs (scratch),
    not the bridge process cwd: a file created relatively then overwritten without a read
    is blocked, and reading it back clears the pending-verify entry for the SAME path."""
    from duplex_bridge.actions.delegate import _ToolFeedback

    agent = _agent([])
    created = await agent._run_shell({"command": "echo one > guarded.txt"})
    assert created["status"] == "ok"
    # Overwriting the now-existing relative file without reading it first must be blocked —
    # which only happens if the existence check resolved into the scratch dir.
    with pytest.raises(_ToolFeedback):
        await agent._run_shell({"command": "echo two > guarded.txt"})
    # Reading it back clears pending-verify for the same resolved path.
    read = await agent._run_shell({"command": "cat guarded.txt"})
    assert read["stdout"].strip() == "one"
    assert not agent._pending_verify_paths


# --- demo-blocker fix (2b): deterministic verify-before-done — forced read-back round -------
#
# 2026-07-17 live run: task-1 skipped the read-back and reported done on an unverified write.
# The 2026-07-23 run DID read back — so today this is intermittent (the _SYSTEM prompt text
# asks nicely but nothing enforces it). These tests pin a deterministic guardrail: a task-wide
# mutation with no subsequent read is never allowed to finalize as "done" on the first
# zero-tool-call round — the agent must force exactly one extra round demanding read-back
# before it can finalize.


async def test_verify_before_done_forces_readback_of_unread_mutation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mutating run_shell call followed by a premature zero-tool-call 'done' must NOT
    finalize — the agent forces one extra round demanding read-back of the unread mutated
    target; once the model reads it back and finishes, the result finalizes as done."""
    target = tmp_path / "output.txt"
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command=f"echo new > {target}")),
            _interaction("i2", _text_output("Wrote the file.")),  # premature — mutation unread
            _interaction("i3", _call("run_shell", call_id="c3", command=f"cat {target}")),
            _interaction("i4", _text_output("Verified: output.txt now contains 'new'.")),
        ]
    )
    caplog.set_level("INFO", logger="duplex_bridge.actions.delegate")

    result = await agent.run("write to output.txt")

    assert result.status == "done"
    assert "Verified" in result.note
    assert target.read_text().strip() == "new"

    client: Any = agent._client
    # write -> premature-done attempt -> forced verify round -> real read-back -> final done.
    assert len(client.requests) == 4
    forced_round_input = str(client.requests[2]["input"]).lower()
    assert "output.txt" in forced_round_input or str(target).lower() in forced_round_input
    assert "read" in forced_round_input or "verify" in forced_round_input

    forced_logs = [
        r
        for r in caplog.records
        if r.name == "duplex_bridge.actions.delegate" and "forc" in r.getMessage().lower()
    ]
    assert forced_logs
    assert forced_logs[0].levelname == "INFO"


async def test_verify_before_done_no_forced_round_without_mutations() -> None:
    """No mutations this task: the first zero-tool-call round finalizes immediately — no
    forced extra round, no false positives on read-only work."""
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command="pwd")),
            _interaction("i2", _text_output("You are in /Users/jp.")),
        ]
    )
    result = await agent.run("where am I?")
    assert result.status == "done"
    client: Any = agent._client
    assert len(client.requests) == 2  # no forced verify round


async def test_verify_before_done_finalizes_after_exactly_one_forced_round_if_model_refuses(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If the model still refuses to read back on the forced round, the agent finalizes
    anyway (no infinite loop) after exactly one forced round, logging the unverified state."""
    target = tmp_path / "output.txt"
    agent = _agent(
        [
            _interaction("i1", _call("run_shell", call_id="c1", command=f"echo new > {target}")),
            _interaction("i2", _text_output("Wrote the file.")),  # premature — mutation unread
            _interaction("i3", _text_output("Already wrote it, that's enough.")),  # refuses
        ]
    )
    caplog.set_level("INFO", logger="duplex_bridge.actions.delegate")

    result = await agent.run("write to output.txt")

    assert result.status == "done"  # finalizes anyway — never loops forever
    client: Any = agent._client
    assert len(client.requests) == 3  # exactly one forced round, not a second

    unverified_logs = [
        r
        for r in caplog.records
        if r.name == "duplex_bridge.actions.delegate"
        and ("unverif" in r.getMessage().lower() or "forc" in r.getMessage().lower())
    ]
    assert unverified_logs
    assert unverified_logs[0].levelname == "INFO"
