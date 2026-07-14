"""DelegateAgent — deterministic tests over a fake async Interactions client.

Mirrors test_computer_policy.py's fake-client pattern (async variant): canned interactions,
recorded requests, no network. Covers handler dispatch + function_result round-trips,
interaction-id chaining, the nested computer_use fallback against FakeComputer, and the
confirmation pause/resume mechanic.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from duplex_bridge.actions.computer import Action, FakeComputer, Policy
from duplex_bridge.actions.delegate import (
    ConfirmationRequired,
    DelegateAgent,
    DelegateAgentConfig,
)


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
