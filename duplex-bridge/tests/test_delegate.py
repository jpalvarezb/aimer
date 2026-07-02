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

from duplex_bridge.actions.computer import Action, FakeComputer
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
    assert "You are in /Users/jp." == result.note
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
