"""Computer-use seam: the cross-application perceive->decide->act loop (deterministic)."""

from __future__ import annotations

import asyncio

import pytest
from duplex_bridge.actions import TOOL_DECLARATIONS
from duplex_bridge.actions.computer import (
    Action,
    ComputerUseExecutor,
    FakeComputer,
    MacOSComputer,
)


def _scripted(actions: list[Action]):
    it = iter(actions)

    def policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        try:
            return next(it)
        except StopIteration:
            return Action("done", note="exhausted")

    return policy


# --- executor loop --------------------------------------------------------


async def test_executor_runs_sequence_then_done():
    comp = FakeComputer()
    policy = _scripted(
        [Action("click", x=10, y=20), Action("type", text="hello"), Action("done", note="ok")]
    )
    res = await ComputerUseExecutor(comp, policy).run("do the thing")
    assert res.done and res.final_note == "ok" and res.steps == 2
    assert [a.type for a in comp.calls] == ["click", "type"]
    assert comp.shots == 3  # one screenshot before each policy decision


async def test_executor_respects_max_steps():
    comp = FakeComputer()
    res = await ComputerUseExecutor(
        comp, lambda g, s, h: Action("click", x=1, y=1), max_steps=4
    ).run("loop forever")
    assert not res.done and res.steps == 4 and len(comp.calls) == 4


async def test_executor_stops_when_should_stop_flags():
    # Cancellation/shutdown can't kill the worker thread the run lives in — the stop
    # flag must halt the loop at the next tick so an orphan run stops driving the mouse.
    comp = FakeComputer()
    flags = iter([False, False, True])
    clicker = _scripted([Action("click", x=1, y=1)] * 10)
    executor = ComputerUseExecutor(comp, clicker, max_steps=10, should_stop=lambda: next(flags))
    res = await executor.run("loop forever")
    assert not res.done and res.steps == 2 and res.final_note == "stopped by caller"
    assert len(comp.calls) == 2


async def test_executor_supports_async_policy():
    comp = FakeComputer()

    async def policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        return Action("done", note="async") if history else Action("key", keys=("cmd", "s"))

    res = await ComputerUseExecutor(comp, policy).run("save")
    assert res.done and comp.calls == [Action("key", keys=("cmd", "s"))]


# --- primitive dispatch ---------------------------------------------------


def test_apply_dispatches_every_primitive():
    comp = FakeComputer()
    for a in (
        Action("click", x=1, y=2),
        Action("double_click", x=3, y=4),
        Action("move", x=5, y=6),
        Action("type", text="x"),
        Action("key", keys=("cmd", "a")),
        Action("scroll", x=0, y=0, dy=-5),
        Action("screenshot"),  # no-op
        Action("done"),  # no-op
    ):
        comp.apply(a)
    assert [c.type for c in comp.calls] == [
        "click",
        "double_click",
        "move",
        "type",
        "key",
        "scroll",
    ]


def test_apply_rejects_unknown_action():
    with pytest.raises(ValueError, match="unknown action"):
        FakeComputer().apply(Action("teleport"))


# --- macOS impl + tool decl ----------------------------------------------


def test_macos_computer_constructs_without_importing_quartz():
    # Lazy PyObjC import: constructing must not require Quartz (import happens on first use).
    MacOSComputer()


def test_computer_use_is_declared_as_a_general_tool():
    assert "computer_use" in {d["name"] for d in TOOL_DECLARATIONS}


# --- live-fix (4b): run_computer_use_with_timeout — the tool call must ALWAYS return -------
#
# Live finding: two Teams-click computer_use calls never produced a final result — the
# hosted policy dangled forever with no per-goal wall-clock timeout, so the tool call sat
# silent. A timeout must turn a dangling run into an explicit error result, never a hang.


def test_run_computer_use_with_timeout_returns_error_result_never_hangs():
    # Sync entry point (mirrors run_computer_use — wraps its own asyncio.run so it is safe
    # to call via asyncio.to_thread from the caller's own running event loop); called here
    # directly from a plain sync test, matching how the worker thread invokes it live.
    from duplex_bridge.actions.computer import run_computer_use_with_timeout

    comp = FakeComputer()

    async def _hanging_policy(goal: str, shot: bytes, history: list[Action]) -> Action:
        await asyncio.sleep(9999)
        return Action("done")  # pragma: no cover — never reached

    result = run_computer_use_with_timeout(
        "goal that never finishes", comp, _hanging_policy, max_steps=50, timeout_s=0.05
    )

    assert result.done is False
    assert "timed out" in result.final_note.lower()


def test_run_computer_use_with_timeout_returns_normally_when_the_run_finishes_in_time():
    from duplex_bridge.actions.computer import run_computer_use_with_timeout

    comp = FakeComputer()
    policy = _scripted([Action("click", x=1, y=1), Action("done", note="ok")])

    result = run_computer_use_with_timeout("do it", comp, policy, max_steps=5, timeout_s=10.0)

    assert result.done is True
    assert result.final_note == "ok"


# --- live-fix (4c): click_pointer — deterministic click on what the user is pointing at ----


def test_click_pointer_issues_exactly_one_click_with_no_policy_invocations():
    from duplex_bridge.actions.computer import click_pointer

    comp = FakeComputer()
    policy_invocations = {"count": 0}

    def _policy_should_never_run(goal: str, shot: bytes, history: list[Action]) -> Action:
        policy_invocations["count"] += 1
        return Action("done")

    result = click_pointer(comp, 123, 456, "the blue Submit button")

    assert comp.calls == [Action("click", x=123, y=456)]
    assert comp.shots == 0  # no screenshot round-trip — this is a direct actuation
    assert policy_invocations["count"] == 0
    assert "the blue Submit button" in result
