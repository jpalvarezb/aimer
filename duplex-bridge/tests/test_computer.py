"""Computer-use seam: the cross-application perceive->decide->act loop (deterministic)."""

from __future__ import annotations

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
