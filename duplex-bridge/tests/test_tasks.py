"""TaskManager — concurrency, the desktop mutex, and the confirm flow.

Fake async Interactions clients per agent (test_delegate.py's pattern); the serialization
proof records which task's actions hit a SHARED FakeComputer and asserts they never
interleave, while independent slow tools from two tasks run with gather-level parallelism.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace
from typing import Any

from duplex_bridge.actions.computer import Action, FakeComputer
from duplex_bridge.actions.delegate import DelegateAgent
from duplex_bridge.actions.tasks import TaskManager


def _call(name: str, call_id: str = "c1", **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", id=call_id, name=name, arguments=arguments)


def _interaction(interaction_id: str, *steps: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(id=interaction_id, steps=list(steps))


def _text_output(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="model_output", content=[SimpleNamespace(text=text)])


class _FakeAsyncClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        outer = self

        class _Interactions:
            async def create(self, **kwargs: Any) -> SimpleNamespace:
                outer.requests.append(kwargs)
                return outer.responses.pop(0)

        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.aio = SimpleNamespace(interactions=_Interactions())


def _computer_use_responses() -> list[SimpleNamespace]:
    return [
        _interaction("i1", _call("computer_use", goal="drive")),
        _interaction("i2", _text_output("done driving")),
    ]


async def test_two_computer_use_tasks_never_interleave_on_the_desktop_mutex() -> None:
    mutex = asyncio.Lock()
    shared_computer = FakeComputer()

    def _scripted_policy(base: int):
        # FakeComputer records only coordinates, so tasks are told apart by x-range.
        state = {"step": 0}

        def _policy(goal: str, shot: bytes, history: list[Action]) -> Action:
            time.sleep(0.01)  # runs in to_thread — real interleaving chance without the mutex
            state["step"] += 1
            if state["step"] > 3:
                return Action("done")
            return Action("click", x=base + state["step"], y=0)

        return _policy

    def _factory(task_id: str) -> DelegateAgent:
        base = 100 * int(task_id.split("-")[1])  # task-1 -> 100s, task-2 -> 200s
        return DelegateAgent(
            client=_FakeAsyncClient(_computer_use_responses()),
            computer_factory=lambda: shared_computer,
            policy_factory=lambda: _scripted_policy(base),
            desktop_mutex=mutex,
        )

    manager = TaskManager(_factory)
    await asyncio.gather(manager.run("task a"), manager.run("task b"))

    owners = [(action.x or 0) // 100 for action in shared_computer.calls]
    assert len(owners) == 6  # 3 clicks per task
    switches = sum(1 for i in range(1, len(owners)) if owners[i] != owners[i - 1])
    assert switches == 1  # one contiguous block per task — never interleaved


async def test_independent_tasks_run_concurrently() -> None:
    async def _slow_tool(args: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return {"status": "ok"}

    def _factory(task_id: str) -> DelegateAgent:
        return DelegateAgent(
            client=_FakeAsyncClient(
                [
                    _interaction("i1", _call("run_shell", command="sleep")),
                    _interaction("i2", _text_output("slept")),
                ]
            ),
            tool_handlers={"run_shell": _slow_tool},
        )

    manager = TaskManager(_factory)
    start = time.perf_counter()
    results = await asyncio.gather(manager.run("a"), manager.run("b"), manager.run("c"))
    wall = time.perf_counter() - start

    assert all(r["status"] == "done" for r in results)
    assert wall < 0.45  # ~max(0.2), not sum(0.6) — the tasks overlapped


async def test_confirmation_flow_through_manager() -> None:
    def _factory(task_id: str) -> DelegateAgent:
        agent = DelegateAgent(
            client=_FakeAsyncClient(
                [
                    _interaction("i1", _call("run_shell", call_id="c1", command="rm -rf /x")),
                    _interaction("i2", _text_output("deleted it")),
                ]
            )
        )

        async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
            agent.require_confirmation(args["command"], "destructive delete")
            return {"status": "ok"}

        agent._handlers["run_shell"] = _guarded
        return agent

    manager = TaskManager(_factory)
    paused = await manager.run("delete x")
    assert paused["status"] == "awaiting_confirmation"
    assert paused["action"] == "rm -rf /x"
    assert "confirm_task" in paused["message"]
    task_id = paused["task_id"]

    summary = await manager.summarize()
    assert summary["tasks"][0]["status"] == "awaiting_confirmation"

    final = await manager.confirm(task_id, approved=True)
    assert final["status"] == "done"
    assert final["note"] == "deleted it"
    summary = await manager.summarize()
    assert summary["tasks"][0]["status"] == "done"


async def test_confirmation_budget_aborts_runaway_task() -> None:
    """A task that pauses for confirmation on every step gets aborted, not looped forever.

    Observed live: a delegate driving always-confirm UI scripting paused ~40 times while
    the live model rubber-stamped each one — the budget turns that into a hard stop.
    """
    from duplex_bridge.actions.tasks import MAX_CONFIRMATIONS_PER_TASK

    responses = [
        _interaction(f"i{n}", _call("run_shell", call_id=f"c{n}", command=f"step {n}"))
        for n in range(MAX_CONFIRMATIONS_PER_TASK + 2)
    ]

    def _factory(task_id: str) -> DelegateAgent:
        agent = DelegateAgent(client=_FakeAsyncClient(responses))

        async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
            agent.require_confirmation(args["command"], "privileged step")
            return {"status": "ok"}

        agent._handlers["run_shell"] = _guarded
        return agent

    manager = TaskManager(_factory)
    outcome = await manager.run("do the privileged thing")
    assert outcome["status"] == "awaiting_confirmation"
    task_id = outcome["task_id"]

    for _ in range(MAX_CONFIRMATIONS_PER_TASK):
        outcome = await manager.confirm(task_id, approved=True)
        assert outcome["status"] == "awaiting_confirmation"

    outcome = await manager.confirm(task_id, approved=True)
    assert outcome["status"] == "error"
    assert "confirmations" in outcome["note"]
    summary = await manager.summarize()
    assert summary["tasks"][0]["status"] == "error"


async def test_pending_note_reports_paused_tasks_for_reconnect_priming() -> None:
    """pending_note() surfaces paused tasks so an amnesiac post-reconnect session re-asks
    the user (live finding 2026-07-03: a 1006 drop orphaned a paused task forever)."""

    def _factory(task_id: str) -> DelegateAgent:
        agent = DelegateAgent(
            client=_FakeAsyncClient(
                [_interaction("i1", _call("run_shell", call_id="c1", command="rm -rf /x"))]
            )
        )

        async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
            agent.require_confirmation(args["command"], "destructive delete")
            return {"status": "ok"}

        agent._handlers["run_shell"] = _guarded
        return agent

    manager = TaskManager(_factory)
    assert manager.pending_note() == ""  # nothing outstanding

    paused = await manager.run("delete x")
    task_id = paused["task_id"]
    note = manager.pending_note()
    assert task_id in note
    assert "PAUSED" in note
    assert "rm -rf /x" in note
    assert "confirm_task" in note


async def test_confirm_unknown_or_not_paused_task_errors() -> None:
    manager = TaskManager(lambda tid: DelegateAgent(client=_FakeAsyncClient([])))
    outcome = await manager.confirm("task-99", approved=True)
    assert outcome["status"] == "error"
    assert "unknown task" in outcome["error"]


async def test_on_task_end_fires_after_completion_but_not_while_paused() -> None:
    ended: list[str] = []

    async def _on_end(task_id: str) -> None:
        ended.append(task_id)

    def _factory(task_id: str) -> DelegateAgent:
        agent = DelegateAgent(
            client=_FakeAsyncClient(
                [
                    _interaction("i1", _call("run_shell", call_id="c1", command="sudo x")),
                    _interaction("i2", _text_output("done")),
                ]
            )
        )

        async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
            agent.require_confirmation(args["command"], "sudo")
            return {"status": "ok"}

        agent._handlers["run_shell"] = _guarded
        return agent

    manager = TaskManager(_factory, on_task_end=_on_end)
    paused = await manager.run("do a sudo thing")
    assert ended == []  # page kept alive across the confirmation pause
    await manager.confirm(paused["task_id"], approved=True)
    assert ended == [paused["task_id"]]


async def test_cancellation_marks_record_cancelled() -> None:
    started = asyncio.Event()

    async def _hang(args: dict[str, Any]) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(60)
        return {}

    def _factory(task_id: str) -> DelegateAgent:
        return DelegateAgent(
            client=_FakeAsyncClient([_interaction("i1", _call("run_shell", command="sleep 60"))]),
            tool_handlers={"run_shell": _hang},
        )

    manager = TaskManager(_factory)
    job = asyncio.ensure_future(manager.run("hang"))
    await started.wait()
    job.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await job
    summary = await manager.summarize()
    assert summary["tasks"][0]["status"] == "cancelled"


async def test_empty_goal_is_rejected() -> None:
    manager = TaskManager(lambda tid: DelegateAgent(client=_FakeAsyncClient([])))
    outcome = await manager.run("   ")
    assert outcome["status"] == "error"


# --- live-fix (2d): per-task "approve all" grant — one spoken yes pre-approves the rest ----
#
# Live finding: a multi-step task (e.g. renaming a dozen files) paused for a fresh voice
# confirmation on EVERY non-allowlisted step, even after the user had already said "yes, do
# all of them". confirm(task_id, approved=True, approve_all=True) sets a persistent bypass
# for the REST of that task's non-destructive confirmations; destructive-reason pauses still
# confirm individually, and the runaway-task budget still applies.


async def test_confirm_with_approve_all_preapproves_subsequent_confirmations() -> None:
    responses = [
        _interaction("i1", _call("run_shell", call_id="c1", command="step 1")),
        _interaction("i2", _call("run_shell", call_id="c2", command="step 2")),
        _interaction("i3", _call("run_shell", call_id="c3", command="step 3")),
        _interaction("i4", _text_output("all steps done")),
    ]

    def _factory(task_id: str) -> DelegateAgent:
        agent = DelegateAgent(client=_FakeAsyncClient(responses))

        async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
            agent.require_confirmation(args["command"], "not on the allowlist")
            return {"status": "ok"}

        agent._handlers["run_shell"] = _guarded
        return agent

    manager = TaskManager(_factory)
    paused = await manager.run("do three things")
    assert paused["status"] == "awaiting_confirmation"
    task_id = paused["task_id"]

    final = await manager.confirm(task_id, approved=True, approve_all=True)
    assert final["status"] == "done"
    assert final["note"] == "all steps done"


async def test_approve_all_still_pauses_individually_for_destructive_reason() -> None:
    responses = [
        _interaction("i1", _call("run_shell", call_id="c1", command="rm -rf /a")),
        _interaction("i2", _call("run_shell", call_id="c2", command="rm -rf /b")),
        _interaction("i3", _text_output("both removed")),
    ]

    def _factory(task_id: str) -> DelegateAgent:
        agent = DelegateAgent(client=_FakeAsyncClient(responses))

        async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
            agent.require_confirmation(args["command"], "matches a destructive pattern (rm)")
            return {"status": "ok"}

        agent._handlers["run_shell"] = _guarded
        return agent

    manager = TaskManager(_factory)
    paused = await manager.run("delete two things")
    task_id = paused["task_id"]

    # approve_all=True on the FIRST destructive confirmation must not silently wave through
    # the second destructive step too — every destructive action confirms on its own.
    still_paused = await manager.confirm(task_id, approved=True, approve_all=True)
    assert still_paused["status"] == "awaiting_confirmation"
    assert "rm -rf /b" in still_paused["action"]

    final = await manager.confirm(task_id, approved=True, approve_all=True)
    assert final["status"] == "done"
    assert final["note"] == "both removed"


async def test_max_confirmations_still_aborts_runaway_task_under_approve_all() -> None:
    from duplex_bridge.actions.tasks import MAX_CONFIRMATIONS_PER_TASK

    responses = [
        _interaction(f"i{n}", _call("run_shell", call_id=f"c{n}", command=f"step {n}"))
        for n in range(MAX_CONFIRMATIONS_PER_TASK + 2)
    ]

    def _factory(task_id: str) -> DelegateAgent:
        agent = DelegateAgent(client=_FakeAsyncClient(responses))

        async def _guarded(args: dict[str, Any]) -> dict[str, Any]:
            agent.require_confirmation(
                args["command"], "matches a destructive pattern (privileged)"
            )
            return {"status": "ok"}

        agent._handlers["run_shell"] = _guarded
        return agent

    manager = TaskManager(_factory)
    outcome = await manager.run("do the privileged thing")
    task_id = outcome["task_id"]

    for _ in range(MAX_CONFIRMATIONS_PER_TASK):
        outcome = await manager.confirm(task_id, approved=True, approve_all=True)
        assert outcome["status"] == "awaiting_confirmation"

    outcome = await manager.confirm(task_id, approved=True, approve_all=True)
    assert outcome["status"] == "error"
    assert "confirmations" in outcome["note"]
    summary = await manager.summarize()
    assert summary["tasks"][0]["status"] == "error"
