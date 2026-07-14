"""TaskManager — named, concurrent delegate tasks behind the live model's three tools.

The live model sees exactly three functions:

  - ``delegate_task(goal)``   — start a task; NON_BLOCKING, so the model acks by voice
                                ("on it") and keeps conversing while the task runs
  - ``check_tasks()``         — "how's that rename going?" — status of every task
  - ``confirm_task(task_id, approved)`` — resume a task paused on a safety confirmation

Concurrency model: each ``delegate_task`` call runs one :class:`DelegateAgent` coroutine
on the event loop (submitted via the Week-6 worker, so the audio tick never blocks).
Shell/AppleScript/browser tool calls parallelize freely across tasks; the ONE desktop
(mouse/keyboard) is guarded by the shared ``asyncio.Lock`` each agent's ``computer_use``
handler holds for its whole nested drive — passed in via ``agent_factory``.

Result flow reuses the Phase-1 plumbing end to end: the handler coroutine IS the worker
job, so its return value travels back as the tool's final FunctionResponse (WHEN_IDLE) and
a barge-in ``tool_call_cancellation`` cancels the coroutine mid-flight.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from .delegate import DelegateAgent, DelegateResult, PendingConfirmation

logger = logging.getLogger(__name__)

TaskStatus = Literal["running", "awaiting_confirmation", "done", "error", "cancelled"]

# A single task asking for this many spoken confirmations is a runaway (observed live:
# a delegate driving Gmail via always-confirm UI scripting paused ~40 times while the
# live model rubber-stamped every one). Abort instead of looping forever.
MAX_CONFIRMATIONS_PER_TASK = 6


@dataclass
class DelegateTask:
    """Bookkeeping record for one delegated goal (what check_tasks reports)."""

    id: str
    goal: str
    status: TaskStatus = "running"
    note: str = ""
    pending: PendingConfirmation | None = None
    confirmations: int = 0


@dataclass
class _Managed:
    record: DelegateTask
    agent: DelegateAgent


class TaskManager:
    """Owns the task registry and the agent instances (kept alive for resume)."""

    def __init__(
        self,
        agent_factory: Callable[[str], DelegateAgent],
        *,
        on_task_end: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._on_task_end = on_task_end
        self._tasks: dict[str, _Managed] = {}
        self._ids = itertools.count(1)

    # -- delegate_task ------------------------------------------------------------------

    async def run(self, goal: str, *, context: str = "") -> dict[str, Any]:
        """Run one delegated goal to completion (or its confirmation pause).

        This coroutine is the worker job for a ``delegate_task`` call: its return value
        becomes the final FunctionResponse the live model speaks.
        """
        if not goal.strip():
            return {"status": "error", "error": "empty goal"}
        task_id = f"task-{next(self._ids)}"
        managed = _Managed(DelegateTask(id=task_id, goal=goal), self._agent_factory(task_id))
        self._tasks[task_id] = managed
        logger.info("[tasks] %s started: %s", task_id, goal[:120])
        try:
            result = await managed.agent.run(goal, context)
        except BaseException as exc:
            managed.record.status = "cancelled" if _is_cancel(exc) else "error"
            managed.record.note = str(exc)
            await self._task_ended(task_id)
            raise
        payload = self._settle(managed.record, result)
        if payload["status"] != "awaiting_confirmation":
            await self._task_ended(task_id)
        return payload

    # -- confirm_task ---------------------------------------------------------------------

    async def confirm(
        self, task_id: str, approved: bool, approve_all: bool = False
    ) -> dict[str, Any]:
        """Resume a paused task with the user's spoken decision.

        ``approve_all`` (live de-nagging fix 2d): one spoken "yes, do all of them" pre-
        approves the REST of this task's non-destructive confirmations (threaded through to
        ``DelegateAgent.resume``); destructive-pattern confirmations still pause
        individually, and ``MAX_CONFIRMATIONS_PER_TASK`` still bounds the total either way.
        """
        managed = self._tasks.get(task_id)
        if managed is None:
            known = ", ".join(self._tasks) or "none"
            return {"status": "error", "error": f"unknown task {task_id!r} (known: {known})"}
        if managed.record.status != "awaiting_confirmation":
            return {
                "status": "error",
                "error": f"task {task_id} is {managed.record.status}, not awaiting confirmation",
            }
        managed.record.confirmations += 1
        if managed.record.confirmations > MAX_CONFIRMATIONS_PER_TASK:
            managed.record.status = "error"
            managed.record.pending = None
            managed.record.note = (
                f"aborted: needed more than {MAX_CONFIRMATIONS_PER_TASK} confirmations — "
                "every step of this approach requires privileged actions. Tell the user "
                "the task was stopped and why; do not restart it the same way."
            )
            logger.warning("[tasks] %s aborted: confirmation budget exceeded", task_id)
            await self._task_ended(task_id)
            return {"status": "error", "task_id": task_id, "note": managed.record.note}
        managed.record.status = "running"
        managed.record.pending = None
        logger.info(
            "[tasks] %s resumed (approved=%s, approve_all=%s)", task_id, approved, approve_all
        )
        try:
            result = await managed.agent.resume(approved, approve_all=approve_all)
        except BaseException as exc:
            managed.record.status = "cancelled" if _is_cancel(exc) else "error"
            managed.record.note = str(exc)
            await self._task_ended(task_id)
            raise
        payload = self._settle(managed.record, result)
        if payload["status"] != "awaiting_confirmation":
            await self._task_ended(task_id)
        return payload

    # -- check_tasks ------------------------------------------------------------------------

    async def summarize(self) -> dict[str, Any]:
        """Status of every task this session — backs the check_tasks tool."""
        return {
            "tasks": [
                {
                    "task_id": managed.record.id,
                    "goal": managed.record.goal[:100],
                    "status": managed.record.status,
                    "note": managed.record.note[:200],
                }
                for managed in self._tasks.values()
            ]
        }

    def pending_note(self) -> str:
        """One-line summary of unfinished tasks — primes a fresh Live session after reconnect.

        A reconnect builds a brand-new Live session with no conversational memory, so a
        task paused on a confirmation is otherwise orphaned: the model that promised to
        ask the user no longer exists (live finding 2026-07-03 — the 1006 drop at
        21:00:54 stranded the "July 3rd" note). Wired as the session's resume-context
        provider; empty string when nothing is outstanding.
        """
        notes = []
        for managed in self._tasks.values():
            record = managed.record
            if record.status == "awaiting_confirmation" and record.pending is not None:
                notes.append(
                    f"{record.id} is PAUSED awaiting the user's confirmation to run "
                    f"{record.pending.command[:120]!r} ({record.pending.reason}). Ask the "
                    f"user out loud now, then call confirm_task({record.id!r}, ...)"
                )
            elif record.status == "running":
                notes.append(f"{record.id} is still running: {record.goal[:80]}")
        return " | ".join(notes)

    # -- internals ----------------------------------------------------------------------

    def _settle(self, record: DelegateTask, result: DelegateResult) -> dict[str, Any]:
        if result.status == "awaiting_confirmation" and result.pending is not None:
            record.status = "awaiting_confirmation"
            record.pending = result.pending
            record.note = result.note
            # Live-debuggability: without this, a paused task is indistinguishable from a
            # completed one in the bridge log (live finding 2026-07-03).
            logger.info(
                "[tasks] %s awaiting confirmation: %s — %s",
                record.id,
                result.pending.reason,
                result.pending.command[:160],
            )
            return {
                "status": "awaiting_confirmation",
                "task_id": record.id,
                "action": result.pending.command,
                "reason": result.pending.reason,
                "message": (
                    "This action needs the user's confirmation. Ask them out loud, WAIT for "
                    "their answer, and only then call "
                    f"confirm_task(task_id={record.id!r}, approved=true or false). Do not "
                    "approve on their behalf."
                ),
            }
        record.status = "done" if result.status == "done" else "error"
        record.note = result.note
        logger.info("[tasks] %s %s: %s", record.id, record.status, record.note[:200])
        return {"status": record.status, "task_id": record.id, "note": result.note}

    async def _task_ended(self, task_id: str) -> None:
        if self._on_task_end is None:
            return
        try:
            await self._on_task_end(task_id)
        except Exception:  # noqa: BLE001 — cleanup must never mask the task result
            logger.exception("[tasks] on_task_end failed for %s", task_id)


def _is_cancel(exc: BaseException) -> bool:
    import asyncio  # noqa: PLC0415

    return isinstance(exc, asyncio.CancelledError)
