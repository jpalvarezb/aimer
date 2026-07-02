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


@dataclass
class DelegateTask:
    """Bookkeeping record for one delegated goal (what check_tasks reports)."""

    id: str
    goal: str
    status: TaskStatus = "running"
    note: str = ""
    pending: PendingConfirmation | None = None


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

    async def confirm(self, task_id: str, approved: bool) -> dict[str, Any]:
        """Resume a paused task with the user's spoken decision."""
        managed = self._tasks.get(task_id)
        if managed is None:
            known = ", ".join(self._tasks) or "none"
            return {"status": "error", "error": f"unknown task {task_id!r} (known: {known})"}
        if managed.record.status != "awaiting_confirmation":
            return {
                "status": "error",
                "error": f"task {task_id} is {managed.record.status}, not awaiting confirmation",
            }
        managed.record.status = "running"
        managed.record.pending = None
        logger.info("[tasks] %s resumed (approved=%s)", task_id, approved)
        try:
            result = await managed.agent.resume(approved)
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

    # -- internals ----------------------------------------------------------------------

    def _settle(self, record: DelegateTask, result: DelegateResult) -> dict[str, Any]:
        if result.status == "awaiting_confirmation" and result.pending is not None:
            record.status = "awaiting_confirmation"
            record.pending = result.pending
            record.note = result.note
            return {
                "status": "awaiting_confirmation",
                "task_id": record.id,
                "action": result.pending.command,
                "reason": result.pending.reason,
                "message": (
                    "This action needs the user's confirmation. Ask them out loud, then call "
                    f"confirm_task(task_id={record.id!r}, approved=true or false)."
                ),
            }
        record.status = "done" if result.status == "done" else "error"
        record.note = result.note
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
