"""Async background worker — runs long tool calls OFF the duplex audio hot path.

The bridge's asyncio event loop services audio on a tight cadence: ``GeminiLiveSession._recv_loop``
pulls model audio and enqueues it for the speaker, and mic frames go out via ``send_audio``. Any
work that runs *on that loop* and takes more than a few ms stalls audio — the recv loop stops
pulling model audio and the speaker queue underruns. The two ways tool work stalls the loop:

  1. an ``async`` tool handler **awaited inline** in the recv loop (see the ``on_tool_call``
     dispatch in ``gemini_live.py``), or
  2. a **blocking** call on the loop thread (sync HTTP, file I/O, a subprocess for code edits,
     CPU-bound reasoning).

:class:`BackgroundWorker` removes both: ``submit()`` schedules the work and returns immediately —
coroutines run as tasks, blocking callables run in a thread pool — so the caller (the recv loop's
tool callback) never blocks. Results are delivered out-of-band via an ``on_result`` callback.

This is the generalisation of the Week-5 ``EntityPipeline.schedule`` pattern to arbitrary tool
calls, and it preserves the ``DuplexSession`` seam: the session still just invokes the registered
``on_tool_call`` callback; that callback now hands off to this worker instead of doing the work.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobResult:
    """Outcome of one background job, delivered to the ``on_result`` callback."""

    name: str
    ok: bool
    result: Any = None
    error: BaseException | None = None
    duration_ms: float = 0.0


ResultCallback = Callable[[JobResult], None]


class BackgroundWorker:
    """Run jobs off the event loop so the duplex audio tick is never blocked.

    ``submit`` accepts a coroutine, an async function, or a plain (blocking) callable. Async work
    is scheduled as a task on the loop; blocking/CPU work is offloaded to a thread pool. In both
    cases ``submit`` returns a future *immediately* — it never awaits the job — so the hot path
    that called it keeps running. CPU-bound jobs that hold the GIL should use ``submit`` with a
    process-backed callable; for the common I/O-bound tool calls (web, file, subprocess, model
    reasoning) the thread pool keeps the loop fully free.
    """

    def __init__(self, max_workers: int = 4, on_result: ResultCallback | None = None) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="bridge-worker"
        )
        self._on_result = on_result
        self._inflight: set[asyncio.Future[Any]] = set()
        self._submitted = 0
        self._completed = 0
        self._failed = 0

    @property
    def inflight(self) -> int:
        """Number of jobs currently running or queued."""
        return len(self._inflight)

    @property
    def stats(self) -> dict[str, int]:
        """Counters for diagnostics / the load proof."""
        return {
            "submitted": self._submitted,
            "completed": self._completed,
            "failed": self._failed,
            "inflight": len(self._inflight),
        }

    def submit(
        self, job: Any, *args: Any, name: str | None = None, **kwargs: Any
    ) -> asyncio.Future[Any]:
        """Schedule ``job`` off the hot path and return its future immediately.

        ``job`` may be a coroutine, an async function, or a blocking callable. Blocking callables
        run in the thread pool; everything else runs as a loop task. This call does NOT await the
        job — it returns as soon as the work is scheduled, so the duplex audio loop keeps ticking.
        """
        loop = asyncio.get_running_loop()
        label = name or getattr(job, "__name__", job.__class__.__name__)
        started = time.perf_counter()

        fut: asyncio.Future[Any]
        if asyncio.iscoroutine(job):
            fut = asyncio.ensure_future(job)
        elif inspect.iscoroutinefunction(job):
            fut = asyncio.ensure_future(job(*args, **kwargs))
        elif callable(job):
            call = functools.partial(job, *args, **kwargs)
            fut = asyncio.ensure_future(loop.run_in_executor(self._executor, call))
        else:
            raise TypeError(f"BackgroundWorker.submit: unsupported job type {type(job)!r}")

        self._submitted += 1
        self._inflight.add(fut)
        fut.add_done_callback(lambda f: self._finish(f, label, started))
        return fut

    def _finish(self, fut: asyncio.Future[Any], label: str, started: float) -> None:
        self._inflight.discard(fut)
        duration_ms = (time.perf_counter() - started) * 1000.0
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            self._failed += 1
            logger.warning("[worker] job %s failed: %s", label, exc)
            outcome = JobResult(label, ok=False, error=exc, duration_ms=duration_ms)
        else:
            self._completed += 1
            outcome = JobResult(label, ok=True, result=fut.result(), duration_ms=duration_ms)
        if self._on_result is not None:
            try:
                self._on_result(outcome)
            except Exception:  # noqa: BLE001 — a bad result handler must not kill the worker
                logger.exception("[worker] on_result handler raised")

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for all in-flight jobs to finish (or until ``timeout``)."""
        if not self._inflight:
            return
        await asyncio.wait(set(self._inflight), timeout=timeout)

    async def aclose(self) -> None:
        """Drain in-flight jobs and shut down the thread pool."""
        await self.drain()
        self._executor.shutdown(wait=True)


class ToolDispatcher:
    """Route model-emitted tool calls to handlers, executed via :class:`BackgroundWorker`.

    Register as the session's ``on_tool_call`` callback. ``dispatch`` parses the provider tool
    call, looks up each function's handler, and ``submit``s it to the worker — returning at once,
    so the recv loop that delivered the tool call is never blocked. Unknown tools are logged and
    skipped. Provider-neutral: it reads Gemini-style ``function_calls`` but also accepts a plain
    ``{"name", "args"}`` mapping.
    """

    def __init__(
        self, worker: BackgroundWorker, handlers: Mapping[str, Callable[..., Any]] | None = None
    ) -> None:
        self._worker = worker
        self._handlers: dict[str, Callable[..., Any]] = dict(handlers or {})

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        """Register a handler for a named tool."""
        self._handlers[name] = handler

    def dispatch(self, tool_call: Any) -> None:
        """Submit every function call in ``tool_call`` to the worker (non-blocking).

        Returns ``None`` to satisfy the ``on_tool_call`` callback contract; submitted work is
        tracked on the worker (``inflight`` / ``stats``) and surfaced via its ``on_result``.
        """
        for name, fn_args in _iter_function_calls(tool_call):
            handler = self._handlers.get(name)
            if handler is None:
                logger.info("[tools] no handler for tool %r — skipping", name)
                continue
            self._worker.submit(handler, fn_args, name=name)


def _iter_function_calls(tool_call: Any) -> list[tuple[str, dict[str, Any]]]:
    """Extract (name, args) pairs from a Gemini ToolCall or a plain mapping."""
    calls: list[tuple[str, dict[str, Any]]] = []
    function_calls = getattr(tool_call, "function_calls", None)
    if function_calls:
        for fc in function_calls:
            name = getattr(fc, "name", None)
            if name:
                calls.append((name, dict(getattr(fc, "args", None) or {})))
        return calls
    if isinstance(tool_call, Mapping) and "name" in tool_call:
        calls.append((str(tool_call["name"]), dict(tool_call.get("args") or {})))
    return calls
