"""Week-6 async background worker: tool calls run off the hot path; the 200 ms tick survives."""

from __future__ import annotations

import asyncio
import time

from duplex_bridge.worker import BackgroundWorker, JobResult, ToolDispatcher

# --- worker basics --------------------------------------------------------


async def test_submit_returns_immediately_for_blocking_job():
    """submit() must NOT await the job — it returns as soon as work is scheduled."""
    worker = BackgroundWorker(max_workers=2)
    t0 = time.perf_counter()
    fut = worker.submit(time.sleep, 0.3, name="sleep")
    submit_ms = (time.perf_counter() - t0) * 1000.0
    assert submit_ms < 20.0  # scheduling is instant; the 0.3 s sleep happens off-loop
    await fut
    await worker.aclose()


async def test_event_loop_not_blocked_during_blocking_job():
    """While a 0.3 s blocking job runs, a 50 ms loop sleep still completes on time."""
    worker = BackgroundWorker(max_workers=2)
    worker.submit(time.sleep, 0.3, name="blocker")
    t0 = time.perf_counter()
    await asyncio.sleep(0.05)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.12  # loop was free; the 50 ms sleep was not delayed by the blocker
    await worker.aclose()


async def test_blocking_job_result_delivered():
    results: list[JobResult] = []
    worker = BackgroundWorker(on_result=results.append)
    await worker.submit(lambda x: x * 2, 21, name="double")
    await worker.drain()
    assert len(results) == 1
    assert results[0].ok and results[0].result == 42 and results[0].name == "double"
    await worker.aclose()


async def test_async_job_runs_as_task():
    results: list[JobResult] = []
    worker = BackgroundWorker(on_result=results.append)

    async def fetch() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    await worker.submit(fetch, name="fetch")
    await worker.drain()
    assert results[0].ok and results[0].result == "ok"
    await worker.aclose()


async def test_job_error_is_captured_not_raised():
    results: list[JobResult] = []
    worker = BackgroundWorker(on_result=results.append)

    def boom() -> None:
        raise ValueError("kaboom")

    fut = worker.submit(boom, name="boom")
    await asyncio.gather(fut, return_exceptions=True)
    await worker.drain()
    assert results[0].ok is False
    assert isinstance(results[0].error, ValueError)
    assert worker.stats["failed"] == 1
    await worker.aclose()


async def test_drain_waits_for_inflight():
    worker = BackgroundWorker(max_workers=4)
    for i in range(4):
        worker.submit(time.sleep, 0.1, name=f"j{i}")
    assert worker.inflight == 4
    await worker.drain()
    assert worker.inflight == 0
    assert worker.stats["completed"] == 4
    await worker.aclose()


# --- dispatcher -----------------------------------------------------------


async def test_dispatcher_routes_to_handler_off_path():
    seen: list[dict] = []
    worker = BackgroundWorker()
    dispatcher = ToolDispatcher(worker, handlers={"open_maps": lambda args: seen.append(args)})

    # plain-mapping tool call (provider-neutral path)
    dispatcher.dispatch({"name": "open_maps", "args": {"q": "1600 Amphitheatre"}})
    assert worker.stats["submitted"] == 1
    await worker.drain()
    assert seen == [{"q": "1600 Amphitheatre"}]
    await worker.aclose()


async def test_dispatcher_skips_unknown_tool():
    worker = BackgroundWorker()
    dispatcher = ToolDispatcher(worker)
    dispatcher.dispatch({"name": "nonexistent", "args": {}})
    assert worker.stats["submitted"] == 0
    await worker.aclose()


async def test_dispatcher_parses_gemini_style_function_calls():
    seen: list[dict] = []
    worker = BackgroundWorker()
    dispatcher = ToolDispatcher(worker, handlers={"edit": lambda a: seen.append(a)})

    class _FC:
        name = "edit"
        args = {"file": "api.py"}

    class _ToolCall:
        function_calls = [_FC()]

    dispatcher.dispatch(_ToolCall())
    await worker.drain()
    assert seen == [{"file": "api.py"}]
    await worker.aclose()


# --- the milestone proof: 200 ms tick never stalls under load -------------


async def _tick_lateness_under_load(*, offload: bool) -> list[float]:
    """Run a 200 ms audio-servicing tick while a burst of heavy (0.2 s) tool jobs fires.

    offload=True submits jobs to the worker (off the loop); offload=False runs the SAME
    blocking work inline on the loop (what awaiting a slow tool callback in the recv loop
    does). Returns per-tick lateness in ms = actual_interval - 200 ms.
    """
    nominal = 0.200
    n_ticks = 12
    n_jobs = 10
    job_seconds = 0.2
    lateness_ms: list[float] = []
    worker = BackgroundWorker(max_workers=4)

    async def ticker() -> None:
        prev = time.perf_counter()
        for _ in range(n_ticks):
            await asyncio.sleep(nominal)
            now = time.perf_counter()
            lateness_ms.append(((now - prev) - nominal) * 1000.0)
            prev = now

    tick_task = asyncio.create_task(ticker())
    await asyncio.sleep(nominal * 2)  # let the tick settle, then slam it with tool work

    def blocking_tool() -> None:
        time.sleep(job_seconds)  # models sync HTTP / file I/O / subprocess / model call

    if offload:
        for i in range(n_jobs):
            worker.submit(blocking_tool, name=f"tool{i}")
    else:
        for _ in range(n_jobs):
            blocking_tool()  # inline on the loop — this is the stall we are preventing

    await tick_task
    await worker.aclose()
    return lateness_ms


async def test_tick_survives_load_when_offloaded_and_stalls_when_inline():
    offloaded = await _tick_lateness_under_load(offload=True)
    inline = await _tick_lateness_under_load(offload=False)

    max_offloaded = max(offloaded)
    max_inline = max(inline)

    # With the worker, the 200 ms tick stays on-cadence (no stall) even under 10x0.2 s of load.
    assert max_offloaded < 60.0, f"tick stalled under offload: max lateness {max_offloaded:.0f}ms"
    # Control: the identical work run inline DOES stall the loop — proves the test has teeth.
    assert max_inline > 400.0, f"inline control did not stall as expected: {max_inline:.0f} ms"
