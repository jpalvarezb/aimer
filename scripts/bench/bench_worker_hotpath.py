"""Week-6 proof: the 200 ms duplex audio tick never stalls when tool calls run off the hot path.

Models the bridge's hot path as a 200 ms "audio servicing" tick on the asyncio event loop, then
slams it with a burst of long blocking tool calls (sync HTTP / file I/O / subprocess / model
reasoning, modelled as time.sleep, which releases the GIL like real I/O). Two arms:

  - OFFLOAD: tool calls go through BackgroundWorker (thread pool) — the loop stays free.
  - INLINE : the SAME work runs on the loop (what awaiting a slow tool callback in the recv
             loop does) — the loop, and thus audio, stalls.

Reports per-tick lateness (actual interval - 200 ms) p50/p95/max for both, proving the worker
keeps the tick on-cadence. Run: uv run --no-sync python scripts/bench/bench_worker_hotpath.py
Out:  scripts/bench/results/week6_worker_hotpath.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))

from duplex_bridge.worker import BackgroundWorker  # noqa: E402

_OUT = _HERE / "results" / "week6_worker_hotpath.json"

TICK_MS = 200.0
N_TICKS = 25
N_JOBS = 20
JOB_SECONDS = 0.3
MAX_WORKERS = 4


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


async def _run(offload: bool) -> dict:
    nominal = TICK_MS / 1000.0
    lateness: list[float] = []
    worker = BackgroundWorker(max_workers=MAX_WORKERS)
    load_wall = {"start": 0.0, "end": 0.0}

    async def ticker() -> None:
        prev = time.perf_counter()
        for _ in range(N_TICKS):
            await asyncio.sleep(nominal)
            now = time.perf_counter()
            lateness.append(((now - prev) - nominal) * 1000.0)
            prev = now

    def blocking_tool() -> None:
        time.sleep(JOB_SECONDS)

    tick_task = asyncio.create_task(ticker())
    await asyncio.sleep(nominal * 3)  # let the tick settle

    load_wall["start"] = time.perf_counter()
    if offload:
        for i in range(N_JOBS):
            worker.submit(blocking_tool, name=f"tool{i}")
    else:
        for _ in range(N_JOBS):
            blocking_tool()
    await worker.drain()
    load_wall["end"] = time.perf_counter()

    await tick_task
    stats = dict(worker.stats)
    await worker.aclose()

    return {
        "arm": "offload" if offload else "inline",
        "tick_ms_nominal": TICK_MS,
        "lateness_ms_p50": round(_pct(lateness, 0.50), 1),
        "lateness_ms_p95": round(_pct(lateness, 0.95), 1),
        "lateness_ms_max": round(max(lateness), 1),
        "load_complete_wall_s": round(load_wall["end"] - load_wall["start"], 2),
        "worker_stats": stats,
    }


async def main() -> int:
    offload = await _run(offload=True)
    inline = await _run(offload=False)

    print("=" * 68)
    print(f"WEEK-6 HOT-PATH PROOF — {N_JOBS} tool calls x {JOB_SECONDS}s, {TICK_MS:.0f}ms tick")
    print("=" * 68)
    for arm in (offload, inline):
        print(
            f"  {arm['arm']:<8} tick lateness  p50={arm['lateness_ms_p50']:>7.1f}ms  "
            f"p95={arm['lateness_ms_p95']:>7.1f}ms  max={arm['lateness_ms_max']:>8.1f}ms  "
            f"| load done in {arm['load_complete_wall_s']}s"
        )
    verdict = (
        "PASS — tick stayed on-cadence under load (off hot path); inline control stalled"
        if offload["lateness_ms_max"] < 60.0 and inline["lateness_ms_max"] > 400.0
        else "FAIL"
    )
    print(f"\n  {verdict}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps({"offload": offload, "inline": inline, "verdict": verdict}, indent=2)
    )
    print(f"  written: {_OUT}")
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
