"""Week-7 host-action demo — the full chain minus the live-audio hop.

Simulates the tool calls the duplex model emits for the two demos and runs them through the REAL
bridge path (ToolDispatcher -> BackgroundWorker -> actuator), off the audio hot path:

  1. IDE  : "rewrite this function async" -> rewrite_function_async on a real temp file.
  2. Chrome: "compare these products"     -> compare_products opens a real comparison in Chromium
             (Playwright, headless + screenshot). Falls back to reporting the built URL if a
             browser is unavailable.

The only piece this does NOT exercise is speech -> Gemini -> tool-call (you drive that live; see
the live-run command in docs/week7-host-actions-acceptance.md). Run:
    uv run --no-sync python scripts/demo/demo_host_actions.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))

from duplex_bridge.actions import compare_products, rewrite_function_async  # noqa: E402
from duplex_bridge.actions.chrome import playwright_navigator  # noqa: E402
from duplex_bridge.worker import BackgroundWorker, JobResult, ToolDispatcher  # noqa: E402

_SAMPLE = """import time
import requests


def fetch_user(user_id):
    resp = requests.get(f"https://api.example.com/users/{user_id}")
    time.sleep(0.1)
    return resp.json()
"""

_SHOT = _ROOT / "scripts" / "bench" / "results" / "week7_chrome_compare.png"


async def main() -> int:
    results: list[JobResult] = []
    worker = BackgroundWorker(on_result=results.append)
    dispatcher = ToolDispatcher(worker)
    dispatcher.register(
        "rewrite_function_async",
        lambda a: rewrite_function_async(a["file"], a["function"], a.get("new_source")),
    )
    dispatcher.register(
        "compare_products",
        lambda a: compare_products(
            list(a.get("products", [])),
            navigate=playwright_navigator(headless=True),
            screenshot=str(_SHOT),
        ),
    )

    # --- Demo 1: IDE "rewrite this function async" ---
    tmp = Path(tempfile.mkdtemp()) / "service.py"
    tmp.write_text(_SAMPLE)
    print("=" * 70)
    print('DEMO 1 — IDE: "rewrite this function async" (fetch_user)')
    print("=" * 70)
    print("BEFORE:\n" + _SAMPLE)
    dispatcher.dispatch(
        {"name": "rewrite_function_async", "args": {"file": str(tmp), "function": "fetch_user"}}
    )
    await worker.drain()
    print("AFTER:\n" + tmp.read_text())

    # --- Demo 2: Chrome "compare these products" ---
    print("=" * 70)
    print('DEMO 2 — Chrome: "compare these products" (PS5 vs Xbox Series X)')
    print("=" * 70)
    _SHOT.parent.mkdir(parents=True, exist_ok=True)
    dispatcher.dispatch(
        {"name": "compare_products", "args": {"products": ["PlayStation 5", "Xbox Series X"]}}
    )
    await worker.drain()

    print("\nResults (each ran off the hot path via the worker):")
    for r in results:
        d = r.result
        ms = f"{r.duration_ms:.0f} ms"
        if hasattr(d, "file"):  # RewriteResult
            print(f"  - {r.name}: applied={d.applied} async={d.is_async} ({ms})")
        else:  # ComparisonResult
            extra = f" screenshot={d.screenshot}" if d.screenshot else ""
            extra += f" error={d.error}" if d.error else ""
            print(f"  - {r.name}: opened={d.opened} page={d.page_path}{extra} ({ms})")
    await worker.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
