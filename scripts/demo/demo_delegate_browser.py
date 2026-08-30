"""User-run demo: the DelegateAgent drives the persistent browser against a real page.

Runs one delegate goal end-to-end with the REAL Gemini model and REAL Chromium (headed),
no mic needed: the browser_* tools registered exactly as __main__ wires them. Requires
GEMINI_API_KEY (read from .env like the bench scripts) and `playwright install chromium`.

Usage:
    uv run python scripts/demo/demo_delegate_browser.py
    uv run python scripts/demo/demo_delegate_browser.py --goal "read example.com's headline"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "bench"))

from duplex_bridge.actions.browser import BROWSER_TOOL_SPECS, DelegateBrowser  # noqa: E402
from duplex_bridge.actions.delegate import DelegateAgent  # noqa: E402
from measure_ttfb import load_api_key  # noqa: E402

DEFAULT_GOAL = "Open https://en.wikipedia.org/wiki/Coffee and tell me its page title."


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    import os

    os.environ.setdefault("GEMINI_API_KEY", load_api_key())
    browser = DelegateBrowser(headless=args.headless)
    agent = DelegateAgent(
        tool_handlers=browser.handlers_for_task("demo"),
        extra_tools=BROWSER_TOOL_SPECS,
    )
    try:
        result = await agent.run(args.goal)
        print(f"\nstatus: {result.status}\nnote:   {result.note}")
        return 0 if result.status == "done" else 1
    finally:
        await browser.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
