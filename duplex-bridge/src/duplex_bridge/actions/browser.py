"""DelegateBrowser — one persistent async Chromium shared by all delegate tasks.

Why a second Playwright call site (chrome.py already launches one): the Week-7
``playwright_navigator`` uses the SYNC API launched per call, which is correct where it
runs — inside the BackgroundWorker *thread pool*, no event loop in that thread. The
delegate agent is the opposite: an awaited coroutine on the main loop (sync Playwright
raises there), issuing several browser calls per task (launch-per-call would pay ~1-2 s
each time). So the delegate gets a persistent ``playwright.async_api`` browser: one
Chromium process, one isolated ``BrowserContext``+``Page`` per task id — concurrent tasks
browse in parallel without sharing cookies or clobbering each other's page state.

``playwright_factory`` is the injectable seam (the ``Navigator``/``Fetcher`` pattern from
chrome.py): tests pass a stub that never launches a real browser.

Live finding 2026-07-03: launching Playwright's *bundled* Chromium ("Chrome for Testing")
flashes a visible test-browser window on the user's desktop for what is meant to be
headless background research. ``_ensure_started`` now prefers the user's installed Google
Chrome (``channel="chrome"``) and falls back to the bundled Chromium only if that channel
is unavailable on the machine; ``headless`` defaults to ``True`` so research browsing never
flashes a window. When the delegate wants the *user* to actually see a page, it should not
drive this headless research browser at all — it should surface the URL via ``open <url>``
(macOS) so it opens in the user's real default browser (see ``actions/delegate.py`` system
prompt guidance).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Cap what read_text returns to the delegate model per call.
_MAX_TEXT_CHARS = 6000

# Tool specs for a DelegateAgent's extra_tools (name -> Interactions function spec).
BROWSER_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "browser_navigate": {
        "description": "Open a URL in your persistent browser page and return its title.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "browser_click": {
        "description": "Click the element matching a CSS selector (or text= selector).",
        "parameters": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
    "browser_fill": {
        "description": "Fill the input matching a CSS selector with text.",
        "parameters": {
            "type": "object",
            "properties": {"selector": {"type": "string"}, "text": {"type": "string"}},
            "required": ["selector", "text"],
        },
    },
    "browser_read_text": {
        "description": (
            "Read visible text from the current page — the whole body, or one element "
            "when a CSS selector is given."
        ),
        "parameters": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
        },
    },
}


class DelegateBrowser:
    """Lazy-started shared Chromium; per-task isolated contexts; safe concurrent use."""

    def __init__(
        self,
        *,
        headless: bool = True,
        playwright_factory: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        # headless=True by default: this browser drives background research for the
        # delegate agent, never something the user is meant to watch. Mirrors the
        # headless-toggle convention in actions/chrome.py's playwright_navigator — pass
        # headless=False explicitly for local debugging only.
        self._headless = headless
        self._playwright_factory = playwright_factory
        self._playwright: Any = None
        self._browser: Any = None
        self._pages: dict[str, Any] = {}
        self._start_lock = asyncio.Lock()

    async def _ensure_started(self) -> Any:
        async with self._start_lock:
            if self._browser is None:
                if self._playwright_factory is not None:
                    self._playwright = await self._playwright_factory()
                else:
                    from playwright.async_api import async_playwright  # noqa: PLC0415

                    self._playwright = await async_playwright().start()
                try:
                    self._browser = await self._playwright.chromium.launch(
                        channel="chrome", headless=self._headless
                    )
                    logger.info(
                        "[browser] chromium started via user's Chrome (headless=%s)",
                        self._headless,
                    )
                except Exception:
                    logger.info(
                        "[browser] Chrome channel unavailable, falling back to bundled "
                        "chromium (headless=%s)",
                        self._headless,
                    )
                    self._browser = await self._playwright.chromium.launch(headless=self._headless)
        return self._browser

    async def _page(self, task_id: str) -> Any:
        browser = await self._ensure_started()
        page = self._pages.get(task_id)
        if page is None:
            context = await browser.new_context()
            page = await context.new_page()
            self._pages[task_id] = page
        return page

    # -- tool operations (all return JSON-serializable dicts) -------------------------

    async def navigate(self, task_id: str, url: str) -> dict[str, Any]:
        page = await self._page(task_id)
        await page.goto(url)
        return {"status": "ok", "url": page.url, "title": await page.title()}

    async def click(self, task_id: str, selector: str) -> dict[str, Any]:
        page = await self._page(task_id)
        await page.click(selector, timeout=10_000)
        return {"status": "ok", "url": page.url}

    async def fill(self, task_id: str, selector: str, text: str) -> dict[str, Any]:
        page = await self._page(task_id)
        await page.fill(selector, text, timeout=10_000)
        return {"status": "ok"}

    async def read_text(self, task_id: str, selector: str | None = None) -> dict[str, Any]:
        page = await self._page(task_id)
        text = await page.inner_text(selector or "body")
        return {"status": "ok", "text": text[:_MAX_TEXT_CHARS]}

    async def close_page(self, task_id: str) -> None:
        page = self._pages.pop(task_id, None)
        if page is not None:
            await page.context.close()

    async def aclose(self) -> None:
        """Tear down all task contexts, the browser, and the playwright driver."""
        for task_id in list(self._pages):
            await self.close_page(task_id)
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    # -- DelegateAgent integration -----------------------------------------------------

    def handlers_for_task(
        self, task_id: str
    ) -> dict[str, Callable[[dict[str, Any]], Awaitable[Any]]]:
        """Bound browser_* handlers for one task's DelegateAgent (pair with BROWSER_TOOL_SPECS)."""

        async def _navigate(args: dict[str, Any]) -> dict[str, Any]:
            return await self.navigate(task_id, str(args.get("url") or ""))

        async def _click(args: dict[str, Any]) -> dict[str, Any]:
            return await self.click(task_id, str(args.get("selector") or ""))

        async def _fill(args: dict[str, Any]) -> dict[str, Any]:
            return await self.fill(
                task_id, str(args.get("selector") or ""), str(args.get("text") or "")
            )

        async def _read_text(args: dict[str, Any]) -> dict[str, Any]:
            selector = args.get("selector")
            return await self.read_text(task_id, str(selector) if selector else None)

        return {
            "browser_navigate": _navigate,
            "browser_click": _click,
            "browser_fill": _fill,
            "browser_read_text": _read_text,
        }
