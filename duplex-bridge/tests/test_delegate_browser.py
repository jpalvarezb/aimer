"""DelegateBrowser — stubbed Playwright factory, no real browser launched.

The stub mirrors the async Playwright surface the browser touches (chromium.launch,
new_context, new_page, goto/click/fill/inner_text/title, context.close, stop). A real
headed run is scripts/demo/demo_delegate_browser.py (user-run, like demo_host_actions).
"""

from __future__ import annotations

from typing import Any

from duplex_bridge.actions.browser import BROWSER_TOOL_SPECS, DelegateBrowser


class _StubPage:
    def __init__(self, context: _StubContext) -> None:
        self.context = context
        self.url = "about:blank"
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def goto(self, url: str) -> None:
        self.calls.append(("goto", (url,)))
        self.url = url

    async def title(self) -> str:
        return f"title-of:{self.url}"

    async def click(self, selector: str, timeout: float = 0) -> None:
        self.calls.append(("click", (selector,)))

    async def fill(self, selector: str, text: str, timeout: float = 0) -> None:
        self.calls.append(("fill", (selector, text)))

    async def inner_text(self, selector: str) -> str:
        return f"text-of:{selector}@{self.url}"


class _StubContext:
    def __init__(self) -> None:
        self.closed = False
        self.page: _StubPage | None = None

    async def new_page(self) -> _StubPage:
        self.page = _StubPage(self)
        return self.page

    async def close(self) -> None:
        self.closed = True


class _StubBrowser:
    def __init__(self) -> None:
        self.contexts: list[_StubContext] = []
        self.closed = False

    async def new_context(self) -> _StubContext:
        context = _StubContext()
        self.contexts.append(context)
        return context

    async def close(self) -> None:
        self.closed = True


class _StubChromium:
    def __init__(self, browser: _StubBrowser) -> None:
        self._browser = browser
        self.launches: list[dict[str, Any]] = []

    async def launch(self, headless: bool = True) -> _StubBrowser:
        self.launches.append({"headless": headless})
        return self._browser


class _StubPlaywright:
    def __init__(self, browser: _StubBrowser) -> None:
        self.chromium = _StubChromium(browser)
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


def _browser_pair() -> tuple[DelegateBrowser, _StubBrowser, _StubPlaywright]:
    stub_browser = _StubBrowser()
    stub_pw = _StubPlaywright(stub_browser)

    async def _factory() -> _StubPlaywright:
        return stub_pw

    return DelegateBrowser(playwright_factory=_factory), stub_browser, stub_pw


async def test_lazy_start_and_navigate() -> None:
    browser, stub_browser, stub_pw = _browser_pair()
    assert stub_pw.chromium.launches == []  # nothing until first use
    outcome = await browser.navigate("t1", "https://example.com")
    assert outcome == {
        "status": "ok",
        "url": "https://example.com",
        "title": "title-of:https://example.com",
    }
    assert len(stub_pw.chromium.launches) == 1


async def test_tasks_get_isolated_contexts_and_pages() -> None:
    browser, stub_browser, _ = _browser_pair()
    await browser.navigate("t1", "https://a.example")
    await browser.navigate("t2", "https://b.example")
    await browser.click("t1", "#buy")
    assert len(stub_browser.contexts) == 2  # one context per task
    t1_page = stub_browser.contexts[0].page
    t2_page = stub_browser.contexts[1].page
    assert ("click", ("#buy",)) in t1_page.calls
    assert all(call[0] != "click" for call in t2_page.calls)


async def test_read_text_defaults_to_body_and_caps() -> None:
    browser, _, _ = _browser_pair()
    await browser.navigate("t1", "https://a.example")
    outcome = await browser.read_text("t1")
    assert outcome["text"] == "text-of:body@https://a.example"


async def test_handlers_for_task_route_and_match_specs() -> None:
    browser, stub_browser, _ = _browser_pair()
    handlers = browser.handlers_for_task("t9")
    assert set(handlers) == set(BROWSER_TOOL_SPECS)
    await handlers["browser_navigate"]({"url": "https://x.example"})
    await handlers["browser_fill"]({"selector": "#q", "text": "coffee"})
    page = stub_browser.contexts[0].page
    assert ("goto", ("https://x.example",)) in page.calls
    assert ("fill", ("#q", "coffee")) in page.calls


async def test_aclose_tears_down_contexts_browser_and_driver() -> None:
    browser, stub_browser, stub_pw = _browser_pair()
    await browser.navigate("t1", "https://a.example")
    await browser.navigate("t2", "https://b.example")
    await browser.aclose()
    assert all(context.closed for context in stub_browser.contexts)
    assert stub_browser.closed
    assert stub_pw.stopped


async def test_close_page_releases_only_that_task() -> None:
    browser, stub_browser, _ = _browser_pair()
    await browser.navigate("t1", "https://a.example")
    await browser.navigate("t2", "https://b.example")
    await browser.close_page("t1")
    assert stub_browser.contexts[0].closed
    assert not stub_browser.contexts[1].closed


# --- live-fix (1): user's real Chrome, headless default, no bundled-Chromium flash --------
#
# Live finding 2026-07-03: _ensure_started launched Playwright's BUNDLED Chromium ("Chrome
# for Testing"), which flashes a visible test-browser window on the user's desktop. The fix
# tries the user's installed Google Chrome via channel="chrome" first, falling back to the
# bundled chromium only if that channel is unavailable, and defaults headless=True so
# research browsing never flashes a window at all.


class _StubChromiumWithChannel:
    """Records every launch() call's kwargs; optionally fails one specific channel."""

    def __init__(self, browser: _StubBrowser, *, fail_channel: str | None = None) -> None:
        self._browser = browser
        self._fail_channel = fail_channel
        self.launches: list[dict[str, Any]] = []

    async def launch(self, headless: bool = True, channel: str | None = None) -> _StubBrowser:
        self.launches.append({"headless": headless, "channel": channel})
        if channel is not None and channel == self._fail_channel:
            raise RuntimeError(f"channel {channel!r} not found on this machine")
        return self._browser


class _StubPlaywrightWithChannel:
    def __init__(self, browser: _StubBrowser, *, fail_channel: str | None = None) -> None:
        self.chromium = _StubChromiumWithChannel(browser, fail_channel=fail_channel)
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


def _browser_pair_with_channel(
    *, fail_channel: str | None = None, **browser_kwargs: Any
) -> tuple[DelegateBrowser, _StubBrowser, _StubPlaywrightWithChannel]:
    stub_browser = _StubBrowser()
    stub_pw = _StubPlaywrightWithChannel(stub_browser, fail_channel=fail_channel)

    async def _factory() -> _StubPlaywrightWithChannel:
        return stub_pw

    return (
        DelegateBrowser(playwright_factory=_factory, **browser_kwargs),
        stub_browser,
        stub_pw,
    )


async def test_ensure_started_tries_the_users_chrome_channel_first() -> None:
    browser, _, stub_pw = _browser_pair_with_channel()
    await browser.navigate("t1", "https://example.com")
    assert stub_pw.chromium.launches == [{"headless": True, "channel": "chrome"}]


async def test_ensure_started_falls_back_to_bundled_chromium_when_chrome_channel_missing() -> None:
    browser, _, stub_pw = _browser_pair_with_channel(fail_channel="chrome")
    await browser.navigate("t1", "https://example.com")
    # First attempt uses the user's Chrome channel and fails; the fallback retries without
    # a channel (Playwright's bundled Chromium), and that one must succeed.
    assert stub_pw.chromium.launches == [
        {"headless": True, "channel": "chrome"},
        {"headless": True, "channel": None},
    ]


async def test_default_headless_is_true_for_research_browsing() -> None:
    browser, _, stub_pw = _browser_pair_with_channel()
    await browser.navigate("t1", "https://example.com")
    assert stub_pw.chromium.launches[0]["headless"] is True


async def test_headless_false_can_still_be_requested_explicitly() -> None:
    browser, _, stub_pw = _browser_pair_with_channel(headless=False)
    await browser.navigate("t1", "https://example.com")
    assert stub_pw.chromium.launches[0]["headless"] is False
