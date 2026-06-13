"""Chrome host action — open a side-by-side comparison of the pointed-at products.

The user points at two or more products and says "compare these"; entity extraction (Week 5)
yields the product names, the model emits a ``compare_products`` tool call, and this actuator:

  1. fetches a short summary for each product (Wikipedia REST API — bot-friendly, unlike a
     headless search-engine query, which gets CAPTCHA'd);
  2. renders a real side-by-side comparison page; and
  3. opens it in real Chromium via Playwright.

``fetch`` and ``navigate`` are injectable, so tests assert the rendered comparison + the opened
page without touching the network or launching a browser. The ``navigate`` seam also lets a fuller
computer-use agent replace Playwright later.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ProductCard = dict[str, Any]  # {title, summary, url, thumbnail}
Fetcher = Callable[[str], ProductCard]
# A navigator opens a URL and returns metadata about the loaded page (title, optional screenshot).
Navigator = Callable[[str, str | None], dict[str, Any]]


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of a compare-products action."""

    products: list[str]
    opened: bool
    page_path: str | None = None
    url: str | None = None
    title: str | None = None
    screenshot: str | None = None
    cards: list[ProductCard] = field(default_factory=list)
    error: str | None = None


def compare_products(
    products: list[str],
    fetch: Fetcher | None = None,
    navigate: Navigator | None = None,
    screenshot: str | None = None,
    out_dir: str | Path | None = None,
) -> ComparisonResult:
    """Open a real side-by-side comparison of ``products`` in Chrome.

    Needs >=2 non-empty product names. ``fetch`` defaults to a Wikipedia-summary lookup;
    ``navigate`` defaults to headless Chromium via Playwright. Both are injectable for tests.
    """
    cleaned = [p.strip() for p in products if p and p.strip()]
    if len(cleaned) < 2:
        return ComparisonResult(
            products=cleaned, opened=False, error="need >=2 products to compare"
        )

    fetcher = fetch or _wikipedia_fetch
    cards: list[ProductCard] = []
    for name in cleaned:
        try:
            cards.append(fetcher(name))
        except Exception as exc:  # noqa: BLE001 — a failed lookup degrades, never crashes
            logger.warning("[chrome] product lookup failed for %s: %s", name, exc)
            cards.append(
                {"title": name, "summary": "(no info available)", "url": "", "thumbnail": None}
            )

    page_html = _render_comparison_html(cleaned, cards)
    out_path = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="aimer-compare-"))
    out_path.mkdir(parents=True, exist_ok=True)
    page_file = out_path / "comparison.html"
    page_file.write_text(page_html, encoding="utf-8")
    url = page_file.as_uri()

    nav = navigate or playwright_navigator(headless=True)
    try:
        info = nav(url, screenshot)
    except Exception as exc:  # noqa: BLE001 — a browser failure must not crash the worker
        logger.warning("[chrome] compare_products navigation failed: %s", exc)
        return ComparisonResult(
            products=cleaned,
            opened=False,
            page_path=str(page_file),
            url=url,
            cards=cards,
            error=str(exc),
        )

    logger.info("[chrome] opened comparison for %s", cleaned)
    return ComparisonResult(
        products=cleaned,
        opened=True,
        page_path=str(page_file),
        url=url,
        title=info.get("title"),
        screenshot=info.get("screenshot"),
        cards=cards,
    )


def _wikipedia_fetch(name: str) -> ProductCard:
    """Fetch a product's title/summary/url/thumbnail from the Wikipedia REST summary API."""
    title = name.strip()
    api = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
        title.replace(" ", "_")
    )
    req = urllib.request.Request(api, headers={"User-Agent": "Aimer/0.1 (host-action demo)"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — https, fixed host
        data = json.load(resp)
    return {
        "title": data.get("title", title),
        "summary": data.get("extract", ""),
        "url": ((data.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
        "thumbnail": (data.get("thumbnail") or {}).get("source"),
    }


def _render_comparison_html(products: list[str], cards: list[ProductCard]) -> str:
    """Render a self-contained side-by-side comparison page (no external CSS/JS)."""
    heading = html_lib.escape(" vs ".join(products))
    columns = []
    for card in cards:
        thumb = card.get("thumbnail")
        img = f'<img src="{html_lib.escape(thumb)}" style="max-width:180px">' if thumb else ""
        columns.append(
            "<td style='vertical-align:top;width:50%;padding:18px;border:1px solid #ddd'>"
            f"<h2>{html_lib.escape(str(card.get('title', '')))}</h2>{img}"
            f"<p>{html_lib.escape(str(card.get('summary', '')))}</p>"
            f"<a href='{html_lib.escape(str(card.get('url', '')))}'>source</a></td>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Compare: {heading}</title></head>"
        "<body style='font-family:-apple-system,sans-serif;max-width:1100px;margin:24px auto'>"
        f"<h1>Comparison: {heading}</h1>"
        f"<table style='width:100%;border-collapse:collapse'><tr>{''.join(columns)}</tr></table>"
        "</body></html>"
    )


def playwright_navigator(headless: bool = True) -> Navigator:
    """Return a Navigator that opens a URL in real Chromium via Playwright.

    ``headless=True`` for headless verification (screenshots); ``headless=False`` for the live
    demo so the user sees Chrome open. This is genuine browser automation — the seam (the
    ``navigate`` parameter of :func:`compare_products`) lets a fuller computer-use agent replace it.
    """

    def _navigate(url: str, screenshot: str | None) -> dict[str, Any]:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415 (optional heavy import)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                title = page.title()
                shot: str | None = None
                if screenshot:
                    page.screenshot(path=screenshot, full_page=True)
                    shot = screenshot
                if not headless:
                    page.wait_for_timeout(4_000)  # keep the comparison visible during a live demo
                return {"title": title, "screenshot": shot}
            finally:
                browser.close()

    return _navigate
