"""Playwright-based harvester that generates a ~50-task deictic eval dataset.

Captures real, diverse websites (Wikipedia, MDN, Hacker News, httpbin) and
produces JPEG 512x512 tile screenshots with cursor markers plus a JSONL task
file compatible with scripts/bench/eval_deictic.py.

Usage:
    uv run python scripts/bench/harvest_web_tasks.py
    uv run python scripts/bench/harvest_web_tasks.py --limit 10
    uv run python scripts/bench/harvest_web_tasks.py --out custom.jsonl --img-dir custom_imgs/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("harvest_web")

_HERE = Path(__file__).parent  # scripts/bench/

# ---------------------------------------------------------------------------
# Site & target definitions
# ---------------------------------------------------------------------------
# Each target dict:
#   selector  - CSS selector
#   kind      - element type: heading|paragraph|image|link|table_cell|code|input|button|label
#   nth       - optional int (0-based) to pick nth match (default: 0 = first visible)
#   label     - optional override label text (for inputs with no visible text)
#   relational- bool: capture TWO elements in one tile
#   pair_sel  - second selector (required when relational=True)
#   pair_nth  - optional int for pair_sel (default: 0)
#
# VERIFIED LIVE (2026-06) — key findings:
#   - page.evaluate(fn, [arg1, arg2]) in Playwright 1.60 passes the array as the SINGLE
#     first argument to the JS function.  Multi-arg JS must use object destructuring:
#     evaluate("({sel, idx}) => {...}", {"sel": sel, "idx": idx})  ← CORRECT
#     evaluate("(sel, idx) => {...}", [sel, idx])                   ← BROKEN (sel=[arr])
#   - Wikipedia: #firstHeading and h2 work; infobox th/td work after scrollIntoView fix.
#     p[1] is the first lead paragraph; p[0] is invisible boilerplate.
#   - HN: .titleline, .titleline a, .subtext, .score all resolve. Domain links are
#     even-indexed; story titles are odd among .titleline a (use nth carefully).
#   - httpbin: no button[type=submit]; the submit is <button> (plain). Labels have text.
#     No id/for linking; labels contain their inputs as children.
#   - MDN: pre code has bounding_box=None (rendered off-screen in shadow); use 'p code'
#     (inline code refs, always visible). .notecard p works. h2 and h1 work fine.

SITE_TARGETS: list[dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Wikipedia — Coffee article  (heading, para, image, link, infobox table, relational)
    # VERIFIED 2026-06: Row3 td = "Yemen[1][2][3]" — cursor hits citation [1] via
    # elementFromPoint.  Use row5 (Color) and row6 (Flavor) instead — clean text.
    # Links: "brewed"=idx0, "coffee beans"=idx1 — both non-footnote.
    # -----------------------------------------------------------------------
    {
        "url": "https://en.wikipedia.org/wiki/Coffee?banner=false",
        "targets": [
            # Page title heading
            {"selector": "#firstHeading", "kind": "heading"},
            # Lead paragraph (nth=1 — nth=0 is invisible nav boilerplate)
            {"selector": "#mw-content-text > div > p", "kind": "paragraph", "nth": 1},
            # Second body paragraph (different content)
            {"selector": "#mw-content-text > div > p", "kind": "paragraph", "nth": 2},
            # First article image
            {"selector": "#mw-content-text figure img", "kind": "image"},
            # Non-footnote link (nth=0 = "brewed")
            {"selector": "#mw-content-text > div > p a", "kind": "link", "nth": 0},
            # Another link (nth=1 = "coffee beans")
            {"selector": "#mw-content-text > div > p a", "kind": "link", "nth": 1},
            # Section heading (h2[1] = "Etymology", h2[0]="Contents")
            {"selector": "h2", "kind": "heading", "nth": 1},
            # Infobox cell: row 2 td ("Usually hot; can be iced")
            {"selector": ".infobox tbody tr:nth-child(2) td", "kind": "table_cell"},
            # Infobox cell: row 5 td ("Black, dark brown…") — row3 has citation noise
            {"selector": ".infobox tbody tr:nth-child(5) td", "kind": "table_cell"},
            # Relational: infobox th (label) + td (value) in same row — dist ~136 px
            {
                "selector": ".infobox tbody tr:nth-child(4) th",
                "kind": "heading",
                "relational": True,
                "pair_sel": ".infobox tbody tr:nth-child(4) td",
            },
        ],
    },
    # -----------------------------------------------------------------------
    # Wikipedia — Pointing device article
    # -----------------------------------------------------------------------
    {
        "url": "https://en.wikipedia.org/wiki/Pointing_device?banner=false",
        "targets": [
            {"selector": "#firstHeading", "kind": "heading"},
            # Lead paragraph
            {"selector": "#mw-content-text > div > p", "kind": "paragraph", "nth": 1},
            # First article image
            {"selector": "#mw-content-text figure img", "kind": "image"},
            # Non-footnote link (nth=0)
            {"selector": "#mw-content-text > div > p a", "kind": "link", "nth": 0},
            # Section headings — h2[1] = "Classification"
            {"selector": "h2", "kind": "heading", "nth": 1},
        ],
    },
    # -----------------------------------------------------------------------
    # Wikipedia — Mount Everest  (rich infobox for relational)
    # VERIFIED: row2 td = multi-lang name block (OK for table_cell). Link nth=0
    # resolves to "Nepal" (real link). Relational on row5 (Elevation th+td).
    # -----------------------------------------------------------------------
    {
        "url": "https://en.wikipedia.org/wiki/Mount_Everest?banner=false",
        "targets": [
            {"selector": "#firstHeading", "kind": "heading"},
            # Lead paragraph (nth=1)
            {"selector": "#mw-content-text > div > p", "kind": "paragraph", "nth": 1},
            # First image
            {"selector": "#mw-content-text figure img", "kind": "image"},
            # Non-footnote link: "Nepal"
            {"selector": "#mw-content-text > div > p a", "kind": "link", "nth": 0},
            # Infobox Elevation td (row 5)
            {"selector": ".infobox tbody tr:nth-child(5) td", "kind": "table_cell"},
            # Infobox Prominence td (row 6)
            {"selector": ".infobox tbody tr:nth-child(6) td", "kind": "table_cell"},
            # Relational: Elevation th + td (row 5)
            {
                "selector": ".infobox tbody tr:nth-child(5) th",
                "kind": "heading",
                "relational": True,
                "pair_sel": ".infobox tbody tr:nth-child(5) td",
            },
        ],
    },
    # -----------------------------------------------------------------------
    # Wikipedia — Internet  (different topic; supplies extra variety)
    # Link nth=2 onwards avoids "[a]" footnote markers.
    # -----------------------------------------------------------------------
    {
        "url": "https://en.wikipedia.org/wiki/Internet?banner=false",
        "targets": [
            {"selector": "#firstHeading", "kind": "heading"},
            {"selector": "#mw-content-text > div > p", "kind": "paragraph", "nth": 1},
            # Link nth=2 to skip footnote markers at nth=0/1
            {"selector": "#mw-content-text > div > p a", "kind": "link", "nth": 2},
            {"selector": "h2", "kind": "heading", "nth": 1},
            # Relational: two adjacent real links (nth=2 and nth=3)
            {
                "selector": "#mw-content-text > div > p a",
                "kind": "link",
                "nth": 2,
                "relational": True,
                "pair_sel": "#mw-content-text > div > p a",
                "pair_nth": 3,
            },
        ],
    },
    # -----------------------------------------------------------------------
    # Hacker News — front page  (links, subtext, relational adjacent stories)
    # VERIFIED: .titleline (container div) has clean full title text. .score is ok.
    # -----------------------------------------------------------------------
    {
        "url": "https://news.ycombinator.com/",
        "targets": [
            # Story title links — .titleline[0] is first story
            {"selector": ".titleline", "kind": "link", "nth": 0},
            # Story 2 title
            {"selector": ".titleline", "kind": "link", "nth": 1},
            # Story 3 title
            {"selector": ".titleline", "kind": "link", "nth": 2},
            # Subtext row (points/author/time) — full metadata
            {"selector": ".subtext", "kind": "paragraph", "nth": 0},
            {"selector": ".subtext", "kind": "paragraph", "nth": 2},
            # Score span (isolated vote count)
            {"selector": ".score", "kind": "paragraph", "nth": 0},
            # Relational: two adjacent story titles (stories 1 & 2; dist ~22px)
            {
                "selector": ".titleline",
                "kind": "link",
                "nth": 0,
                "relational": True,
                "pair_sel": ".titleline",
                "pair_nth": 1,
            },
        ],
    },
    # -----------------------------------------------------------------------
    # httpbin — HTML form  (inputs, labels, button)
    # VERIFIED: <button>Submit order</button> (no type attr); labels contain inputs.
    # Radio/checkbox inputs also visible. No button[type=submit].
    # -----------------------------------------------------------------------
    {
        "url": "https://httpbin.org/forms/post",
        "targets": [
            # Text input fields
            {"selector": 'input[name="custname"]', "kind": "input", "label": "Customer name"},
            {"selector": 'input[name="custtel"]', "kind": "input", "label": "Telephone"},
            {"selector": 'input[name="custemail"]', "kind": "input", "label": "Email address"},
            # Textarea for delivery instructions
            {
                "selector": 'textarea[name="comments"]',
                "kind": "input",
                "label": "Delivery instructions",
            },
            # Submit button — plain <button>Submit order</button>
            {"selector": "button", "kind": "button"},
            # Labels (no for= attr; labels contain their inputs as children)
            {"selector": "label", "kind": "label", "nth": 0},
            {"selector": "label", "kind": "label", "nth": 1},
            {"selector": "label", "kind": "label", "nth": 2},
            # Radio input for pizza size
            {
                "selector": 'input[type="radio"][value="small"]',
                "kind": "input",
                "label": "Small pizza",
            },
            # Checkbox input for topping
            {
                "selector": 'input[type="checkbox"][value="bacon"]',
                "kind": "input",
                "label": "Bacon topping",
            },
            # Relational: two adjacent labels (Customer name + Telephone)
            {
                "selector": "label",
                "kind": "label",
                "nth": 0,
                "relational": True,
                "pair_sel": "label",
                "pair_nth": 1,
            },
        ],
    },
    # -----------------------------------------------------------------------
    # MDN — CSS flex documentation  (headings, notecard, code)
    # VERIFIED: h1/h2/.notecard p work; p code (inline refs) always visible;
    # pre code has bounding_box=None (off-screen rendering). Use p code instead.
    # p code: nth=0,1="flex"; nth=2="flex-grow"; nth=6="flex-shrink" — use distinct ones.
    # h2: skip 0="In this article", 1="Try it"; 2="Constituent properties", 3="Syntax".
    # Relational: h2[2]+h2[3] — verified these differ.
    # -----------------------------------------------------------------------
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/CSS/flex",
        "targets": [
            # Page h1 heading
            {"selector": "h1", "kind": "heading"},
            # Warning notecard
            {"selector": ".notecard p", "kind": "paragraph"},
            # Section headings (skip h2[0]="In this article" and h2[1]="Try it")
            {"selector": "h2", "kind": "heading", "nth": 2},
            {"selector": "h2", "kind": "heading", "nth": 3},
            # Distinct inline code refs: nth=2 = "flex-grow", nth=6 = "flex-shrink"
            {"selector": "p code", "kind": "code", "nth": 2},
            {"selector": "p code", "kind": "code", "nth": 6},
            # Relational: two adjacent h2 section headings (Constituent + Syntax)
            {
                "selector": "h2",
                "kind": "heading",
                "nth": 2,
                "relational": True,
                "pair_sel": "h2",
                "pair_nth": 3,
            },
        ],
    },
    # -----------------------------------------------------------------------
    # MDN — CSS display documentation  (extra code + heading variety)
    # -----------------------------------------------------------------------
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/CSS/display",
        "targets": [
            {"selector": "h1", "kind": "heading"},
            {"selector": ".notecard p", "kind": "paragraph"},
            {"selector": "p code", "kind": "code", "nth": 0},
            {"selector": "h2", "kind": "heading", "nth": 2},
        ],
    },
    # -----------------------------------------------------------------------
    # Wikipedia — Artificial intelligence  (extra article variety)
    # AI article has no traditional infobox — skip infobox relational; use link relational.
    # -----------------------------------------------------------------------
    {
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence?banner=false",
        "targets": [
            {"selector": "#firstHeading", "kind": "heading"},
            {"selector": "#mw-content-text > div > p", "kind": "paragraph", "nth": 1},
            # Non-footnote link (nth=0 usually skips footnote on AI article)
            {"selector": "#mw-content-text > div > p a", "kind": "link", "nth": 0},
            {"selector": "h2", "kind": "heading", "nth": 1},
            # Relational: two adjacent links in same paragraph
            {
                "selector": "#mw-content-text > div > p a",
                "kind": "link",
                "nth": 0,
                "relational": True,
                "pair_sel": "#mw-content-text > div > p a",
                "pair_nth": 1,
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Utterance / referent derivation by element kind
# ---------------------------------------------------------------------------

_KIND_META: dict[str, dict[str, str]] = {
    "heading": {
        "utterance": "What is this section about?",
        "referent_tmpl": "the section heading '{text}'",
        "tags": "text,heading",
    },
    "paragraph": {
        "utterance": "Summarize that",
        "referent_tmpl": "the paragraph that begins '{text}'",
        "tags": "text",
    },
    "image": {
        "utterance": "What is this?",
        "referent_tmpl": "the image of {text}",
        "tags": "image",
    },
    "link": {
        "utterance": "What is this?",
        "referent_tmpl": "the link labeled '{text}'",
        "tags": "link",
    },
    "table_cell": {
        "utterance": "What's this value?",
        "referent_tmpl": "the table cell containing '{text}'",
        "tags": "table",
    },
    "code": {
        "utterance": "Explain this",
        "referent_tmpl": "the code snippet '{text}'",
        "tags": "code",
    },
    "input": {
        "utterance": "What does this do?",
        "referent_tmpl": "the '{label}' field",
        "tags": "ui",
    },
    "button": {
        "utterance": "What does this do?",
        "referent_tmpl": "the '{text}' button",
        "tags": "ui",
    },
    "label": {
        "utterance": "What is this for?",
        "referent_tmpl": "the '{text}' label",
        "tags": "ui",
    },
}


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    words = s.split()
    if len(words) <= n:
        return s
    return " ".join(words[:n]) + "…"


def _keywords(text: str, n: int = 5) -> list[str]:
    """Extract n salient lowercased words from text."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        lt = t.lower()
        if lt not in seen:
            seen.add(lt)
            result.append(lt)
        if len(result) >= n:
            break
    return result


# ---------------------------------------------------------------------------
# Cursor point selection helpers
# ---------------------------------------------------------------------------


def _cursor_point_for_kind(box: dict[str, float], kind: str) -> tuple[float, float]:
    """Choose a cursor point ON rendered content (not the element's empty center).

    Block elements (h1, p, div) are often wider than their text, so the geometric
    center lands in whitespace.  Use a point near the start of the text instead.
    For images/buttons/inputs the center is fine.
    """
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    cy = y + h / 2
    if kind in ("heading", "paragraph", "link", "label", "table_cell"):
        # Block elements: text starts near the left — use 30% or 70px, whichever is less.
        cx = x + min(w * 0.30, 70.0)
    elif kind == "code":
        cx = x + min(w * 0.25, 60.0)
    else:
        # Buttons, inputs, images — center is fine
        cx = x + w / 2
    return cx, cy


# ---------------------------------------------------------------------------
# Core screenshot helpers
# ---------------------------------------------------------------------------


async def _inject_cursor_ring(page: Any, cx: float, cy: float) -> None:
    """Inject a red cursor ring overlay at (cx, cy) in CSS pixels."""
    await page.evaluate(
        """([x, y]) => {
            const d = document.createElement('div');
            d.className = '__aimer_ring__';
            d.style.cssText = `position:fixed;left:${x-15}px;top:${y-15}px;width:30px;height:30px;
              border:3px solid rgba(255,40,20,.95);border-radius:50%;
              box-shadow:0 0 0 2px rgba(255,255,255,.9);z-index:2147483647;pointer-events:none;`;
            document.body.appendChild(d);
        }""",
        [cx, cy],
    )


async def _inject_cursor_rings(page: Any, points: list[tuple[float, float]]) -> None:
    """Inject multiple cursor rings (for relational tasks)."""
    for cx, cy in points:
        await _inject_cursor_ring(page, cx, cy)


async def _remove_cursor_rings(page: Any) -> None:
    """Remove all injected cursor rings."""
    await page.evaluate(
        """() => {
            document.querySelectorAll('.__aimer_ring__').forEach(r => r.remove());
        }"""
    )


def _downscale_to_jpeg(
    png_bytes: bytes, long_edge_px: int = 1024, quality: float = 0.82
) -> bytes | None:
    """Downscale a PNG screenshot to `long_edge_px` on its long edge, return JPEG bytes (Quartz)."""
    import Quartz
    from Foundation import NSData, NSMutableData

    src = Quartz.CGImageSourceCreateWithData(
        NSData.dataWithBytes_length_(png_bytes, len(png_bytes)), None
    )
    if src is None:
        return None
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return None
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    scale = min(1.0, long_edge_px / max(w, h))
    ow, oh = max(1, int(w * scale)), max(1, int(h * scale))
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(
        None, ow, oh, 8, 0, cs, Quartz.kCGImageAlphaPremultipliedLast
    )
    if ctx is None:
        return None
    Quartz.CGContextSetInterpolationQuality(ctx, Quartz.kCGInterpolationHigh)
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, ow, oh), img)
    out = Quartz.CGBitmapContextCreateImage(ctx)
    data = NSMutableData.data()
    dest = Quartz.CGImageDestinationCreateWithData(data, "public.jpeg", 1, None)
    Quartz.CGImageDestinationAddImage(
        dest, out, {Quartz.kCGImageDestinationLossyCompressionQuality: quality}
    )
    if not Quartz.CGImageDestinationFinalize(dest):
        return None
    return bytes(data)


async def _capture_full_frame(page: Any, frame_path: Path, long_edge_px: int = 1024) -> bool:
    """Screenshot the viewport (cursor rings still present) and save a downscaled JPEG frame."""
    png = await page.screenshot(type="png", full_page=False)
    jpg = _downscale_to_jpeg(png, long_edge_px)
    if jpg is None:
        return False
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(jpg)
    return True


async def _elementFromPoint(page: Any, cx: float, cy: float) -> dict[str, Any] | None:
    """Return element info dict at (cx, cy) in viewport-CSS coords.

    Note: passes [cx, cy] as a SINGLE array argument to JS, which DESTRUCTURES it
    via ([x, y]) — this is the correct Playwright 1.x multi-value pattern.
    """
    info: dict[str, Any] | None = await page.evaluate(
        """([x, y]) => {
            const el = document.elementFromPoint(x, y);
            if (!el) return null;
            return {
                tag: el.tagName.toLowerCase(),
                role: el.getAttribute('role'),
                aria: el.getAttribute('aria-label'),
                alt: el.getAttribute('alt'),
                text: (el.innerText || el.textContent || '').trim().slice(0, 200),
                value: el.value || null,
            };
        }""",
        [cx, cy],
    )
    return info


_FOOTNOTE_RE = re.compile(r"^\[([a-z]|\d+|note\s*\d*)\]$", re.IGNORECASE)


def _is_junk_element(info: dict[str, Any] | None, kind: str) -> bool:
    """Return True if the element info is unusable for a task.

    Rejects:
    - None / missing info
    - Structural wrapper tags with no readable text
    - Wikipedia footnote markers like [a], [1], [note 2]
    """
    if info is None:
        return True
    tag = info.get("tag", "")
    text = (info.get("text") or "").strip()
    # Wrapper / structural tags with no semantic content
    _wrapper_tags = ("body", "html", "main", "section", "article", "div", "header", "footer")
    if tag in _wrapper_tags and len(text) < 4:
        return True
    # Reject footnote markers — they look like [a], [1], [note 2] and are not deictic targets
    if text and _FOOTNOTE_RE.match(text):
        return True
    return kind not in ("image", "button", "input") and not text


def _clamp_clip(
    cx: float,
    cy: float,
    half: int,
    viewport_w: int,
    viewport_h: int,
) -> dict[str, float]:
    """Return a clip dict clamped to viewport bounds."""
    x = max(0.0, min(cx - half, viewport_w - half * 2))
    y = max(0.0, min(cy - half, viewport_h - half * 2))
    return {"x": x, "y": y, "width": half * 2, "height": half * 2}


async def _resolve_locator(
    page: Any,
    selector: str,
    nth: int,
    timeout_ms: int = 8000,
) -> tuple[Any, dict[str, float]] | tuple[None, None]:
    """Resolve a locator to a bounding box.

    Strategy:
    1. Pick the nth visible match.  If count < nth+1, fall back to (count-1).
    2. Scroll the element into the viewport via JS.
    3. Re-read bounding_box() after scroll so coords are viewport-relative.

    CRITICAL: page.evaluate(fn, [a, b]) passes the array as ONE arg to JS.
    Multi-arg JS MUST use object destructuring: ({sel, idx}) with {"sel":...,"idx":...}.

    Returns (locator, box) or (None, None) on failure.
    """
    try:
        loc_all = page.locator(selector)
        count = await loc_all.count()
        if count == 0:
            logger.debug("selector '%s' matched 0 elements", selector)
            return None, None

        # Pick the right nth
        idx = min(nth, count - 1)
        loc = loc_all.nth(idx)

        # Scroll into view via JS using object-destructuring arg (avoids the
        # Playwright array-as-single-arg bug where (sel, idx) receives sel=[arr]).
        await page.evaluate(
            """({sel, idx}) => {
                const els = document.querySelectorAll(sel);
                if (els[idx]) els[idx].scrollIntoView({block:'center', behavior:'instant'});
            }""",
            {"sel": selector, "idx": idx},
        )
        await page.wait_for_timeout(200)

        box = await loc.bounding_box(timeout=timeout_ms)
        if box is None:
            return None, None
        if box["width"] < 3 or box["height"] < 3:
            return None, None
        return loc, box
    except Exception as exc:
        logger.debug("_resolve_locator '%s'[%d]: %s", selector, nth, exc)
        return None, None


async def _get_image_alt(page: Any, selector: str, nth: int) -> str:
    """Extract alt / title / figcaption text for an img element.

    Uses object-destructure arg pattern to avoid the Playwright multi-arg bug.
    """
    result: str = await page.evaluate(
        """({sel, idx}) => {
            const imgs = document.querySelectorAll(sel);
            const img = imgs[idx];
            if (!img) return '';
            const alt = img.getAttribute('alt') || '';
            if (alt) return alt;
            // Try figcaption
            const fig = img.closest('figure');
            if (fig) {
                const cap = fig.querySelector('figcaption');
                if (cap) return cap.innerText.trim().slice(0, 100);
            }
            // Try title
            return img.getAttribute('title') || img.getAttribute('aria-label') || '';
        }""",
        {"sel": selector, "idx": nth},
    )
    return result.strip()


# ---------------------------------------------------------------------------
# Single-target harvester
# ---------------------------------------------------------------------------


async def _harvest_single(
    page: Any,
    page_title: str,
    url: str,
    target: dict[str, Any],
    task_id: str,
    img_path: Path,
    viewport_w: int,
    viewport_h: int,
) -> dict[str, Any] | None:
    """Attempt to harvest one single-referent target. Returns task dict or None."""
    selector = target["selector"]
    kind = target["kind"]
    nth = target.get("nth", 0)
    label_override = target.get("label")

    loc, box = await _resolve_locator(page, selector, nth)
    if loc is None or box is None:
        logger.warning("[%s] could not resolve '%s'[%d]", task_id, selector, nth)
        return None

    # Choose cursor point on rendered content
    cx, cy = _cursor_point_for_kind(box, kind)

    # For images, get alt via JS directly from the <img> element
    raw_text = ""
    if kind == "image":
        raw_text = await _get_image_alt(page, selector, nth)
        if not raw_text:
            raw_text = "the image at this location"

    # Re-query elementFromPoint AFTER choosing the point
    info = await _elementFromPoint(page, cx, cy)

    if _is_junk_element(info, kind):
        logger.warning("[%s] junk elementFromPoint at (%.0f,%.0f): %s", task_id, cx, cy, info)
        return None

    # Build label text
    if kind == "image":
        pass  # raw_text already set above
    elif kind == "input" and label_override:
        raw_text = label_override
    elif kind == "button":
        raw_text = (
            (info or {}).get("text") or (info or {}).get("value") or label_override or "Submit"
        )
    else:
        raw_text = (info or {}).get("text") or label_override or ""

    if not raw_text.strip() and kind not in ("image",):
        logger.warning("[%s] empty text for '%s'[%d]", task_id, selector, nth)
        return None

    # Derive utterance and expected_referent
    meta = _KIND_META.get(kind, _KIND_META["paragraph"])
    utterance = meta["utterance"]

    if kind in ("paragraph", "code"):
        short_text = _truncate(raw_text, 10)
    else:
        short_text = _truncate(raw_text, 12)

    referent_tmpl = meta["referent_tmpl"]
    if "{text}" in referent_tmpl:
        expected_referent = referent_tmpl.format(text=short_text)
    elif "{label}" in referent_tmpl:
        expected_referent = referent_tmpl.format(label=label_override or short_text)
    else:
        expected_referent = referent_tmpl

    tags = meta["tags"].split(",")
    keywords = _keywords(raw_text, 6)

    # Inject cursor ring and take screenshot
    await _inject_cursor_ring(page, cx, cy)
    clip = _clamp_clip(cx, cy, 128, viewport_w, viewport_h)
    await page.screenshot(path=str(img_path), type="jpeg", quality=85, clip=clip)

    # Marked downscaled full frame (viewport with ring) for the escalation A/B.
    frame_path = img_path.parent.parent / "images_web_frame" / img_path.name
    frame_ok = await _capture_full_frame(page, frame_path)

    # Cursor tile coords in device pixels
    tile_x = int((cx - clip["x"]) * 2)
    tile_y = int((cy - clip["y"]) * 2)

    await _remove_cursor_rings(page)

    task = {
        "id": task_id,
        "image_path": f"fixtures/images_web/{img_path.name}",
        "app": "Chrome",
        "window_title": page_title,
        "cursor_tile_x": tile_x,
        "cursor_tile_y": tile_y,
        "utterance": utterance,
        "expected_referent": expected_referent,
        "expected_referent_keywords": keywords,
        "tags": tags,
        "notes": f"{url} | selector={selector}[{nth}]",
    }
    if frame_ok:
        task["full_frame_path"] = f"fixtures/images_web_frame/{img_path.name}"
    return task


# ---------------------------------------------------------------------------
# Relational target harvester
# ---------------------------------------------------------------------------


async def _harvest_relational(
    page: Any,
    page_title: str,
    url: str,
    target: dict[str, Any],
    task_id: str,
    img_path: Path,
    viewport_w: int,
    viewport_h: int,
) -> dict[str, Any] | None:
    """Attempt to harvest a relational (two-element) target. Returns task dict or None."""
    sel_a = target["selector"]
    sel_b = target["pair_sel"]
    kind = target["kind"]
    nth_a = target.get("nth", 0)
    nth_b = target.get("pair_nth", 0)

    loc_a, box_a = await _resolve_locator(page, sel_a, nth_a)
    if loc_a is None or box_a is None:
        logger.warning("[%s] relational: could not resolve A '%s'[%d]", task_id, sel_a, nth_a)
        return None

    loc_b, box_b = await _resolve_locator(page, sel_b, nth_b)
    if loc_b is None or box_b is None:
        logger.warning("[%s] relational: could not resolve B '%s'[%d]", task_id, sel_b, nth_b)
        return None

    cx_a, cy_a = _cursor_point_for_kind(box_a, kind)
    cx_b, cy_b = _cursor_point_for_kind(box_b, kind)

    # Check distance — if too far, skip (220 CSS px = 440 device px, fits two rings in 512 tile)
    dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    if dist > 220:
        logger.warning(
            "[%s] relational: elements too far apart (%.0f CSS px), skipping", task_id, dist
        )
        return None

    info_a = await _elementFromPoint(page, cx_a, cy_a)
    info_b = await _elementFromPoint(page, cx_b, cy_b)

    if _is_junk_element(info_a, kind) or _is_junk_element(info_b, kind):
        logger.warning("[%s] relational: junk elements %s / %s", task_id, info_a, info_b)
        return None

    text_a = (info_a or {}).get("text") or ""
    text_b = (info_b or {}).get("text") or ""

    if not text_a.strip() or not text_b.strip():
        logger.warning("[%s] relational: empty text in pair", task_id)
        return None

    # Avoid trivially identical referents
    if text_a.strip()[:30] == text_b.strip()[:30]:
        logger.warning("[%s] relational: A and B have identical text, skipping", task_id)
        return None

    short_a = _truncate(text_a, 6)
    short_b = _truncate(text_b, 6)

    # Clip centred on midpoint of the two cursor points
    mid_cx = (cx_a + cx_b) / 2
    mid_cy = (cy_a + cy_b) / 2
    clip = _clamp_clip(mid_cx, mid_cy, 128, viewport_w, viewport_h)

    # Inject two rings
    await _inject_cursor_rings(page, [(cx_a, cy_a), (cx_b, cy_b)])
    await page.screenshot(path=str(img_path), type="jpeg", quality=85, clip=clip)

    # Marked downscaled full frame (viewport with both rings) for the escalation A/B.
    frame_path = img_path.parent.parent / "images_web_frame" / img_path.name
    frame_ok = await _capture_full_frame(page, frame_path)

    tile_x = int((mid_cx - clip["x"]) * 2)
    tile_y = int((mid_cy - clip["y"]) * 2)

    await _remove_cursor_rings(page)

    keywords = _keywords(text_a + " " + text_b, 6)

    task = {
        "id": task_id,
        "image_path": f"fixtures/images_web/{img_path.name}",
        "app": "Chrome",
        "window_title": page_title,
        "cursor_tile_x": tile_x,
        "cursor_tile_y": tile_y,
        "utterance": "Compare these two",
        "expected_referent": f"the two values '{short_a}' and '{short_b}'",
        "expected_referent_keywords": keywords,
        "tags": ["relational", "multi-referent"],
        "notes": f"{url} | {sel_a}[{nth_a}] + {sel_b}[{nth_b}]",
    }
    if frame_ok:
        task["full_frame_path"] = f"fixtures/images_web_frame/{img_path.name}"
    return task


# ---------------------------------------------------------------------------
# Per-page harvester
# ---------------------------------------------------------------------------


async def _harvest_page(
    browser: Any,
    site: dict[str, Any],
    task_counter: list[int],
    tasks: list[dict[str, Any]],
    img_dir: Path,
    limit: int | None,
    viewport_w: int = 1280,
    viewport_h: int = 1600,
) -> int:
    """Harvest all targets from one site. Returns number of tasks added."""
    url = site["url"]
    targets = site["targets"]
    added = 0

    logger.info("Opening page: %s", url)
    page = await browser.new_page(
        viewport={"width": viewport_w, "height": viewport_h},
        device_scale_factor=2,
    )
    try:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as exc:
            logger.error("Failed to load %s: %s", url, exc)
            return 0

        await page.wait_for_timeout(800)

        # Dismiss cookie / consent banners by pressing Escape once
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)

        # Get page title
        page_title = await page.title()

        for target in targets:
            if limit is not None and task_counter[0] >= limit:
                break

            task_counter[0] += 1
            task_id = f"web_{task_counter[0]:04d}"
            img_path = img_dir / f"{task_id}.jpg"

            is_relational = target.get("relational", False)
            try:
                if is_relational:
                    task = await _harvest_relational(
                        page, page_title, url, target, task_id, img_path, viewport_w, viewport_h
                    )
                else:
                    task = await _harvest_single(
                        page, page_title, url, target, task_id, img_path, viewport_w, viewport_h
                    )
            except Exception as exc:
                logger.error("[%s] unexpected error: %s", task_id, exc)
                task = None

            if task is not None:
                tasks.append(task)
                added += 1
                logger.info(
                    "[%s] OK kind=%s referent=%s",
                    task_id,
                    target["kind"],
                    task["expected_referent"][:60],
                )
            else:
                logger.warning("[%s] SKIPPED", task_id)

    finally:
        await page.close()

    return added


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_tasks(tasks: list[dict[str, Any]], img_dir: Path) -> None:
    """Verify images exist, are >5 KB, and spot-check 5 tasks."""
    logger.info("--- Self-verification ---")
    ok = 0
    bad: list[str] = []

    for task in tasks:
        img_path = img_dir / Path(task["image_path"]).name
        if not img_path.exists():
            bad.append(f"{task['id']}: image file missing")
            continue
        size = img_path.stat().st_size
        if size < 5000:
            bad.append(f"{task['id']}: image too small ({size} bytes)")
            continue
        ok += 1

    if bad:
        for b in bad:
            logger.warning("VERIFY FAIL: %s", b)
    else:
        logger.info("All %d images exist and are >5 KB", ok)

    # Spot-check 5 evenly-spaced tasks
    step = max(1, len(tasks) // 5)
    sample = tasks[::step][:5]
    logger.info("Spot-check sample (%d tasks):", len(sample))
    for task in sample:
        img_path = img_dir / Path(task["image_path"]).name
        size = img_path.stat().st_size if img_path.exists() else 0
        keywords = task.get("expected_referent_keywords", [])
        kw_ok = len(keywords) > 0
        logger.info(
            "  %s | %d bytes | referent_kw=%s | kw_nonempty=%s",
            task["id"],
            size,
            keywords[:3],
            kw_ok,
        )

    # Summary by tag
    tag_counts: dict[str, int] = {}
    for task in tasks:
        for tag in task.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("\n=== HARVEST SUMMARY ===")
    print(f"Total tasks: {len(tasks)}")
    print("Tag breakdown:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")

    # Per-site breakdown
    site_counts: dict[str, int] = {}
    for task in tasks:
        notes = task.get("notes", "")
        m = re.search(r"https?://([^/]+)", notes)
        domain = m.group(1) if m else "unknown"
        site_counts[domain] = site_counts.get(domain, 0) + 1
    print("Per-site breakdown:")
    for site, count in sorted(site_counts.items()):
        print(f"  {site}: {count}")

    relational_count = sum(1 for t in tasks if "relational" in t.get("tags", []))
    print(f"Relational tasks: {relational_count}")
    print(f"Image verification: {ok}/{len(tasks)} passed (>5 KB exists)")
    if bad:
        print(f"FAILURES: {len(bad)}")
        for b in bad[:5]:
            print(f"  {b}")
    print("=======================\n")


# ---------------------------------------------------------------------------
# Main async entry
# ---------------------------------------------------------------------------


async def _main(
    out: Path,
    img_dir: Path,
    limit: int | None,
) -> int:
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415
    except ImportError:
        print("ERROR: playwright is not installed. Run: uv pip install playwright", file=sys.stderr)
        return 1

    img_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    task_counter = [0]  # mutable counter passed by reference

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            for site in SITE_TARGETS:
                if limit is not None and task_counter[0] >= limit:
                    break
                n = await _harvest_page(browser, site, task_counter, tasks, img_dir, limit)
                domain = site["url"].split("/")[2]
                logger.info(
                    "Site %s: contributed %d tasks (total so far: %d)", domain, n, len(tasks)
                )
        finally:
            await browser.close()

    # Write JSONL
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for task in tasks:
            fh.write(json.dumps(task) + "\n")
    logger.info("Written %d tasks to %s", len(tasks), out)

    # Verify
    _verify_tasks(tasks, img_dir)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_HERE / "fixtures" / "deictic_tasks_web.jsonl",
        help="Output JSONL path (default: fixtures/deictic_tasks_web.jsonl)",
    )
    ap.add_argument(
        "--img-dir",
        type=Path,
        default=_HERE / "fixtures" / "images_web",
        help="Directory for JPEG tile images (default: fixtures/images_web/)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N task attempts (useful for smoke testing)",
    )
    args = ap.parse_args()
    return asyncio.run(_main(out=args.out, img_dir=args.img_dir, limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
