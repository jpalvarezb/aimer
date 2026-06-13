"""Generate synthetic-but-real deictic fixtures via CoreText (headless, no screen).

Renders actual code/text/UI content into 512x512 JPEG tiles with the cursor marker
drawn on a known referent, and writes a labelled tasks JSONL. Unlike the blank
placeholder images, these contain real content a model can ground on — so the eval
produces a meaningful score without capturing the user's screen.

Real captured screenshots remain the gold standard; this is a reproducible,
inspectable stand-in. Run:  uv run python scripts/bench/fixtures/make_text_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import CoreText
import Quartz
from Foundation import NSAttributedString, NSMutableData

_HERE = Path(__file__).parent
_IMG_DIR = _HERE / "images_synth"
W = H = 512
_CS = Quartz.CGColorSpaceCreateDeviceRGB()
_BLACK = Quartz.CGColorCreate(_CS, (0.1, 0.1, 0.1, 1))
_RED = Quartz.CGColorCreate(_CS, (0.85, 0.1, 0.1, 1))
_BLUE = Quartz.CGColorCreate(_CS, (0.1, 0.3, 0.85, 1))


def _ctx() -> Any:
    ctx = Quartz.CGBitmapContextCreate(None, W, H, 8, 0, _CS, Quartz.kCGImageAlphaPremultipliedLast)
    Quartz.CGContextSetRGBFillColor(ctx, 1, 1, 1, 1)
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, W, H))
    return ctx


def _text(
    ctx: Any,
    x: float,
    y_top: float,
    s: str,
    *,
    size: float = 22,
    color: Any = _BLACK,
    font: str = "Menlo",
) -> None:
    """Draw one line; y_top is distance from the TOP (we flip to Quartz baseline)."""
    f = CoreText.CTFontCreateWithName(font, size, None)
    attrs = {CoreText.kCTFontAttributeName: f, CoreText.kCTForegroundColorAttributeName: color}
    line = CoreText.CTLineCreateWithAttributedString(
        NSAttributedString.alloc().initWithString_attributes_(s, attrs)
    )
    Quartz.CGContextSetTextPosition(ctx, x, H - y_top - size)
    CoreText.CTLineDraw(line, ctx)


def _marker(ctx: Any, x: float, y_top: float, r: float = 13) -> None:
    """Draw the cursor ring (white halo + red ring) at a top-left coordinate."""
    qy = H - y_top
    for color, lw, rr in ((1, 4, r + 1), (0, 2.5, r)):
        if color == 1:
            Quartz.CGContextSetRGBStrokeColor(ctx, 1, 1, 1, 0.9)
        else:
            Quartz.CGContextSetRGBStrokeColor(ctx, 1, 0.2, 0.1, 0.95)
        Quartz.CGContextSetLineWidth(ctx, lw)
        Quartz.CGContextAddArc(ctx, x, qy, rr, 0, 6.2832, 0)
        Quartz.CGContextStrokePath(ctx)


def _save(ctx: Any, name: str) -> str:
    img = Quartz.CGBitmapContextCreateImage(ctx)
    data = NSMutableData.data()
    dest = Quartz.CGImageDestinationCreateWithData(data, "public.jpeg", 1, None)
    Quartz.CGImageDestinationAddImage(
        dest, img, {Quartz.kCGImageDestinationLossyCompressionQuality: 0.85}
    )
    Quartz.CGImageDestinationFinalize(dest)
    path = _IMG_DIR / name
    path.write_bytes(bytes(data))
    return f"fixtures/images_synth/{name}"


def build() -> list[dict[str, Any]]:
    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []

    # 1. Code: function definition missing a colon, cursor on that line.
    c = _ctx()
    _text(c, 30, 60, "def total(items):", size=32)
    _text(c, 30, 95, "    s = 0", size=32)
    _text(c, 30, 130, "    for it in items", size=32, color=_RED)  # missing colon
    _text(c, 30, 165, "        s += it.price", size=32)
    _marker(c, 250, 138)
    tasks.append(
        {
            "id": "syn_code_colon",
            "image_path": _save(c, "syn_code_colon.jpg"),
            "cursor_tile_x": 250,
            "cursor_tile_y": 138,
            "utterance": "Fix this",
            "expected_referent": "the for-loop line 'for it in items' missing a trailing colon",
            "expected_referent_keywords": ["for", "colon", "syntax", "loop"],
            "tags": ["code", "single-referent"],
            "notes": "marker on the red for-loop line",
        }
    )

    # 2. Prose paragraph, cursor on the second sentence.
    c = _ctx()
    _text(c, 24, 50, "The Aimer pointer captures a tile around", size=31, font="Helvetica")
    _text(c, 24, 82, "the cursor at 10 Hz. Each tile is sent to", size=31, font="Helvetica")
    _text(c, 24, 114, "a duplex model with the cursor position.", size=31, font="Helvetica")
    _text(c, 24, 146, "Latency stays under one second per turn.", size=31, font="Helvetica")
    _marker(c, 200, 128)
    tasks.append(
        {
            "id": "syn_text_summarize",
            "image_path": _save(c, "syn_text_summarize.jpg"),
            "cursor_tile_x": 200,
            "cursor_tile_y": 128,
            "utterance": "Summarize that",
            "expected_referent": "the paragraph about the Aimer pointer capturing tiles",
            "expected_referent_keywords": ["tile", "cursor", "duplex", "capture", "pointer"],
            "tags": ["text", "single-referent"],
            "notes": "marker within the paragraph",
        }
    )

    # 3. UI button labelled Submit, cursor on it.
    c = _ctx()
    _text(c, 60, 60, "Contact form", size=34, font="Helvetica")
    Quartz.CGContextSetRGBStrokeColor(c, 0.1, 0.3, 0.85, 1)
    Quartz.CGContextSetLineWidth(c, 2)
    Quartz.CGContextStrokeRect(c, Quartz.CGRectMake(150, H - 300, 170, 56))  # box near y_top=244
    _text(c, 188, 256, "Submit", size=34, color=_BLUE, font="Helvetica")
    _marker(c, 235, 272)
    tasks.append(
        {
            "id": "syn_ui_button",
            "image_path": _save(c, "syn_ui_button.jpg"),
            "cursor_tile_x": 235,
            "cursor_tile_y": 272,
            "utterance": "What does this do?",
            "expected_referent": "the 'Submit' button of the contact form",
            "expected_referent_keywords": ["submit", "button", "form"],
            "tags": ["ui", "single-referent"],
            "notes": "marker on the Submit button",
        }
    )

    # 4. Relational: two prices side by side, cursor between them.
    c = _ctx()
    _text(c, 50, 70, "Plan A", size=32, font="Helvetica")
    _text(c, 50, 105, "$19.99 / mo", size=36, color=_BLUE)
    _text(c, 300, 70, "Plan B", size=32, font="Helvetica")
    _text(c, 300, 105, "$24.99 / mo", size=36, color=_BLUE)
    _marker(c, 90, 118)
    _marker(c, 340, 118)
    tasks.append(
        {
            "id": "syn_relational_prices",
            "image_path": _save(c, "syn_relational_prices.jpg"),
            "cursor_tile_x": 200,
            "cursor_tile_y": 118,
            "utterance": "Compare these two",
            "expected_referent": "the two prices, $19.99/mo (Plan A) and $24.99/mo (Plan B)",
            "expected_referent_keywords": ["19.99", "24.99", "plan", "price", "compare"],
            "tags": ["relational", "multi-referent"],
            "notes": "two markers on the two prices",
        }
    )

    # 5. Code: undefined variable usage, cursor on it.
    c = _ctx()
    _text(c, 30, 60, "user = get_user(uid)", size=32)
    _text(c, 30, 95, "name = user.name", size=32)
    _text(c, 30, 130, "send(emial, name)", size=32, color=_RED)  # typo: emial
    _marker(c, 175, 138)
    tasks.append(
        {
            "id": "syn_code_typo_var",
            "image_path": _save(c, "syn_code_typo_var.jpg"),
            "cursor_tile_x": 175,
            "cursor_tile_y": 138,
            "utterance": "What's wrong here?",
            "expected_referent": "the misspelled variable 'emial' (should be 'email')",
            "expected_referent_keywords": ["emial", "email", "typo", "variable", "undefined"],
            "tags": ["code", "single-referent"],
            "notes": "marker on the typo'd identifier",
        }
    )

    return tasks


def main() -> None:
    tasks = build()
    out = _HERE / "deictic_tasks_synth.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    print(f"wrote {len(tasks)} tasks -> {out}")
    print(f"images -> {_IMG_DIR}")


if __name__ == "__main__":
    main()
