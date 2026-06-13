"""Harvest a few REAL, cluttered entity tiles for the Week-5 entity eval.

Unlike the synthetic demo tiles, these are crops of real web pages around a known entity
(address, date, product), so the local VLM is tested on realistic background clutter.
Each target locates the entity by visible text, scrolls to it, draws a cursor ring, and
crops a 256-CSS-px tile at 2x (512px) — the same geometry as the production tile.

Out: fixtures/entity_tiles_real/<id>.jpg  +  fixtures/entity_labels_real.jsonl
Run:  uv run --no-sync python scripts/bench/harvest_entity_tiles.py
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

_HERE = Path(__file__).resolve().parent
_TILE_DIR = _HERE / "fixtures" / "entity_tiles_real"
_LABELS = _HERE / "fixtures" / "entity_labels_real.jsonl"

_VIEWPORT = {"width": 1280, "height": 1600}
_SCALE = 2
_HALF_CSS = 128  # 256 CSS px box -> 512px tile at 2x

# (id, url, locate_text, expected_type, expected_value_substr)
_TARGETS = [
    # place -> Maps
    (
        "ent_place_apple",
        "https://en.wikipedia.org/wiki/Apple_Park",
        "Apple Park Way",
        "place",
        "Apple Park Way",
    ),
    (
        "ent_place_google",
        "https://en.wikipedia.org/wiki/Googleplex",
        "Amphitheatre",
        "place",
        "Amphitheatre",
    ),
    (
        "ent_place_liberty",
        "https://en.wikipedia.org/wiki/Statue_of_Liberty",
        "Liberty Island",
        "place",
        "Liberty Island",
    ),
    # date -> Calendar
    (
        "ent_date_eiffel",
        "https://en.wikipedia.org/wiki/Eiffel_Tower",
        "31 March 1889",
        "date",
        "1889",
    ),
    ("ent_date_apollo", "https://en.wikipedia.org/wiki/Apollo_11", "July 16, 1969", "date", "1969"),
    (
        "ent_date_python",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "1991",
        "date",
        "1991",
    ),
    # product (type detection; routes to "none")
    ("ent_product_iphone", "https://en.wikipedia.org/wiki/IPhone", "iPhone", "product", "iPhone"),
    (
        "ent_product_ps5",
        "https://en.wikipedia.org/wiki/PlayStation_5",
        "PlayStation 5",
        "product",
        "PlayStation",
    ),
    # REAL target-app UIs (the actual routing surfaces): Google Maps + GitHub code view.
    (
        "ent_place_gmaps",
        "https://www.google.com/maps/place/1600+Amphitheatre+Parkway,+Mountain+View,+CA",
        "Amphitheatre",
        "place",
        "Amphitheatre",
    ),
    (
        "ent_code_github",
        "https://github.com/psf/requests/blob/main/src/requests/api.py",
        "def get",
        "code_span",
        "def get",
    ),
    # REAL Google Calendar UI via the ungated public-calendar embed (US holidays).
    (
        "ent_date_gcal",
        "https://calendar.google.com/calendar/embed?src=en.usa%23holiday%40group.v.calendar.google.com&ctz=America/New_York",
        "Juneteenth",
        "date",
        "",
    ),
]


def main() -> None:
    _TILE_DIR.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for tid, url, locate, etype, value in _TARGETS:
            page = browser.new_page(viewport=_VIEWPORT, device_scale_factor=_SCALE)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(800)
                page.keyboard.press("Escape")
                page.wait_for_timeout(200)
                loc = page.get_by_text(locate, exact=False).first
                loc.scroll_into_view_if_needed(timeout=8000)
                page.wait_for_timeout(300)
                box = loc.bounding_box()
                if box is None:
                    print(f"  SKIP {tid}: no bounding box for {locate!r}")
                    continue
                cx = box["x"] + box["width"] / 2
                cy = box["y"] + box["height"] / 2
                # cursor ring at the entity
                page.evaluate(
                    """([x, y]) => {
                        const r = document.createElement('div');
                        const s = r.style;
                        s.position = 'fixed';
                        s.left = (x - 22) + 'px';
                        s.top = (y - 22) + 'px';
                        s.width = '44px';
                        s.height = '44px';
                        s.border = '4px solid red';
                        s.borderRadius = '50%';
                        s.zIndex = '2147483647';
                        s.pointerEvents = 'none';
                        document.body.appendChild(r);
                    }""",
                    [cx, cy],
                )
                vw, vh = _VIEWPORT["width"], _VIEWPORT["height"]
                clip = {
                    "x": max(0.0, min(cx - _HALF_CSS, vw - 2 * _HALF_CSS)),
                    "y": max(0.0, min(cy - _HALF_CSS, vh - 2 * _HALF_CSS)),
                    "width": 2 * _HALF_CSS,
                    "height": 2 * _HALF_CSS,
                }
                out = _TILE_DIR / f"{tid}.jpg"
                page.screenshot(path=str(out), type="jpeg", quality=85, clip=clip)
                labels.append(
                    {
                        "id": tid,
                        "image": f"fixtures/entity_tiles_real/{tid}.jpg",
                        "expected_type": etype,
                        "expected_value_substr": value,
                        "url": url,
                    }
                )
                print(f"  OK {tid}: {etype} @ {locate!r}")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {tid}: {exc}")
            finally:
                page.close()
        browser.close()

    with _LABELS.open("w") as fh:
        for row in labels:
            fh.write(json.dumps(row) + "\n")
    print(f"\nharvested {len(labels)}/{len(_TARGETS)} real entity tiles -> {_LABELS}")


if __name__ == "__main__":
    main()
