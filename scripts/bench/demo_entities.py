"""Week-5 measured demo: local Qwen3-VL extracts typed entities from tiles and routes them.

Generates a few 512x512 cursor tiles containing real entities (an address, a date, a code
snippet), plus reuses a real deictic fixture tile, then runs the LOCAL MLX Qwen3-VL
extractor and the typed router (place->Maps, date->Calendar, code_span->IDE). Prints the
measured entities + routed actions and writes them to results/week5_entities.json.

Run:  uv run --no-sync python scripts/bench/demo_entities.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "aimer-core" / "src"))

from duplex_bridge.entities import (  # noqa: E402
    EntityPipeline,
    EntityRouter,
    ExtractionContext,
    LocalQwenVLExtractor,
)

_TILE_DIR = _HERE / "fixtures" / "entity_tiles"
_OUT = _HERE / "results" / "week5_entities.json"

_SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
_MONO = "/System/Library/Fonts/Monaco.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _make_tile(name: str, lines: list[str], *, mono: bool = False) -> Path:
    """Render a 512x512 white tile with text + a cursor ring at center."""
    _TILE_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    font = _font(_MONO if mono else _SANS, 30)
    y = 200
    for line in lines:
        d.text((40, y), line, fill="black", font=font)
        y += 44
    # cursor ring near the first line (mimics the production tile's cursor marker)
    d.ellipse((232, 208, 280, 256), outline="red", width=4)
    path = _TILE_DIR / f"{name}.jpg"
    img.save(path, quality=90)
    return path


# (tile name, lines, mono?, context) — controlled entities we expect to extract.
_TILES = [
    (
        "address",
        ["1600 Amphitheatre Parkway", "Mountain View, CA 94043"],
        False,
        ExtractionContext(app="Safari", window_title="Directions"),
    ),
    (
        "date",
        ["Team sync", "March 14, 2026 at 3:00 PM"],
        False,
        ExtractionContext(app="Mail", window_title="Invite"),
    ),
    (
        "code",
        ["async def fetch(url):", "    return await client.get(url)"],
        True,
        ExtractionContext(app="VS Code", window_title="api.py"),
    ),
]


async def main() -> int:
    extractor = LocalQwenVLExtractor()
    router = EntityRouter()
    pipe = EntityPipeline(extractor, router)

    results: list[dict] = []

    # 1) synthetic controlled tiles
    cases: list[tuple[str, Path, ExtractionContext]] = []
    for name, lines, mono, ctx in _TILES:
        cases.append((name, _make_tile(name, lines, mono=mono), ctx))

    # 2) a real deictic fixture tile (flex-grow code snippet) for realism
    real = _HERE / "fixtures" / "images_web" / "web_0050.jpg"
    if real.exists():
        cases.append(
            ("real_web_0050_flexgrow", real, ExtractionContext(app="Chrome", window_title="MDN"))
        )

    print(f"\n=== Week-5 local VLM entity extraction ({extractor.name}) ===\n")
    for name, tile_path, ctx in cases:
        tile_bytes = tile_path.read_bytes()
        entities, actions = await pipe.process_tile(tile_bytes, ctx)
        print(f"tile: {name}")
        for e in entities:
            print(f"   entity: type={e.type:<10} value={e.value!r}")
        for a in actions:
            print(f"   ROUTED -> {a.target:<9} {a.payload}")
        if not entities:
            print("   (no entities)")
        print()
        results.append(
            {
                "tile": name,
                "entities": [{"type": e.type, "value": e.value} for e in entities],
                "routed": [{"target": a.target, "payload": a.payload} for a in actions],
            }
        )

    targets = {a["target"] for r in results for a in r["routed"]}
    print(f"distinct route targets hit: {sorted(targets)}")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps({"model": extractor.name, "results": results}, indent=2))
    print(f"written: {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
