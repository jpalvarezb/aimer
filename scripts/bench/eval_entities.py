"""Week-5 entity-extraction eval — local Qwen3-VL on REAL cluttered tiles.

Measures the local VLM's entity extraction + routing on real UI tiles (Google Maps,
Google Calendar, GitHub, Wikipedia), not the synthetic happy-path demo:

  - POSITIVES: tiles with a known entity at the cursor (place/date/product/code_span).
    Scored for type-recall (did it emit the right type) and value-recall (matching value).
  - NEGATIVES: real heading/image tiles with NO routable entity at the cursor. Scored for
    precision (did it AVOID hallucinating a Maps/Calendar/IDE entity).
  - LATENCY: per-tile extraction wall time (p50/p95) — relevant to off-hot-path running.

Sustained 4B MLX inference can trigger a cumulative Metal GPU fault, so this eval is
**crash-resumable**: it processes ENTITY_EVAL_BATCH tiles per invocation, dumps results
incrementally, writes a tentative "crashed" row BEFORE each extraction (so a poison tile
is skipped on the next run), and resumes from results/week5_entity_eval.json. Drive it
with run_entity_eval.sh, which restarts a fresh process per batch (each exit resets the GPU).

Run:  ENTITY_EVAL_BATCH=3 uv run --no-sync python scripts/bench/eval_entities.py
Out:  results/week5_entity_eval.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "aimer-core" / "src"))

from duplex_bridge.entities import (  # noqa: E402
    EntityRouter,
    ExtractionContext,
    GeminiFlashLiteExtractor,
    LocalQwenVLExtractor,
)

_LABELS_REAL = _HERE / "fixtures" / "entity_labels_real.jsonl"
_DEICTIC = _HERE / "fixtures" / "deictic_tasks_web.jsonl"

# Week 5 measures the VISION model: can it read typed entities out of the tile PIXELS? We
# deliberately pass an EMPTY ExtractionContext (no app/window/AX) — the AX label often *is*
# the answer, so feeding it would test passthrough, not vision. Production may additionally
# pass AX, so real-world accuracy is >= these numbers; this is the honest vision-only floor.
#   ENTITY_EVAL_BACKEND = local (default) | gemini      (gemini = GPU-free fallback)
#   ENTITY_EVAL_MODEL   = HF id override for local (e.g. mlx-community/Qwen3-VL-4B-Instruct-4bit)
#   ENTITY_EVAL_OUT     = results filename (local + gemini-fallback share ONE combined file)
_BACKEND = os.environ.get("ENTITY_EVAL_BACKEND", "local").lower()
_MODEL_ID = os.environ.get("ENTITY_EVAL_MODEL", "").strip()
_OUT = _HERE / "results" / os.environ.get("ENTITY_EVAL_OUT", "week5_entity_eval.json")
_MODEL = (
    "gemini-flash-lite"
    if _BACKEND == "gemini"
    else (_MODEL_ID.split("/")[-1] if _MODEL_ID else "Qwen3-VL-2B-Instruct-4bit")
)


def _make_extractor():
    """Build the configured extractor: local Qwen3-VL (default) or the Gemini fallback."""
    if _BACKEND == "gemini":
        return GeminiFlashLiteExtractor()
    return LocalQwenVLExtractor(model_id=_MODEL_ID) if _MODEL_ID else LocalQwenVLExtractor()


# Entity types that route to the Week-5 host targets (Maps / Calendar / IDE).
_ROUTABLE = {"place", "date", "code_span"}


@dataclass
class Case:
    cid: str
    image: Path
    kind: str  # "positive" | "negative"
    expected_type: str | None
    expected_value: str  # substring/token hint ("" = type-only)


def _load_cases() -> list[Case]:
    cases: list[Case] = []
    for line in _LABELS_REAL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cases.append(
            Case(
                r["id"],
                _HERE / r["image"],
                "positive",
                r["expected_type"],
                r.get("expected_value_substr", ""),
            )
        )

    deictic = {
        json.loads(x)["id"]: json.loads(x) for x in _DEICTIC.read_text().splitlines() if x.strip()
    }
    extra_pos = [
        ("web_0050", "code_span", "flex-grow"),
        ("web_0051", "code_span", "flex-shrink"),
        ("web_0055", "code_span", "display"),
        ("web_0010", "date", "century"),
    ]
    for cid, etype, val in extra_pos:
        if cid in deictic:
            cases.append(Case(cid, _HERE / deictic[cid]["image_path"], "positive", etype, val))

    for cid, t in deictic.items():
        tags = t.get("tags", [])
        if ("heading" in tags or "image" in tags) and cid not in {c.cid for c in cases}:
            cases.append(Case(cid, _HERE / t["image_path"], "negative", None, ""))
    return cases


def _value_match(expected: str, values: list[str]) -> bool:
    if not expected:
        return True
    blob = " ".join(values).lower()
    tokens = [w for w in expected.lower().split() if len(w) > 2]
    return any(tok in blob for tok in tokens) if tokens else expected.lower() in blob


def _dump(rows: list[dict]) -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps({"model": _MODEL, "rows": rows}, indent=2))


def _scores(ok_rows: list[dict]) -> dict:
    """Type/value recall + precision for one homogeneous set of scored rows."""
    pos = [r for r in ok_rows if r["kind"] == "positive"]
    neg = [r for r in ok_rows if r["kind"] == "negative"]
    return {
        "n": len(ok_rows),
        "pos": len(pos),
        "neg": len(neg),
        "type_recall": (sum(r.get("type_hit", False) for r in pos) / len(pos)) if pos else 0.0,
        "value_recall": (sum(r.get("value_hit", False) for r in pos) / len(pos)) if pos else 0.0,
        "precision": (sum(r.get("precise", False) for r in neg) / len(neg)) if neg else 0.0,
    }


def _summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    crashed = [r for r in rows if r.get("status") == "crashed"]
    pos = [r for r in ok if r["kind"] == "positive"]
    neg = [r for r in ok if r["kind"] == "negative"]
    type_recall = sum(r["type_hit"] for r in pos) / len(pos) if pos else 0.0
    value_recall = sum(r["value_hit"] for r in pos) / len(pos) if pos else 0.0
    precision = sum(r["precise"] for r in neg) / len(neg) if neg else 0.0

    by_type: dict[str, list[dict]] = {}
    for r in pos:
        by_type.setdefault(r["expected_type"], []).append(r)

    lats = sorted(r["latency_ms"] for r in ok if "latency_ms" in r)
    p50 = lats[len(lats) // 2] if lats else 0.0
    p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else 0.0

    backends: dict[str, int] = {}
    for r in ok:
        backends[r.get("backend", "?")] = backends.get(r.get("backend", "?"), 0) + 1

    type_hits = sum(r["type_hit"] for r in pos)
    value_hits = sum(r["value_hit"] for r in pos)
    precise_hits = sum(r["precise"] for r in neg)
    print("\n" + "=" * 64)
    print("WEEK-5 ENTITY EVAL (local Qwen3-VL, real tiles)")
    print("=" * 64)
    print(f"scored: {len(ok)}  pos: {len(pos)}  neg: {len(neg)}  crashed: {len(crashed)}")
    print(f"type-recall (positives) : {type_hits}/{len(pos)} = {type_recall * 100:.0f}%")
    print(f"value-recall (positives): {value_hits}/{len(pos)} = {value_recall * 100:.0f}%")
    print(f"precision (negatives)   : {precise_hits}/{len(neg)} = {precision * 100:.0f}%")
    print("per-type type-recall:")
    for et, group in sorted(by_type.items()):
        print(f"   {et:<10} {sum(g['type_hit'] for g in group)}/{len(group)}")
    print(f"latency: p50={p50:.0f}ms  p95={p95:.0f}ms  (n={len(lats)})")
    print(f"row counts by backend: {backends}")

    # Prod runs ONE backend — report each separately, never a blended score. (Gemini here is a
    # MACHINE fallback so the benchmark survives GPU faults; it is not a prod accuracy fallback.)
    per_backend: dict[str, dict] = {}
    print("\nPER-BACKEND accuracy (prod uses one backend — NOT blended):")
    for b in sorted(backends):
        s = _scores([r for r in ok if r.get("backend") == b])
        per_backend[b] = s
        print(
            f"   {b:<14} n={s['n']:<3} (pos {s['pos']}/neg {s['neg']})  "
            f"type-recall={s['type_recall'] * 100:.0f}%  "
            f"value-recall={s['value_recall'] * 100:.0f}%  "
            f"precision={s['precision'] * 100:.0f}%"
        )

    summary = {
        "scored": len(ok),
        "positives": len(pos),
        "negatives": len(neg),
        "crashed": len(crashed),
        "combined_type_recall": type_recall,
        "combined_value_recall": value_recall,
        "combined_precision": precision,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "by_backend": backends,
        "per_backend": per_backend,
    }
    _OUT.write_text(json.dumps({"model": _MODEL, "summary": summary, "rows": rows}, indent=2))
    print(f"\nwritten: {_OUT}")
    print("ALL DONE", flush=True)


async def main() -> int:
    cases = _load_cases()
    batch = int(os.environ.get("ENTITY_EVAL_BATCH", "3"))

    rows: list[dict] = []
    if _OUT.exists():
        try:
            rows = json.loads(_OUT.read_text()).get("rows", [])
        except (json.JSONDecodeError, OSError):
            rows = []
    # Gemini is GPU-free and safe to retry, so it FILLS IN tiles a prior local run crashed on:
    # drop those crash-markers. The local backend keeps them (poison-tile guard) so it never
    # re-crashes on the same tile and never takes the machine down in a loop.
    if _BACKEND == "gemini":
        rows = [r for r in rows if r.get("status") != "crashed"]
    done_ok = {r["id"] for r in rows if r.get("status") == "ok"}
    crashed_ids = {r["id"] for r in rows if r.get("status") == "crashed"}
    skip = (done_ok | crashed_ids) if _BACKEND == "local" else done_ok
    todo = [c for c in cases if c.cid not in skip]

    if not todo:
        _summarize(rows)
        return 0

    extractor = _make_extractor()
    router = EntityRouter()
    for processed, c in enumerate(todo):
        if processed >= batch:
            break
        # Tentative row written BEFORE extraction: if the GPU aborts the process mid-extract,
        # this "crashed" marker persists and the poison tile is skipped on the next run.
        rows.append(
            {
                "id": c.cid,
                "kind": c.kind,
                "expected_type": c.expected_type,
                "status": "crashed",
                "backend": _BACKEND,
            }
        )
        _dump(rows)

        tile = c.image.read_bytes()
        t0 = time.perf_counter()
        # Vision-ONLY: empty context. We measure what the model reads from the tile pixels,
        # not what an AX hint would hand it.
        entities = await extractor.extract(tile, ExtractionContext())
        dt_ms = (time.perf_counter() - t0) * 1000.0
        actions = router.dispatch(entities)
        ent_pairs = [[e.type, e.value] for e in entities]
        routable = {e.type for e in entities if e.type in _ROUTABLE}

        row: dict = {
            "id": c.cid,
            "kind": c.kind,
            "expected_type": c.expected_type,
            "entities": ent_pairs,
            "routed": [a.target for a in actions],
            "latency_ms": round(dt_ms, 1),
            "status": "ok",
            "backend": _BACKEND,
            "model": _MODEL,
        }
        if c.kind == "positive":
            row["type_hit"] = c.expected_type in {e.type for e in entities}
            row["value_hit"] = row["type_hit"] and _value_match(
                c.expected_value, [e.value for e in entities]
            )
        else:
            row["precise"] = len(routable) == 0
            row["false_positives"] = sorted(routable)
        rows[-1] = row  # replace the tentative marker with the real result
        _dump(rows)
        print(f"{c.cid:<22} {c.kind:<9} {ent_pairs}", flush=True)

    remaining = [c for c in cases if c.cid not in {r["id"] for r in rows}]
    if remaining:
        print(f"REMAINING {len(remaining)}", flush=True)
    else:
        _summarize(rows)
    await extractor.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
