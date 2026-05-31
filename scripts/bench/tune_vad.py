"""Iterative VAD tuning sweep for Week 3 TTFB optimisation.

Tests a matrix of Gemini VAD configurations and produces a ranked comparison
table. The primary metric is last_activity→response (end-of-speech to first
audio out); the acceptance target is p50 ≤ 700 ms.

Usage:
    uv run python scripts/tune_vad.py
    uv run python scripts/tune_vad.py --runs-per-config 5  # faster sweep
    uv run python scripts/tune_vad.py --runs-per-config 10 --model models/gemini-2.0-flash-live-001
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent  # scripts/bench/ -> repo root
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))

from duplex_bridge.audio_metrics import compute_rms_int16  # noqa: E402

# Import shared helpers from measure_ttfb
sys.path.insert(0, str(Path(__file__).parent))
from measure_ttfb import (  # noqa: E402
    BatchResult,
    chunk_pcm,
    generate_speech_pcm,
    load_api_key,
    measure_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("tune_vad")

PASS_THRESHOLD_MS = 700.0


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VadConfig:
    label: str
    vad_silence_ms: int | None = None
    vad_start_sensitivity: str | None = None
    turn_coverage: str | None = None


CONFIGS: list[VadConfig] = [
    VadConfig("baseline (default VAD)"),
    VadConfig("silence=1000ms", vad_silence_ms=1000),
    VadConfig("silence=500ms", vad_silence_ms=500),
    VadConfig("silence=300ms", vad_silence_ms=300),
    VadConfig("silence=200ms", vad_silence_ms=200),
    VadConfig("silence=500ms+high-sens", vad_silence_ms=500, vad_start_sensitivity="high"),
    VadConfig("silence=300ms+high-sens", vad_silence_ms=300, vad_start_sensitivity="high"),
    VadConfig("silence=500ms+act_only", vad_silence_ms=500, turn_coverage="activity_only"),
    VadConfig(
        "silence=500ms+high+act_only",
        vad_silence_ms=500,
        vad_start_sensitivity="high",
        turn_coverage="activity_only",
    ),
    VadConfig(
        "silence=300ms+high+act_only",
        vad_silence_ms=300,
        vad_start_sensitivity="high",
        turn_coverage="activity_only",
    ),
    VadConfig(
        "silence=200ms+high+act_only",
        vad_silence_ms=200,
        vad_start_sensitivity="high",
        turn_coverage="activity_only",
    ),
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL = 34  # label column width


def _row(r: BatchResult) -> str:
    p50 = f"{r.last_activity_p50:.0f}" if r.last_activity_p50 is not None else "---"
    p95 = f"{r.last_activity_p95:.0f}" if r.last_activity_p95 is not None else "---"
    ok = "✓ PASS" if r.passed else "✗"
    valid_str = f"{r.valid}/{r.runs}"
    return f"  {r.label:<{_COL}} {p50:>6} ms  {p95:>6} ms  {valid_str:>5}  {ok}"


def _print_table(results: list[BatchResult]) -> None:
    header = f"  {'Config':<{_COL}} {'p50':>9}  {'p95':>9}  {'valid':>5}  verdict"
    sep = "  " + "-" * (len(header) - 2)
    print("\n" + "=" * len(header))
    print("VAD TUNING RESULTS  (primary metric: last_activity→response)")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(_row(r))
    print()

    passed = [r for r in results if r.passed]
    if passed:
        best = min(passed, key=lambda r: r.last_activity_p50 or float("inf"))
        print(f"Best passing config: [{best.label}]  p50={best.last_activity_p50:.0f} ms")
    else:
        closest = min(results, key=lambda r: r.last_activity_p50 or float("inf"))
        print(
            f"No config passed ≤{PASS_THRESHOLD_MS:.0f} ms target. "
            f"Closest: [{closest.label}] p50={closest.last_activity_p50:.0f} ms"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _main(runs: int, model: str, configs: list[VadConfig]) -> int:
    api_key = load_api_key()
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env", file=sys.stderr)
        return 1
    os.environ["GEMINI_API_KEY"] = api_key

    phrase = "Hello, please respond briefly."
    print(f"Generating speech: '{phrase}' ...")
    pcm = generate_speech_pcm(phrase)
    frames = chunk_pcm(pcm)
    max_rms = max(compute_rms_int16(f) for f in frames)
    print(
        f"  {len(frames)} frames ({len(pcm) / 2 / 16000:.2f}s), max RMS={max_rms:.0f}\n"
        f"  model={model}  runs_per_config={runs}  configs={len(configs)}\n"
    )

    results: list[BatchResult] = []

    for idx, cfg in enumerate(configs):
        print(f"[{idx + 1}/{len(configs)}] {cfg.label} ...")
        r = await measure_batch(
            frames,
            label=cfg.label,
            model=model,
            runs=runs,
            vad_silence_ms=cfg.vad_silence_ms,
            vad_start_sensitivity=cfg.vad_start_sensitivity,
            turn_coverage=cfg.turn_coverage,
            # Shorter inter-run sleep during sweep to keep total time reasonable
            inter_run_sleep=2.0,
        )
        results.append(r)
        p50_str = f"{r.last_activity_p50:.0f} ms" if r.last_activity_p50 else "---"
        print(f"  → last_act p50={p50_str}  valid={r.valid}/{r.runs}")

        # Stop early if we beat the target comfortably (p50 ≤ 500 ms)
        if r.last_activity_p50 is not None and r.last_activity_p50 <= 500:
            print(f"  → early stop: p50={r.last_activity_p50:.0f} ms ≤ 500 ms, skipping remaining")
            break

        # Brief pause between configs
        if idx < len(configs) - 1:
            await asyncio.sleep(3.0)

    _print_table(results)

    out_path = Path(__file__).parent / "vad_tuning_results.json"
    out_path.write_text(json.dumps([r.as_dict() for r in results], indent=2))
    print(f"Full results written to {out_path.relative_to(_ROOT)}")

    any_passed = any(r.passed for r in results)
    return 0 if any_passed else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep Gemini VAD configs to optimise TTFB.")
    p.add_argument(
        "--runs-per-config", type=int, default=7, help="Runs per configuration (default: 7)"
    )
    p.add_argument("--model", default="models/gemini-3.1-flash-live-preview")
    p.add_argument("--configs", nargs="*", help="Subset of config labels to run (default: all)")
    args = p.parse_args()

    configs = CONFIGS
    if args.configs:
        configs = [c for c in CONFIGS if any(sel in c.label for sel in args.configs)]
        if not configs:
            print(f"No configs matched: {args.configs}", file=sys.stderr)
            return 1

    return asyncio.run(_main(args.runs_per_config, args.model, configs))


if __name__ == "__main__":
    raise SystemExit(main())
