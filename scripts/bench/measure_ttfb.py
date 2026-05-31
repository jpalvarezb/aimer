"""Automated TTFB measurement for Week 3 acceptance.

Drives GeminiLiveSession directly (no WebSocket server or pointer agent).
Generates speech with macOS `say` + ffmpeg, sends it as 16 kHz mono int16 PCM,
and records first/last audio-activity → response latency.

Primary acceptance metric: last_activity→response (end-of-speech to first audio out).
Secondary diagnostic: first_activity→response (inflated by phrase duration).

Usage:
    uv run python scripts/measure_ttfb.py
    uv run python scripts/measure_ttfb.py --runs 5 --vad-silence-ms 500
    uv run python scripts/measure_ttfb.py --vad-silence-ms 500 --vad-start-sensitivity high
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Make workspace packages importable when running as a script
_ROOT = Path(__file__).parent.parent.parent  # scripts/bench/ -> repo root
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))

from duplex_bridge.audio_metrics import compute_rms_int16, percentile  # noqa: E402
from duplex_bridge.providers.gemini_live import GeminiLiveSession  # noqa: E402

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_600  # 100 ms at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 = 2 bytes per sample
RMS_THRESHOLD = 300.0
PASS_THRESHOLD_MS = 700.0
SILENCE_FRAMES_AFTER_SPEECH = 15  # 1.5 s of silence to signal VAD end-of-turn
RESPONSE_TIMEOUT_S = 12.0
INTER_RUN_SLEEP_S = 3.0
WARMUP_DRAIN_S = 5.0  # let the model finish a warm-up response before the measured turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("measure_ttfb")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunResult:
    first_activity_ms: float | None  # first RMS-above-threshold frame → first audio out
    last_activity_ms: float | None  # last RMS-above-threshold frame → first audio out


@dataclasses.dataclass
class BatchResult:
    label: str
    model: str
    vad_silence_ms: int | None
    vad_start_sensitivity: str | None
    turn_coverage: str | None
    manual_vad: bool
    thinking_level: str | None
    runs: int
    valid: int
    first_activity_p50: float | None
    first_activity_p95: float | None
    last_activity_p50: float | None
    last_activity_p95: float | None
    passed: bool  # last_activity_p50 ≤ PASS_THRESHOLD_MS

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_api_key(env_path: Path | None = None) -> str:
    path = env_path or (_ROOT / ".env")
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if k.strip() == "GEMINI_API_KEY":
                return v.strip()
    return ""


def generate_speech_pcm(phrase: str) -> bytes:
    """Produce 16 kHz mono int16 raw PCM via macOS `say` + ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        aiff = Path(tmpdir) / "speech.aiff"
        pcm = Path(tmpdir) / "speech.pcm"
        subprocess.run(["say", "-o", str(aiff), phrase], check=True, capture_output=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(aiff),
                "-f",
                "s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                str(pcm),
            ],
            check=True,
            capture_output=True,
        )
        return pcm.read_bytes()


def chunk_pcm(pcm: bytes) -> list[bytes]:
    """Split raw PCM into fixed-size frames, zero-padding the last one."""
    frames = []
    for i in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[i : i + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk += b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)
    return frames


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


async def run_once(
    frames: list[bytes],
    model: str,
    run_index: int,
    vad_silence_ms: int | None = None,
    vad_start_sensitivity: str | None = None,
    vad_end_sensitivity: str | None = None,
    turn_coverage: str | None = None,
    manual_vad: bool = False,
    thinking_level: str | None = None,
    warmup_turns: int = 0,
) -> RunResult:
    """Run one measurement. Returns RunResult with None fields on timeout/error.

    With warmup_turns > 0, the session is primed with that many throwaway turns
    before the measured turn so steady-state (warm) latency is captured instead of
    the inflated cold first turn after connect.
    """
    audio_received: asyncio.Event = asyncio.Event()

    def _on_audio(_data: bytes) -> None:
        audio_received.set()

    session = GeminiLiveSession(
        model=model,
        api_key_env="GEMINI_API_KEY",
        audio_activity_rms_threshold=RMS_THRESHOLD,
        vad_silence_ms=vad_silence_ms,
        vad_start_sensitivity=vad_start_sensitivity,
        vad_end_sensitivity=vad_end_sensitivity,
        turn_coverage=turn_coverage,
        manual_vad=manual_vad,
        thinking_level=thinking_level,
    )
    session.on_audio_out(_on_audio)

    rms_flags = [compute_rms_int16(f) > RMS_THRESHOLD for f in frames]
    active = sum(rms_flags)
    # In manual-VAD mode, stop at the last above-threshold frame so activity_end
    # fires at true end-of-speech (mirrors a real client-side VAD); otherwise the
    # trailing quiet frames in the phrase inflate last_activity→response.
    send_frames = frames
    if manual_vad and active:
        last_active = max(i for i, f in enumerate(rms_flags) if f)
        send_frames = frames[: last_active + 1]

    async def send_turn() -> None:
        if manual_vad:
            await session.send_activity_start()
        for i, frame in enumerate(send_frames):
            await session.send_audio(frame)
            # Pace frames in real time, but skip the gap after the final frame so
            # activity_end fires at end-of-speech (a real client VAD wouldn't wait).
            if not (manual_vad and i == len(send_frames) - 1):
                await asyncio.sleep(0.1)
        if manual_vad:
            # Explicit end-of-turn: the server responds with no silence wait, and
            # we must NOT send audio (silence) after activity_end.
            await session.send_activity_end()
        else:
            # Trailing silence triggers automatic VAD end-of-turn.
            silence = b"\x00" * FRAME_BYTES
            for _ in range(SILENCE_FRAMES_AFTER_SPEECH):
                await session.send_audio(silence)
                await asyncio.sleep(0.1)

    try:
        await session.open()
        await asyncio.sleep(0.3)  # brief settle before sending audio

        if active == 0:
            logger.warning("[run %d] no frames above RMS threshold", run_index)

        # Warm-up turns prime the session so the measured turn reflects steady state.
        for w in range(warmup_turns):
            audio_received.clear()
            await send_turn()
            try:
                await asyncio.wait_for(audio_received.wait(), timeout=RESPONSE_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("[run %d] warm-up turn %d timed out", run_index, w + 1)
                return RunResult(None, None)
            # Let the model finish speaking, then re-zero anchors for the next turn.
            await asyncio.sleep(WARMUP_DRAIN_S)
            session.reset_timing()

        # Measured turn. reset_timing() cleared first_any_send, so the session's own
        # guard ignores any residual warm-up audio until our send begins here.
        audio_received.clear()
        await send_turn()
        try:
            await asyncio.wait_for(audio_received.wait(), timeout=RESPONSE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("[run %d] timed out after %.0fs", run_index, RESPONSE_TIMEOUT_S)
            return RunResult(None, None)

        first_ms: float | None = session.stats.get(  # type: ignore[assignment]
            "first_audio_out_after_first_audio_activity_send_ms"
        )
        last_ms: float | None = session.stats.get(  # type: ignore[assignment]
            "first_audio_out_after_last_audio_activity_ms"
        )

        if first_ms is None:
            logger.warning("[run %d] audio received but activity metric is None", run_index)
        else:
            logger.info(
                "[run %d] first_act→resp=%.0fms  last_act→resp=%s ms",
                run_index,
                first_ms,
                f"{last_ms:.0f}" if last_ms is not None else "n/a",
            )
        return RunResult(first_ms, last_ms)

    except Exception as exc:
        logger.error("[run %d] error: %s", run_index, exc)
        return RunResult(None, None)
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Batch measurement (importable by tune_vad.py)
# ---------------------------------------------------------------------------


async def measure_batch(
    frames: list[bytes],
    *,
    label: str,
    model: str,
    runs: int,
    vad_silence_ms: int | None = None,
    vad_start_sensitivity: str | None = None,
    vad_end_sensitivity: str | None = None,
    turn_coverage: str | None = None,
    manual_vad: bool = False,
    thinking_level: str | None = None,
    warmup_turns: int = 0,
    inter_run_sleep: float = INTER_RUN_SLEEP_S,
) -> BatchResult:
    first_samples: list[float] = []
    last_samples: list[float] = []

    for i in range(runs):
        r = await run_once(
            frames,
            model,
            i + 1,
            vad_silence_ms=vad_silence_ms,
            vad_start_sensitivity=vad_start_sensitivity,
            vad_end_sensitivity=vad_end_sensitivity,
            turn_coverage=turn_coverage,
            manual_vad=manual_vad,
            thinking_level=thinking_level,
            warmup_turns=warmup_turns,
        )
        if r.first_activity_ms is not None:
            first_samples.append(r.first_activity_ms)
        if r.last_activity_ms is not None:
            last_samples.append(r.last_activity_ms)
        if i < runs - 1:
            await asyncio.sleep(inter_run_sleep)

    lp50 = percentile(last_samples, 50)
    lp95 = percentile(last_samples, 95)
    return BatchResult(
        label=label,
        model=model,
        vad_silence_ms=vad_silence_ms,
        vad_start_sensitivity=vad_start_sensitivity,
        turn_coverage=turn_coverage,
        manual_vad=manual_vad,
        thinking_level=thinking_level,
        runs=runs,
        valid=len(first_samples),
        first_activity_p50=percentile(first_samples, 50),
        first_activity_p95=percentile(first_samples, 95),
        last_activity_p50=lp50,
        last_activity_p95=lp95,
        passed=lp50 is not None and lp50 <= PASS_THRESHOLD_MS,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_result(r: BatchResult) -> None:
    print("\n" + "=" * 56)
    print(f"RESULTS  [{r.label}]")
    print("=" * 56)
    print(f"Valid samples     : {r.valid}/{r.runs}")
    if r.last_activity_p50 is None:
        print("No valid samples.")
        return
    print(f"last_act→resp p50 : {r.last_activity_p50:.0f} ms  (PRIMARY — end-of-speech to audio)")
    print(f"last_act→resp p95 : {r.last_activity_p95:.0f} ms")
    print(f"first_act→resp p50: {r.first_activity_p50:.0f} ms  (includes phrase duration)")
    print(f"first_act→resp p95: {r.first_activity_p95:.0f} ms")
    verdict = (
        f"PASS ✓  {r.last_activity_p50:.0f} ms ≤ {PASS_THRESHOLD_MS:.0f} ms"
        if r.passed
        else f"FAIL ✗  {r.last_activity_p50:.0f} ms > {PASS_THRESHOLD_MS:.0f} ms"
    )
    print(f"\nWeek 3 acceptance : {verdict}")


async def _main(
    runs: int,
    model: str,
    vad_silence_ms: int | None,
    vad_start_sensitivity: str | None,
    vad_end_sensitivity: str | None,
    turn_coverage: str | None,
    manual_vad: bool,
    thinking_level: str | None,
    warmup_turns: int,
) -> int:
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
    print(f"  {len(frames)} frames ({len(pcm) / 2 / SAMPLE_RATE:.2f}s), max RMS={max_rms:.0f}")

    label_parts = ["manual_vad" if manual_vad else f"vad_silence={vad_silence_ms or 'default'}"]
    if vad_start_sensitivity:
        label_parts.append(f"start_sens={vad_start_sensitivity}")
    if vad_end_sensitivity:
        label_parts.append(f"end_sens={vad_end_sensitivity}")
    if turn_coverage:
        label_parts.append(f"coverage={turn_coverage}")
    if thinking_level:
        label_parts.append(f"thinking={thinking_level}")
    if warmup_turns:
        label_parts.append(f"warmup={warmup_turns}")
    label = "  ".join(label_parts)

    print(f"\nRunning {runs} measurements — {label}\n")

    result = await measure_batch(
        frames,
        label=label,
        model=model,
        runs=runs,
        vad_silence_ms=vad_silence_ms,
        vad_start_sensitivity=vad_start_sensitivity,
        vad_end_sensitivity=vad_end_sensitivity,
        turn_coverage=turn_coverage,
        manual_vad=manual_vad,
        thinking_level=thinking_level,
        warmup_turns=warmup_turns,
    )

    _print_result(result)

    results_path = Path(__file__).parent / "ttfb_results.json"
    results_path.write_text(json.dumps(result.as_dict(), indent=2))
    print(f"\nResults written to {results_path.relative_to(_ROOT)}")

    return 0 if result.passed else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Measure Gemini Live TTFB for Week 3 acceptance.")
    p.add_argument("--runs", type=int, default=10, help="Number of runs (default: 10)")
    p.add_argument("--model", default="models/gemini-3.1-flash-live-preview")
    p.add_argument(
        "--vad-silence-ms",
        type=int,
        default=None,
        help="Gemini VAD silence duration ms (default: Gemini baseline)",
    )
    p.add_argument("--vad-start-sensitivity", choices=("high", "low"), default=None)
    p.add_argument(
        "--vad-end-sensitivity",
        choices=("high", "low"),
        default=None,
        help="VAD end-of-speech sensitivity; 'high' shortens end-of-turn detection",
    )
    p.add_argument("--turn-coverage", choices=("all", "activity_only"), default=None)
    p.add_argument(
        "--manual-vad",
        action="store_true",
        help="Disable server VAD; signal end-of-turn explicitly (no silence wait)",
    )
    p.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Thinking budget; 'minimal' for lowest latency",
    )
    p.add_argument(
        "--warmup-turns",
        type=int,
        default=0,
        help="Throwaway turns to prime the session before the measured turn (warm latency)",
    )
    args = p.parse_args()
    return asyncio.run(
        _main(
            args.runs,
            args.model,
            args.vad_silence_ms,
            args.vad_start_sensitivity,
            args.vad_end_sensitivity,
            args.turn_coverage,
            args.manual_vad,
            args.thinking_level,
            args.warmup_turns,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
