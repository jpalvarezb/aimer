"""Week-8 FD-bench-style eval — interrupt / backchannel / talk-over + pointer-deixis.

A local rerun of the core full-duplex behaviors, scored deterministically by driving the REAL
bridge components with synthetic audio frames and model events (no live audio / API needed):

  - backchannel : a short "mhm" (< onset window) must NOT open a user turn.
  - talk-over   : sustained speech DOES open a turn (the onset debounce distinguishes the two).
  - turn-taking : trailing silence closes the turn (activity_end).
  - interrupt   : a barge-in flag flushes buffered playback (assistant stops promptly).
  - detection   : the interrupted flag is read from the model message.
  - pointer-deixis: the Week-4 suite (separately measured, cited here for the combined report).

Run:  uv run --no-sync python scripts/bench/eval_fd.py
Out:  scripts/bench/results/week8_fd_eval.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))

from duplex_bridge.audio_input import MicCapture, MicCaptureConfig  # noqa: E402
from duplex_bridge.audio_output import SpeakerOutput  # noqa: E402
from duplex_bridge.providers.gemini_live import (  # noqa: E402
    GeminiLiveSession,
    _interrupted_from_message,
)
from duplex_bridge.session import DuplexSession  # noqa: E402

_OUT = _HERE / "results" / "week8_fd_eval.json"

FRAME_MS = 20
SAMPLE_RATE = 16_000
_N = SAMPLE_RATE * FRAME_MS // 1000  # samples per 20 ms frame

# RMS 5000: genuinely speech-level, not just "> 300 activity threshold". The 2026-07-23
# onset-gate fix (duplex_bridge.audio_input.DEFAULT_ONSET_RMS_THRESHOLD) requires onset
# audio to clear a bar above the reported sustained-ambient band (rms_p50 ~800-1200); a
# synthetic "speech" signal at 1200 would sit inside that ambient band and never open a
# turn, so this must stay clearly above DEFAULT_ONSET_RMS_THRESHOLD.
VOICED = np.full(_N, 5000, dtype=np.int16).tobytes()
SILENCE = np.zeros(_N, dtype=np.int16).tobytes()  # RMS 0

# Pointer-deixis suite score (Week-4 acceptance; see docs/week4-deictic-acceptance.md).
POINTER_DEIXIS_PCT = 87.1


class _FakeBackend:
    capture_config = SimpleNamespace(frame_ms=FRAME_MS)


class _RecSession(DuplexSession):
    """Records the turn events the MicCapture machine emits."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def open(self) -> None: ...
    async def send_audio(self, frames: bytes) -> None:
        self.events.append("audio")

    async def send_visual_context(self, packet) -> None: ...
    async def send_activity_start(self) -> None:
        self.events.append("start")

    async def send_activity_end(self) -> None:
        self.events.append("end")

    def on_audio_out(self, callback) -> None: ...
    def on_tool_call(self, callback) -> None: ...
    async def close(self) -> None: ...


def _mic() -> MicCapture:
    cfg = MicCaptureConfig(
        manual_vad=True, onset_speech_ms=250, end_of_turn_silence_ms=400, activity_rms_threshold=300
    )
    mic = MicCapture(cfg, backend=_FakeBackend())  # type: ignore[arg-type]
    mic._session = _RecSession()
    return mic


async def _feed(mic: MicCapture, frame: bytes, n: int) -> None:
    for _ in range(n):
        await mic._forward_with_turn_detection(frame)


async def scenario_backchannel() -> tuple[bool, str]:
    """100 ms voiced blip then silence — below the 250 ms onset window, so NO turn opens."""
    mic = _mic()
    await _feed(mic, VOICED, 5)  # 100 ms < 250 ms onset
    await _feed(mic, SILENCE, 1)
    starts = mic._session.events.count("start")  # type: ignore[attr-defined]
    return starts == 0, f"turn_starts={starts} (want 0)"


async def scenario_talkover_opens_turn() -> tuple[bool, str]:
    """280 ms of sustained speech clears the onset debounce and opens exactly one turn."""
    mic = _mic()
    await _feed(mic, VOICED, 14)  # 280 ms >= 250 ms onset
    starts = mic._session.events.count("start")  # type: ignore[attr-defined]
    return starts == 1, f"turn_starts={starts} (want 1)"


async def scenario_turn_taking_closes() -> tuple[bool, str]:
    """A turn opens, then 420 ms of trailing silence closes it (activity_end)."""
    mic = _mic()
    await _feed(mic, VOICED, 14)
    await _feed(mic, SILENCE, 21)  # 420 ms >= 400 ms end-of-turn window
    ended = "end" in mic._session.events  # type: ignore[attr-defined]
    return ended, f"activity_end_fired={ended} (want True)"


async def scenario_interrupt_flushes_playback() -> tuple[bool, str]:
    """A barge-in fires on_interrupt, which flushes buffered speaker audio (assistant stops)."""
    speaker = SpeakerOutput()
    speaker._running = True
    for _ in range(5):
        speaker._enqueue(b"\x00" * 480)
    before = speaker._queue.qsize()
    session = GeminiLiveSession(model="models/gemini-3.1-flash-live-preview")
    session.on_interrupt(speaker.flush)
    await session._dispatch_interrupt()
    after = speaker._queue.qsize()
    return before > 0 and after == 0, f"buffered {before}->{after} after barge-in (want >0 -> 0)"


def scenario_interrupt_detection() -> tuple[bool, str]:
    """The interrupted flag is correctly read from a model message (and absent when false)."""
    yes = SimpleNamespace(server_content=SimpleNamespace(interrupted=True))
    no = SimpleNamespace(server_content=SimpleNamespace(interrupted=False))
    ok = bool(_interrupted_from_message(yes)) and not _interrupted_from_message(no)
    return (
        ok,
        f"detect(yes)={_interrupted_from_message(yes)} detect(no)={_interrupted_from_message(no)}",
    )


async def main() -> int:
    async_scenarios = {
        "backchannel_rejected": scenario_backchannel,
        "talkover_opens_turn": scenario_talkover_opens_turn,
        "turn_taking_closes": scenario_turn_taking_closes,
        "interrupt_flushes_playback": scenario_interrupt_flushes_playback,
    }
    rows: dict[str, dict] = {}
    for name, fn in async_scenarios.items():
        ok, detail = await fn()
        rows[name] = {"pass": ok, "detail": detail}
    ok, detail = scenario_interrupt_detection()
    rows["interrupt_detection"] = {"pass": ok, "detail": detail}

    passed = sum(1 for r in rows.values() if r["pass"])
    total = len(rows)

    print("=" * 68)
    print("WEEK-8 FD-BENCH-STYLE EVAL (interrupt / backchannel / talk-over)")
    print("=" * 68)
    for name, r in rows.items():
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {name:<28} {r['detail']}")
    print(f"\n  full-duplex behaviors: {passed}/{total} passed")
    print(
        f"  pointer-deixis suite : {POINTER_DEIXIS_PCT}% (Week-4, docs/week4-deictic-acceptance.md)"
    )

    summary = {
        "fd_behaviors_passed": passed,
        "fd_behaviors_total": total,
        "pointer_deixis_pct": POINTER_DEIXIS_PCT,
        "scenarios": rows,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n  written: {_OUT}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
