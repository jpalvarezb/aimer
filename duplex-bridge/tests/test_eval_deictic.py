"""Tests for scripts/bench/eval_deictic.py's transcript-capture helper.

Week-9 fix: the deictic eval's response capture used a fixed TEXT_SETTLE_S trailing-silence
window (plus a RESPONSE_TIMEOUT_S hard deadline) to decide when the model's turn was "done".
Longer transcripts with a mid-response pause longer than the settle window got truncated,
driving spurious "ambiguous" judge verdicts (docs/week9-delegate-acceptance.md:121-124).

These tests exercise the new capture helper (``eval_deictic._collect_transcript``) against a
fake session that mimics GeminiLiveSession's ``on_text_out`` / ``on_turn_complete`` surface —
no real Gemini connection. They prove: (1) turn-complete-driven capture is immune to the old
settle-window truncation, (2) it returns promptly on completion instead of waiting out a fixed
window, and (3) the hard timeout still backstops a stream that never signals completion.

Follows test_fd.py's / test_delegate_eval.py's sys.path-into-scripts/bench import convention.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))

import eval_deictic  # noqa: E402


class _FakeSession:
    """Duck-types the slice of GeminiLiveSession that eval_deictic's capture helper needs:
    ``on_text_out`` and ``on_turn_complete`` registration. ``run()`` replays scripted text
    chunks (with delays, to simulate real streaming pacing) and then optionally fires
    turn_complete — or never does, to simulate a hung stream.
    """

    def __init__(
        self,
        chunks_with_delays: list[tuple[float, str]],
        *,
        complete_at_end: bool,
    ) -> None:
        self._chunks_with_delays = chunks_with_delays
        self._complete_at_end = complete_at_end
        self._text_callbacks: list = []
        self._turn_complete_callbacks: list = []

    def on_text_out(self, callback) -> None:
        self._text_callbacks.append(callback)

    def on_turn_complete(self, callback) -> None:
        self._turn_complete_callbacks.append(callback)

    async def run(self) -> None:
        for delay, text in self._chunks_with_delays:
            await asyncio.sleep(delay)
            for cb in self._text_callbacks:
                cb(text)
        if self._complete_at_end:
            for cb in self._turn_complete_callbacks:
                cb()


# The production settle window is TEXT_SETTLE_S = 1.5s; these tests scale it down ~7.5x
# (to 0.2s) purely for test speed. Gaps below are deliberately longer than that scaled
# window so the old settle-break logic would have truncated the transcript.
_SCALED_OLD_SETTLE_S = 0.2


async def test_capture_full_transcript_despite_gaps_longer_than_settle_window():
    """Chunks separated by gaps longer than the (scaled) old settle window must all still
    land in the final transcript — nothing dropped at a settle boundary."""
    text_chunks: list[str] = []
    session = _FakeSession(
        [
            (0.01, "Hello "),
            (_SCALED_OLD_SETTLE_S * 2, "world"),  # gap > old settle window
            (_SCALED_OLD_SETTLE_S * 2, "!"),  # gap > old settle window again
        ],
        complete_at_end=True,
    )
    session.on_text_out(text_chunks.append)

    runner = asyncio.create_task(session.run())
    result = await eval_deictic._collect_transcript(session, text_chunks, timeout_s=5.0)
    await runner

    assert result == "Hello world!", (
        f"expected the full transcript, got a truncated capture: {result!r}"
    )


async def test_capture_returns_promptly_on_turn_complete():
    """When turn_complete arrives immediately after the last chunk, capture must return
    promptly — not wait out a fixed settle period."""
    text_chunks: list[str] = []
    session = _FakeSession([(0.01, "Done.")], complete_at_end=True)
    session.on_text_out(text_chunks.append)

    runner = asyncio.create_task(session.run())
    start = time.perf_counter()
    result = await eval_deictic._collect_transcript(session, text_chunks, timeout_s=5.0)
    elapsed = time.perf_counter() - start
    await runner

    assert result == "Done."
    assert elapsed < _SCALED_OLD_SETTLE_S, (
        f"capture took {elapsed:.3f}s — should return promptly on turn_complete, "
        f"well under the {_SCALED_OLD_SETTLE_S}s settle window and the 5.0s timeout"
    )


async def test_capture_hits_hard_timeout_backstop_on_hung_stream():
    """A stream that emits chunks but never signals turn_complete must still return —
    the hard timeout is the safety backstop, with whatever partial text was collected."""
    text_chunks: list[str] = []
    timeout_s = 0.3
    session = _FakeSession([(0.01, "Partial")], complete_at_end=False)
    session.on_text_out(text_chunks.append)

    runner = asyncio.create_task(session.run())
    start = time.perf_counter()
    result = await eval_deictic._collect_transcript(session, text_chunks, timeout_s=timeout_s)
    elapsed = time.perf_counter() - start
    runner.cancel()

    assert result == "Partial"
    assert elapsed >= timeout_s, (
        f"capture returned in {elapsed:.3f}s, before the {timeout_s}s hard timeout backstop"
    )
    assert elapsed < timeout_s + 1.0, "hard timeout backstop did not fire promptly"
