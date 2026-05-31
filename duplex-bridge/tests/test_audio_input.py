from __future__ import annotations

import asyncio

import pytest
from duplex_bridge.audio_backends.base import CaptureFormat
from duplex_bridge.audio_input import MicCapture, MicCaptureConfig


class FakeSession:
    def __init__(self) -> None:
        self.audio: list[bytes] = []

    async def send_audio(self, frames: bytes) -> None:
        self.audio.append(frames)


class FakeManualSession:
    """Records the manual-VAD turn markers and audio frames in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, bytes | None]] = []

    async def send_activity_start(self) -> None:
        self.events.append(("start", None))

    async def send_audio(self, frames: bytes) -> None:
        self.events.append(("audio", frames))

    async def send_activity_end(self) -> None:
        self.events.append(("end", None))


class FakeBackend:
    """Backend stub that lets the test push frames into the pipeline."""

    def __init__(self, *, frame_samples: int = 1_600, start_ok: bool = True) -> None:
        self._fmt = CaptureFormat(sample_rate=16_000, channels=1, frame_samples=frame_samples)
        self._start_ok = start_ok
        self.on_frame = None
        self.started = False
        self.stopped = False

    @property
    def capture_config(self) -> CaptureFormat:
        return self._fmt

    @property
    def stats(self) -> dict[str, int]:
        return {}

    async def start(self, on_frame, session, loop) -> bool:  # noqa: ANN001
        self.on_frame = on_frame
        self.started = self._start_ok
        return self._start_ok

    async def stop(self) -> None:
        self.stopped = True

    def push(self, frames: bytes) -> None:
        assert self.on_frame is not None
        self.on_frame(frames)


# --- Pipeline turn-detection logic (device-independent) ---------------------


@pytest.mark.asyncio
async def test_manual_vad_drives_activity_markers() -> None:
    """Client-side end-of-turn: start on speech, end after the silence window, gate audio."""
    speech = (5000).to_bytes(2, "little", signed=True) * 1600  # RMS well above threshold
    silence = b"\x00\x00" * 1600
    session = FakeManualSession()
    mic = MicCapture(
        MicCaptureConfig(
            manual_vad=True,
            activity_rms_threshold=300.0,
            end_of_turn_silence_ms=200,  # 2 silence frames at 100 ms each
        )
    )
    mic._session = session  # type: ignore[assignment]

    # Turn 1: two speech frames, then two silence frames trigger end-of-turn.
    for frame in (speech, speech, silence, silence):
        await mic._forward_with_turn_detection(frame)

    kinds = [e[0] for e in session.events]
    assert kinds == ["start", "audio", "audio", "audio", "audio", "end"]
    assert mic._in_turn is False

    # Inter-turn silence is not forwarded.
    forwarded = await mic._forward_with_turn_detection(silence)
    assert forwarded is False
    assert [e[0] for e in session.events] == kinds  # unchanged

    # Speech resumes → a new turn starts.
    await mic._forward_with_turn_detection(speech)
    assert session.events[-2][0] == "start"
    assert session.events[-1][0] == "audio"
    assert mic._in_turn is True


@pytest.mark.asyncio
async def test_push_to_talk_gates_turns_on_key_state() -> None:
    """Push-to-talk: start on key down, forward only while held, end immediately on release."""
    speech = (5000).to_bytes(2, "little", signed=True) * 1600
    session = FakeManualSession()
    mic = MicCapture(MicCaptureConfig(push_to_talk=True))
    mic._session = session  # type: ignore[assignment]

    # Key not held → nothing forwarded.
    assert await mic._forward_with_turn_detection(speech) is False
    assert session.events == []

    # Key down → start, then audio while held (RMS is irrelevant in PTT).
    mic.set_talking(True)
    silence = b"\x00\x00" * 1600
    await mic._forward_with_turn_detection(speech)
    await mic._forward_with_turn_detection(silence)  # silence still forwarded while held
    assert [e[0] for e in session.events] == ["start", "audio", "audio"]

    # Key up → end immediately on the next frame, no silence window.
    mic.set_talking(False)
    forwarded = await mic._forward_with_turn_detection(speech)
    assert forwarded is False
    assert [e[0] for e in session.events] == ["start", "audio", "audio", "end"]
    assert mic._in_turn is False


def test_audio_input_drops_when_queue_full() -> None:
    mic = MicCapture(MicCaptureConfig(queue_maxsize=1))
    mic._running = True

    mic._try_put_audio(b"old")
    mic._try_put_audio(b"new")

    assert mic.stats["dropped_frames"] == 1
    assert mic._queue.get_nowait() == b"old"


# --- Pipeline over a backend (device-independent) ---------------------------


@pytest.mark.asyncio
async def test_pipeline_forwards_backend_frames_to_session() -> None:
    backend = FakeBackend()
    session = FakeSession()
    mic = MicCapture(MicCaptureConfig(), backend=backend)

    assert await mic.start(session)
    assert backend.started

    backend.push(b"\x01\x00\x02\x00")
    await asyncio.sleep(0.05)

    assert session.audio == [b"\x01\x00\x02\x00"]
    assert mic.stats["sent_frames"] == 1

    await mic.stop()
    assert backend.stopped


@pytest.mark.asyncio
async def test_pipeline_start_returns_false_when_backend_unavailable() -> None:
    backend = FakeBackend(start_ok=False)
    mic = MicCapture(MicCaptureConfig(), backend=backend)

    assert await mic.start(FakeSession()) is False


@pytest.mark.asyncio
async def test_pipeline_logs_health(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="duplex_bridge.audio_input")
    backend = FakeBackend()
    mic = MicCapture(MicCaptureConfig(health_log_interval_s=0.0), backend=backend)

    assert await mic.start(FakeSession())
    backend.push(b"\x01\x00\x02\x00")
    await asyncio.sleep(0.05)

    assert "chunks_per_sec" in caplog.text
    assert "rms_min" in caplog.text
    assert mic.stats["sent_bytes"] == 4

    await mic.stop()
