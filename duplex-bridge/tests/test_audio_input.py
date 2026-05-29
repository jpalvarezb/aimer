from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from duplex_bridge.audio_input import MicCapture, MicCaptureConfig


class FakeSession:
    def __init__(self) -> None:
        self.audio: list[bytes] = []

    async def send_audio(self, frames: bytes) -> None:
        self.audio.append(frames)


class FakeInputStream:
    instances: list[FakeInputStream] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class FakeAudio:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


@pytest.mark.asyncio
async def test_audio_input_sends_pcm_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeInputStream.instances.clear()
    fake_sounddevice = SimpleNamespace(RawInputStream=FakeInputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    session = FakeSession()
    mic = MicCapture(MicCaptureConfig(blocksize=2, queue_maxsize=2))

    started = await mic.start(session)
    assert started

    stream = FakeInputStream.instances[0]
    stream.kwargs["callback"](FakeAudio(b"\x01\x00\x02\x00"), 2, None, None)
    await asyncio.sleep(0.05)

    assert session.audio == [b"\x01\x00\x02\x00"]
    assert mic.stats["sent_frames"] == 1

    await mic.stop()
    assert stream.closed


def test_audio_input_drops_when_queue_full() -> None:
    mic = MicCapture(MicCaptureConfig(queue_maxsize=1))
    mic._running = True

    mic._try_put_audio(b"old")
    mic._try_put_audio(b"new")

    assert mic.stats["dropped_frames"] == 1
    assert mic._queue.get_nowait() == b"old"


@pytest.mark.asyncio
async def test_audio_input_gracefully_handles_missing_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duplex_bridge.audio_input as audio_input

    def fail_import(name: str):
        if name == "sounddevice":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(audio_input.importlib, "import_module", fail_import)

    mic = MicCapture()

    assert not await mic.start(FakeSession())


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


@pytest.mark.asyncio
async def test_audio_input_logs_health(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    FakeInputStream.instances.clear()
    fake_sounddevice = SimpleNamespace(RawInputStream=FakeInputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    caplog.set_level("INFO", logger="duplex_bridge.audio_input")

    session = FakeSession()
    mic = MicCapture(MicCaptureConfig(health_log_interval_s=0.0))

    started = await mic.start(session)
    assert started

    stream = FakeInputStream.instances[0]
    stream.kwargs["callback"](FakeAudio(b"\x01\x00\x02\x00"), 2, None, None)
    await asyncio.sleep(0.05)

    assert "chunks_per_sec" in caplog.text
    assert "bytes_per_sec" in caplog.text
    assert "rms_min" in caplog.text
    assert mic.stats["sent_bytes"] == 4

    await mic.stop()
