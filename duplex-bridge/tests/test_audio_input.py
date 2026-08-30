from __future__ import annotations

import asyncio

import pytest
from duplex_bridge import audio_input
from duplex_bridge.audio_backends.base import CaptureFormat
from duplex_bridge.audio_input import MicCapture, MicCaptureConfig
from duplex_bridge.audio_metrics import compute_rms_int16


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
            onset_speech_ms=0,  # isolate end-of-turn; onset debounce tested separately
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
async def test_onset_debounce_rejects_transient_but_opens_on_sustained_speech() -> None:
    """A lone loud transient must not open a turn; sustained speech does, un-clipped."""
    speech = (5000).to_bytes(2, "little", signed=True) * 1600  # RMS well above threshold
    silence = b"\x00\x00" * 1600
    session = FakeManualSession()
    mic = MicCapture(
        MicCaptureConfig(
            manual_vad=True,
            activity_rms_threshold=300.0,
            onset_speech_ms=250,  # ~3 frames at 100 ms each
        )
    )
    mic._session = session  # type: ignore[assignment]

    # A single loud frame (e.g. a mouse click) then silence must NOT open a turn.
    assert await mic._forward_with_turn_detection(speech) is False
    assert await mic._forward_with_turn_detection(silence) is False
    assert session.events == []
    assert mic._in_turn is False

    # Three sustained speech frames clear the 250 ms onset bar → the turn opens and the
    # buffered onset frames are flushed, so none of the speech onset is dropped.
    for frame in (speech, speech, speech):
        await mic._forward_with_turn_detection(frame)

    assert [e[0] for e in session.events] == ["start", "audio", "audio", "audio"]
    assert mic._in_turn is True

    # End-of-turn still fires after the silence window once a turn is open.
    mic_silence_frames = int(mic.config.end_of_turn_silence_ms // mic._frame_ms) + 1
    for _ in range(mic_silence_frames):
        await mic._forward_with_turn_detection(silence)
    assert session.events[-1][0] == "end"
    assert mic._in_turn is False


def test_onset_gate_constants_are_module_level_and_drive_config_defaults() -> None:
    """The 2026-07-23 fix requires the onset debounce (sustained-voiced duration) AND the
    onset RMS gate (rejects ambient noise that clears the ~300 activity threshold but isn't
    genuine speech) to be tunable module constants, not literals buried on the dataclass
    field — so they can be tuned without touching MicCaptureConfig's definition."""
    assert hasattr(audio_input, "DEFAULT_ONSET_SPEECH_MS")
    assert hasattr(audio_input, "DEFAULT_ONSET_RMS_THRESHOLD")

    # Sustained-voiced duration bar: "roughly 150-250ms" per the incident writeup.
    assert 150 <= audio_input.DEFAULT_ONSET_SPEECH_MS <= 250

    # The onset gate must sit strictly above the reported ambient band (rms_p50 ~800-1200,
    # rms_max ~3280) and strictly below a genuine speech-level frame (~5000 RMS, the level
    # used across this module's existing "speech" fixtures) — otherwise it either still
    # opens on ambient or never opens on real speech.
    assert 1200 < audio_input.DEFAULT_ONSET_RMS_THRESHOLD < 5000

    config = MicCaptureConfig()
    assert config.onset_speech_ms == audio_input.DEFAULT_ONSET_SPEECH_MS
    assert config.onset_rms_threshold == audio_input.DEFAULT_ONSET_RMS_THRESHOLD


@pytest.mark.asyncio
async def test_onset_requires_consecutive_voiced_frames_not_cumulative_total() -> None:
    """Sub-threshold frames must reset the onset accumulator — total voiced ms across
    non-consecutive spikes is not enough, and a lone high-RMS spike frame never opens a
    turn on its own (both are the 'isolated transient' cases the debounce must reject)."""
    speech = (5000).to_bytes(2, "little", signed=True) * 1600  # RMS well above any threshold
    silence = b"\x00\x00" * 1600
    session = FakeManualSession()
    mic = MicCapture(
        MicCaptureConfig(manual_vad=True, activity_rms_threshold=300.0, onset_speech_ms=250)
    )
    mic._session = session  # type: ignore[assignment]

    # 3 alternating voiced/silence pairs: 300 ms of voiced frames in total (which alone
    # would clear the 250 ms bar) but never 3 CONSECUTIVE voiced frames — each pair is
    # broken by an intervening silence frame, so the onset accumulator must reset each time.
    for _ in range(3):
        await mic._forward_with_turn_detection(speech)
        await mic._forward_with_turn_detection(silence)

    assert session.events == []
    assert mic._in_turn is False

    # A single very-high-RMS frame (matching the incident's reported rms_max ~3280) must
    # also never open a turn by itself — one frame (100 ms) can't clear a 250 ms bar
    # regardless of amplitude, so this also locks the "isolated spike" rejection.
    loud_spike = (3280).to_bytes(2, "little", signed=True) * 1600
    assert compute_rms_int16(loud_spike) == pytest.approx(3280.0, abs=1.0)
    await mic._forward_with_turn_detection(loud_spike)
    await mic._forward_with_turn_detection(silence)

    assert session.events == []
    assert mic._in_turn is False


@pytest.mark.asyncio
async def test_default_config_rejects_ambient_band_but_opens_on_sustained_speech() -> None:
    """2026-07-23 live-run bug: sustained ambient RMS ~800-1200 (well above the 300 activity
    gate, well below genuine speech) opened a phantom turn and cut the model off 'into
    nothing'. With the DEFAULT config (no explicit RMS overrides), sustained ambient-band
    audio must never open a turn, while sustained genuine speech-level audio still opens one
    cleanly after exactly the onset window, with every onset-buffered frame flushed (no
    clipped speech) and no added latency (it opens as soon as the bar is cleared, not later).
    """
    ambient_frame = (900).to_bytes(2, "little", signed=True) * 1600
    assert 800 <= compute_rms_int16(ambient_frame) <= 1200  # really in the reported ambient band

    speech_frame = (5000).to_bytes(2, "little", signed=True) * 1600
    session = FakeManualSession()
    mic = MicCapture(MicCaptureConfig(manual_vad=True))  # all defaults — the live config
    mic._session = session  # type: ignore[assignment]

    frames_to_clear_bar = int(mic.config.onset_speech_ms // mic._frame_ms) + 1

    # Sustained ambient noise, well past the onset window, must never open a turn.
    for _ in range(frames_to_clear_bar + 2):
        await mic._forward_with_turn_detection(ambient_frame)
    assert session.events == []
    assert mic._in_turn is False

    # Sustained genuine speech clears the identical bar and opens the turn immediately once
    # cleared — no extra frames of added latency beyond the onset window itself.
    for _ in range(frames_to_clear_bar):
        await mic._forward_with_turn_detection(speech_frame)
    assert mic._in_turn is True
    kinds = [e[0] for e in session.events]
    assert kinds[0] == "start"
    # every onset-window frame reached the session — the start of speech was not clipped.
    assert kinds.count("audio") == frames_to_clear_bar


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
