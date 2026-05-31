"""Microphone capture pipeline for same-machine duplex sessions.

``MicCapture`` is backend-agnostic: it consumes cleaned 16 kHz mono int16 frames
from an ``AudioBackend`` and runs the turn-detection / VAD / push-to-talk state
machine on top, forwarding to a ``DuplexSession``. Platform device I/O lives in
the backend (see ``duplex_bridge.audio_backends``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field

from duplex_bridge.audio_backends.base import AudioBackend
from duplex_bridge.audio_metrics import compute_rms_int16, percentile
from duplex_bridge.session import DuplexSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicCaptureConfig:
    """Runtime configuration for microphone capture."""

    sample_rate: int = 16_000
    channels: int = 1
    blocksize: int = 1_600
    queue_maxsize: int = 8
    dtype: str = "int16"
    device: str | int | None = None
    health_log_interval_s: float = 10.0
    # Client-side end-of-turn detection. When enabled, the capture loop drives the
    # session's manual VAD: it sends activity_start on speech onset and activity_end
    # after end_of_turn_silence_ms of sub-threshold audio, instead of relying on the
    # server's (slower, ~630 ms effective) silence detection. Audio between turns is
    # not forwarded — the Live API rejects audio sent after activity_end.
    manual_vad: bool = False
    activity_rms_threshold: float = 300.0
    end_of_turn_silence_ms: int = 400
    # Push-to-talk: drive turns from an external key signal (set_talking) instead of
    # RMS. End-of-turn is immediate on release (no silence window), giving the true
    # model+network latency floor. Implies manual VAD.
    push_to_talk: bool = False


@dataclass
class _Stats:
    dropped_frames: int = 0
    sent_frames: int = 0
    sent_bytes: int = 0


@dataclass
class _InputHealthWindow:
    log_interval_s: float
    last_log_at: float = field(default_factory=time.perf_counter)
    chunks_sent: int = 0
    bytes_sent: int = 0
    queue_depths: list[float] = field(default_factory=list)
    rms_values: list[float] = field(default_factory=list)

    def record(self, frames: bytes, queue_depth: int) -> None:
        self.chunks_sent += 1
        self.bytes_sent += len(frames)
        self.queue_depths.append(float(queue_depth))
        self.rms_values.append(compute_rms_int16(frames))

    def maybe_log(self, *, dropped_frames: int) -> None:
        if self.chunks_sent == 0:
            return
        now = time.perf_counter()
        elapsed = now - self.last_log_at
        if self.log_interval_s > 0 and elapsed < self.log_interval_s:
            return

        elapsed = max(elapsed, 1e-9)
        queue_depth_avg = sum(self.queue_depths) / len(self.queue_depths)
        logger.info(
            "[audio-in] health chunks_per_sec=%.1f bytes_per_sec=%.0f "
            "dropped_frames=%d queue_depth_avg=%.1f queue_depth_p95=%.1f "
            "rms_min=%.1f rms_p50=%.1f rms_max=%.1f",
            self.chunks_sent / elapsed,
            self.bytes_sent / elapsed,
            dropped_frames,
            queue_depth_avg,
            percentile(self.queue_depths, 95) or 0.0,
            min(self.rms_values),
            percentile(self.rms_values, 50) or 0.0,
            max(self.rms_values),
        )
        self.last_log_at = now
        self.chunks_sent = 0
        self.bytes_sent = 0
        self.queue_depths.clear()
        self.rms_values.clear()


class MicCapture:
    """Run the turn-detection pipeline over frames from an ``AudioBackend``."""

    def __init__(
        self,
        config: MicCaptureConfig | None = None,
        backend: AudioBackend | None = None,
    ) -> None:
        self.config = config or MicCaptureConfig()
        if backend is None:
            from duplex_bridge.audio_backends.sounddevice_backend import SoundDeviceBackend

            backend = SoundDeviceBackend(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.blocksize,
                dtype=self.config.dtype,
                device=self.config.device,
            )
        self._backend = backend
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=self.config.queue_maxsize)
        self._drain_task: asyncio.Task[None] | None = None
        self._session: DuplexSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._stats = _Stats()
        self._health_window = _InputHealthWindow(self.config.health_log_interval_s)
        # Client-side end-of-turn detection state (manual_vad only).
        self._in_turn = False
        self._silence_ms = 0.0
        self._frame_ms = self._backend.capture_config.frame_ms
        # Push-to-talk: whether the talk key is currently held.
        self._ptt_active = False

    def set_talking(self, active: bool) -> None:
        """Set push-to-talk state. Call via loop.call_soon_threadsafe from a key thread."""
        self._ptt_active = active

    @property
    def stats(self) -> dict[str, int]:
        return {
            "dropped_frames": self._stats.dropped_frames,
            "sent_frames": self._stats.sent_frames,
            "sent_bytes": self._stats.sent_bytes,
        }

    async def start(
        self,
        session: DuplexSession,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> bool:
        """Start the backend and the drain pipeline.

        Returns False if the backend (audio devices / dependencies) is unavailable.
        """
        if self._running:
            return True

        self._session = session
        self._loop = loop or asyncio.get_running_loop()
        self._running = True
        self._drain_task = asyncio.create_task(self._drain_loop())

        if not await self._backend.start(self._try_put_audio, session, self._loop):
            await self.stop()
            return False
        return True

    async def stop(self) -> None:
        """Stop the backend and cancel the drain worker."""
        self._running = False

        await self._backend.stop()

        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None

        self._session = None
        self._loop = None

    def _try_put_audio(self, frames: bytes) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(frames)
        except asyncio.QueueFull:
            # Input overflow drops newest to avoid disturbing already-buffered audio.
            self._stats.dropped_frames += 1

    async def _drain_loop(self) -> None:
        while self._running:
            frames = await self._queue.get()
            if self._session is None:
                continue
            try:
                uses_turn_detection = self.config.manual_vad or self.config.push_to_talk
                if uses_turn_detection and not await self._forward_with_turn_detection(frames):
                    continue
                elif not uses_turn_detection:
                    await self._session.send_audio(frames)
                self._stats.sent_frames += 1
                self._stats.sent_bytes += len(frames)
                self._health_window.record(frames, self._queue.qsize())
                self._health_window.maybe_log(dropped_frames=self._stats.dropped_frames)
            except Exception as exc:
                logger.warning("[audio-in] failed to send audio frame: %s", exc)

    async def _forward_with_turn_detection(self, frames: bytes) -> bool:
        """Drive manual VAD for one frame; return True if the frame was forwarded.

        Sends activity_start on turn onset and activity_end at end-of-turn. Inter-turn
        audio is not forwarded. End-of-turn is detected from the push-to-talk key (on
        release, immediately) or from a trailing silence window (RMS-based).
        """
        assert self._session is not None
        if self.config.push_to_talk:
            is_speech = self._ptt_active
        else:
            is_speech = compute_rms_int16(frames) > self.config.activity_rms_threshold

        if is_speech and not self._in_turn:
            await self._session.send_activity_start()
            self._in_turn = True
            self._silence_ms = 0.0

        if not self._in_turn:
            return False  # between turns: do not forward

        # Push-to-talk: end the turn the instant the key is released (no silence wait).
        if self.config.push_to_talk:
            if not is_speech:
                await self._session.send_activity_end()
                self._in_turn = False
                return False
            await self._session.send_audio(frames)
            return True

        await self._session.send_audio(frames)

        if is_speech:
            self._silence_ms = 0.0
        else:
            self._silence_ms += self._frame_ms
            if self._silence_ms >= self.config.end_of_turn_silence_ms:
                await self._session.send_activity_end()
                self._in_turn = False
        return True
