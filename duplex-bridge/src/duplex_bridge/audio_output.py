"""Speaker playback for same-machine duplex sessions."""

from __future__ import annotations

import contextlib
import importlib
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Any

from duplex_bridge.session import DuplexSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerOutputConfig:
    """Runtime configuration for speaker output."""

    sample_rate: int = 24_000
    channels: int = 1
    blocksize: int = 2_400
    queue_maxsize: int = 16
    dtype: str = "int16"
    device: str | int | None = None
    health_log_interval_s: float = 10.0


@dataclass
class _Stats:
    dropped_frames: int = 0
    played_frames: int = 0
    underruns: int = 0


@dataclass
class _OutputHealthWindow:
    log_interval_s: float
    last_log_at: float = field(default_factory=time.perf_counter)
    last_dropped_frames: int = 0
    played_frames: int = 0
    underruns: int = 0

    def record(self, *, underrun: bool) -> None:
        self.played_frames += 1
        if underrun:
            self.underruns += 1

    def maybe_log(self, *, dropped_frames: int) -> None:
        if self.played_frames == 0:
            return
        now = time.perf_counter()
        elapsed = now - self.last_log_at
        if self.log_interval_s > 0 and elapsed < self.log_interval_s:
            return

        elapsed = max(elapsed, 1e-9)
        dropped_delta = dropped_frames - self.last_dropped_frames
        logger.info(
            "[audio-out] health played_per_sec=%.1f underruns=%d dropped_frames=%d",
            self.played_frames / elapsed,
            self.underruns,
            dropped_delta,
        )
        self.last_log_at = now
        self.last_dropped_frames = dropped_frames
        self.played_frames = 0
        self.underruns = 0


class SpeakerOutput:
    """Play model audio without blocking the PortAudio callback."""

    def __init__(self, config: SpeakerOutputConfig | None = None) -> None:
        self.config = config or SpeakerOutputConfig()
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=self.config.queue_maxsize)
        self._stream: Any = None
        self._running = False
        self._pending = bytearray()
        self._stats = _Stats()
        self._health_window = _OutputHealthWindow(self.config.health_log_interval_s)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "dropped_frames": self._stats.dropped_frames,
            "played_frames": self._stats.played_frames,
            "underruns": self._stats.underruns,
        }

    def start(self, session: DuplexSession | None = None) -> bool:
        """Start speaker playback.

        When ``session`` is given, subscribes ``_enqueue`` to its audio output. Pass
        ``session=None`` to drive playback externally via :meth:`play` — used by the
        software-AEC backend, which intercepts model audio to also build the echo
        reference before playing it.
        """
        if self._running:
            return True

        try:
            sounddevice = importlib.import_module("sounddevice")
        except ImportError:
            logger.warning("[audio-out] sounddevice unavailable; install duplex-bridge[audio]")
            return False

        if session is not None:
            session.on_audio_out(self._enqueue)
        self._running = True
        try:
            self._stream = sounddevice.RawOutputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                device=self.config.device,
                callback=self._output_callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.warning("[audio-out] speaker output unavailable: %s", exc)
            self.stop()
            return False

        logger.info("[audio-out] speaker playback started")
        return True

    def stop(self) -> None:
        """Stop speaker playback."""
        self._running = False

        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None

        self._pending.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def play(self, audio: bytes) -> None:
        """Enqueue model audio for playback (for callers driving output externally)."""
        self._enqueue(audio)

    def _enqueue(self, audio: bytes) -> None:
        if not self._running:
            return

        try:
            self._queue.put_nowait(audio)
            return
        except queue.Full:
            pass

        # Output overflow drops oldest so playback catches up to fresher model audio.
        with contextlib.suppress(queue.Empty):
            self._queue.get_nowait()
            self._stats.dropped_frames += 1
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(audio)

    def _output_callback(
        self,
        outdata: Any,
        frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.debug("[audio-out] output status: %s", status)

        needed = frames * self.config.channels * _bytes_per_sample(self.config.dtype)
        data = self._read_audio_bytes(needed)
        underrun = len(data) < needed
        if underrun:
            self._stats.underruns += 1
            data += b"\x00" * (needed - len(data))

        _write_output_bytes(outdata, data)
        self._stats.played_frames += 1
        self._health_window.record(underrun=underrun)
        self._health_window.maybe_log(dropped_frames=self._stats.dropped_frames)

    def _read_audio_bytes(self, needed: int) -> bytes:
        data = bytearray()
        while len(data) < needed:
            if self._pending:
                take = min(needed - len(data), len(self._pending))
                data += self._pending[:take]
                del self._pending[:take]
                continue

            try:
                self._pending += self._queue.get_nowait()
            except queue.Empty:
                break

        return bytes(data)


def _bytes_per_sample(dtype: str) -> int:
    if dtype == "int16":
        return 2
    raise ValueError(f"unsupported speaker dtype: {dtype}")


def _write_output_bytes(outdata: Any, data: bytes) -> None:
    try:
        view = memoryview(outdata).cast("B")
        view[: len(data)] = data
        return
    except TypeError:
        pass

    # Fallback for test doubles that behave like bytearrays.
    outdata[: len(data)] = data
