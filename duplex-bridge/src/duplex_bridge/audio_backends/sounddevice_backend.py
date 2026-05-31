"""Cross-platform sounddevice audio backend (no echo cancellation).

Owns the microphone ``RawInputStream`` and an internal ``SpeakerOutput`` (which
holds the ``session.on_audio_out`` subscription). Emits raw mic frames to the
pipeline via ``on_frame``. This is the default backend; use headphones or
push-to-talk to avoid speaker-to-mic feedback.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
from typing import Any

from duplex_bridge.audio_backends.base import CaptureFormat, OnFrame
from duplex_bridge.session import DuplexSession

logger = logging.getLogger(__name__)


class SoundDeviceBackend:
    """Microphone capture + speaker playback via sounddevice/PortAudio."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        blocksize: int = 1_600,
        dtype: str = "int16",
        device: str | int | None = None,
    ) -> None:
        self._fmt = CaptureFormat(
            sample_rate=sample_rate, channels=channels, frame_samples=blocksize
        )
        self._dtype = dtype
        self._device = device
        self._on_frame: OnFrame | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._stream: Any = None
        self._speaker: Any = None

    @property
    def capture_config(self) -> CaptureFormat:
        return self._fmt

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._speaker.stats) if self._speaker is not None else {}

    async def start(
        self,
        on_frame: OnFrame,
        session: DuplexSession,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        if self._running:
            return True

        try:
            sounddevice = importlib.import_module("sounddevice")
        except ImportError:
            logger.warning("[audio-in] sounddevice unavailable; install duplex-bridge[audio]")
            return False

        self._on_frame = on_frame
        self._loop = loop
        self._running = True

        # Speaker owns the on_audio_out subscription.
        from duplex_bridge.audio_output import SpeakerOutput

        self._speaker = SpeakerOutput()
        speaker_ok = self._speaker.start(session)

        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=self._fmt.sample_rate,
                channels=self._fmt.channels,
                dtype=self._dtype,
                blocksize=self._fmt.frame_samples,
                device=self._device,
                callback=self._input_callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.warning("[audio-in] microphone unavailable: %s", exc)
            await self.stop()
            return False

        if not speaker_ok:
            await self.stop()
            return False

        logger.info("[audio-in] microphone capture started")
        return True

    async def stop(self) -> None:
        self._running = False

        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None

        if self._speaker is not None:
            self._speaker.stop()
            self._speaker = None

        self._on_frame = None
        self._loop = None

    def _input_callback(
        self,
        indata: Any,
        _frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.debug("[audio-in] input status: %s", status)
        if self._loop is None or not self._running or self._on_frame is None:
            return
        frames_bytes = _pcm_bytes(indata)
        self._loop.call_soon_threadsafe(self._on_frame, frames_bytes)


def _pcm_bytes(indata: Any) -> bytes:
    tobytes = getattr(indata, "tobytes", None)
    if callable(tobytes):
        return bytes(tobytes())
    return bytes(indata)
