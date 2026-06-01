"""Software acoustic-echo-cancellation backend (numpy NLMS).

Cross-platform and CI-testable. Captures the mic via sounddevice, intercepts the
model's output audio to build a time-aligned echo reference, and runs an NLMS
canceller so the assistant's own playback does not leak into the mic (and so the
VAD does not treat it as a barge-in). Emits cleaned 1600-sample (100 ms) frames.

Alignment note: the far-end reference is consumed in lockstep with mic frames,
which assumes capture and playback advance at the same real-time rate. The NLMS
filter absorbs the residual echo delay; a large bulk delay can be pre-compensated
with ``reference_delay_samples``. Robust delay estimation is future work — the
macOS VPIO backend (M4) is the production-grade echo-cancellation path.

Limitations (acceptable for a fallback; not covered by the synthetic tests):
- Model audio arriving in bursts slower than real-time underruns the reference
  buffer, which zero-pads — cancellation pauses until audio resumes.
- The 256-tap (~16 ms) filter covers only short echo tails; far-field speakers or
  reflective rooms (50–150 ms delay) need a larger ``filter_len``.
Emitted frames are 1600 samples (100 ms); MicCapture's VAD timing is frame-count
independent, so ``--end-of-turn-silence-ms`` behaves identically across backends.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from duplex_bridge.audio_backends.base import CaptureFormat, OnFrame
from duplex_bridge.dsp.framing import Reframer
from duplex_bridge.dsp.nlms import NlmsCanceller
from duplex_bridge.dsp.resample import resample_int16
from duplex_bridge.session import DuplexSession

logger = logging.getLogger(__name__)

_CAPTURE_RATE = 16_000
# 100 ms at 16 kHz — matches the sounddevice cadence (~10 sends/sec). Emitting 10 ms
# frames floods the per-frame WebSocket send to Gemini and overflows the input queue,
# producing a choppy stream that breaks automatic end-of-turn detection.
_FRAME_SAMPLES = 1_600


class SoftwareAecBackend:
    """Mic capture + playback with numpy NLMS echo cancellation."""

    def __init__(
        self,
        *,
        sample_rate: int = _CAPTURE_RATE,
        channels: int = 1,
        blocksize: int = 1_600,
        dtype: str = "int16",
        device: str | int | None = None,
        model_rate: int = 24_000,
        filter_len: int = 256,
        reference_delay_samples: int = 0,
    ) -> None:
        if sample_rate != _CAPTURE_RATE:
            raise ValueError("software-aec backend captures at 16 kHz")
        self._fmt = CaptureFormat(
            sample_rate=_CAPTURE_RATE, channels=channels, frame_samples=_FRAME_SAMPLES
        )
        self._mic_blocksize = blocksize
        self._dtype = dtype
        self._device = device
        self._model_rate = model_rate
        self._reference_delay_samples = max(0, reference_delay_samples)

        self._on_frame: OnFrame | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._stream: Any = None
        self._speaker: Any = None

        self._canceller = NlmsCanceller(filter_len=filter_len)
        self._reframer = Reframer(_FRAME_SAMPLES)
        # Far-end (reference) ring buffer of 16 kHz int16 bytes, plus a pre-delay
        # of leading zeros to model bulk echo latency.
        self._far_buf = bytearray(b"\x00\x00" * self._reference_delay_samples)

    @property
    def capture_config(self) -> CaptureFormat:
        return self._fmt

    @property
    def stats(self) -> dict[str, int]:
        stats = dict(self._speaker.stats) if self._speaker is not None else {}
        stats["far_buffer_samples"] = len(self._far_buf) // 2
        return stats

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
            logger.warning("[audio-aec] sounddevice unavailable; install duplex-bridge[audio]")
            return False

        self._on_frame = on_frame
        self._loop = loop
        self._running = True

        # Drive playback ourselves so we can capture the echo reference first.
        from duplex_bridge.audio_output import SpeakerOutput

        self._speaker = SpeakerOutput()
        speaker_ok = self._speaker.start(session=None)
        session.on_audio_out(self._on_model_audio)
        # On barge-in, stop playback and drop the echo reference so the canceller
        # stays aligned with the (now silent) far end.
        session.on_interrupt(self._on_interrupt)

        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=_CAPTURE_RATE,
                channels=self._fmt.channels,
                dtype=self._dtype,
                blocksize=self._mic_blocksize,
                device=self._device,
                callback=self._input_callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.warning("[audio-aec] microphone unavailable: %s", exc)
            await self.stop()
            return False

        if not speaker_ok:
            await self.stop()
            return False

        logger.info("[audio-aec] echo-cancelling capture started (NLMS)")
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

    def _on_interrupt(self) -> None:
        """Barge-in: flush playback and drop the far-end echo reference."""
        if self._speaker is not None:
            self._speaker.flush()
        # Keep only the leading pre-delay zeros so alignment is preserved.
        self._far_buf = bytearray(b"\x00\x00" * self._reference_delay_samples)

    def _on_model_audio(self, audio: bytes) -> None:
        """Play the model audio and append it (resampled) to the echo reference."""
        if self._speaker is not None:
            self._speaker.play(audio)
        reference = resample_int16(audio, self._model_rate, _CAPTURE_RATE)
        self._far_buf.extend(reference)

    def _input_callback(
        self,
        indata: Any,
        _frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.debug("[audio-aec] input status: %s", status)
        if self._loop is None or not self._running:
            return
        self._loop.call_soon_threadsafe(self._ingest_mic, _pcm_bytes(indata))

    def _ingest_mic(self, raw: bytes) -> None:
        for frame in self._reframer.push(raw):
            self._process_frame(frame)

    def _process_frame(self, near_bytes: bytes) -> None:
        """Cancel echo from one mic frame and emit the cleaned frame."""
        if self._on_frame is None:
            return
        near = np.frombuffer(near_bytes, dtype="<i2").astype(np.float64)
        far = self._take_reference(near.shape[0])
        residual = self._canceller.process(near, far)
        cleaned = np.clip(np.rint(residual), -32767, 32767).astype("<i2").tobytes()
        self._on_frame(cleaned)

    def _take_reference(self, n_samples: int) -> NDArray[np.float64]:
        """Pop ``n_samples`` of far-end reference, zero-padding on underrun."""
        need = n_samples * 2
        chunk = self._far_buf[:need]
        del self._far_buf[:need]
        far = np.frombuffer(bytes(chunk), dtype="<i2").astype(np.float64)
        if far.shape[0] < n_samples:
            far = np.pad(far, (0, n_samples - far.shape[0]))
        return far


def _pcm_bytes(indata: Any) -> bytes:
    tobytes = getattr(indata, "tobytes", None)
    if callable(tobytes):
        return bytes(tobytes())
    return bytes(indata)
