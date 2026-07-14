"""macOS Voice Processing I/O backend — hardware acoustic echo cancellation.

Uses a single ``AVAudioEngine`` with ``setVoiceProcessingEnabled`` so Apple's VPIO
cancels the played model audio out of the mic. Capture and playback share one
engine (VPIO derives its echo reference from the engine's own render path), which
is why this backend owns both streams.

Key macOS detail (the reason a naive implementation captures nothing): enabling
voice processing fires an ``AVAudioEngineConfigurationChange`` that STOPS the
engine, so the input tap never fires unless the engine is restarted. A watchdog
restarts it whenever it stops. The VPIO input is multichannel float32 at the
hardware rate (e.g. 48 kHz / 3 ch on a MacBook mic array); an ``AVAudioConverter``
downmixes + resamples it to 16 kHz mono int16. Requires ``duplex-bridge[vpio]``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from typing import Any

import numpy as np

from duplex_bridge.audio_backends.base import CaptureFormat, OnFrame
from duplex_bridge.dsp.resample import resample_f32_to_int16, resample_int16
from duplex_bridge.session import DuplexSession

logger = logging.getLogger(__name__)

_CAPTURE_RATE = 16_000
# 100 ms at 16 kHz. Emitting 10 ms frames floods the per-frame WebSocket send to
# Gemini (~100/sec), overflowing the input queue and producing a choppy stream that
# breaks the model's automatic end-of-turn detection. 100 ms matches the sounddevice
# backend's cadence (~10 sends/sec); the VAD silence-window math is frame-size agnostic.
_FRAME_SAMPLES = 1_600
_WATCHDOG_INTERVAL_S = 0.2


def _unpack(result: Any) -> tuple[Any, Any]:
    """Normalize a PyObjC ``(BOOL, NSError*)`` selector result to ``(ok, err)``.

    Some bindings return a bare bool, others a ``(bool, error)`` tuple; unpacking
    wrong turns a falsy ``(False, error)`` into a truthy value and masks failures.
    """
    return result if isinstance(result, tuple) else (result, None)


def _extract_mono(data: np.ndarray, n_frames: int, channels: int, interleaved: bool) -> np.ndarray:
    """Pull the voice channel (channel 0) out of a raw float32 sample buffer.

    ``data`` is a flat numpy array as read off ``AVAudioPCMBuffer.floatChannelData()``.
    Its layout depends on the tap buffer's actual format:

    - ``interleaved=True``: ``data`` holds ``n_frames * channels`` samples in
      frame-major order (frame0's ``channels`` samples, then frame1's, ...) — this
      is how AVAudioPCMBuffer packs interleaved formats into a single channel
      pointer. Reshape to ``(n_frames, channels)`` and take column 0.
    - ``interleaved=False`` (planar/deinterleaved): each channel already lives in
      its own contiguous block of ``n_frames`` samples; channel 0's block is
      ``data[:n_frames]``.
    - ``channels <= 1``: already mono, pass through unchanged.

    Reading the wrong layout (e.g. blindly slicing the first ``n_frames`` samples
    of an interleaved buffer) silently returns a stride of samples across
    channels instead of one channel's audio — this is the mic-static bug.
    """
    flat = np.asarray(data, dtype=np.float32)
    if channels <= 1:
        return flat
    if interleaved:
        expected = n_frames * channels
        # channels > 1 here (channels <= 1 returned above), so the divide is always safe.
        usable_frames = min(n_frames, flat.size // channels)
        frame_major = flat[: usable_frames * channels].reshape(usable_frames, channels)
        mono = frame_major[:, 0]
        if usable_frames < n_frames:
            logger.debug(
                "[vpio] interleaved tap buffer short: expected %d samples, got %d",
                expected,
                flat.size,
            )
        return np.ascontiguousarray(mono)
    # Planar: defensively clamp to the shorter of the requested frame count and
    # what's actually present, rather than reading past the channel-0 block.
    usable_frames = min(n_frames, flat.size)
    return np.ascontiguousarray(flat[:usable_frames])


class VpioBackend:
    """Mic capture + playback through one VPIO-enabled AVAudioEngine."""

    def __init__(self, *, model_rate: int = 24_000) -> None:
        self._fmt = CaptureFormat(
            sample_rate=_CAPTURE_RATE, channels=1, frame_samples=_FRAME_SAMPLES
        )
        self._model_rate = model_rate
        self._on_frame: OnFrame | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

        self._engine: Any = None
        self._input_node: Any = None
        self._player: Any = None
        self._render_fmt: Any = None
        self._tap_block: Any = None  # pinned so PyObjC doesn't GC the tap callable
        self._reframer: Any = None
        self._watchdog: asyncio.Task[None] | None = None
        self._av: Any = None  # the resolved AVFAudio module namespace
        self._restarts = 0
        # Hold references to scheduled playback buffers until they have rendered —
        # AVAudioPlayerNode does not retain them through PyObjC, so a local buffer is
        # GC'd before it plays (→ silence). maxlen bounds memory; 128 buffers is far
        # more than are ever in flight (each plays within ~100 ms).
        self._scheduled: deque[Any] = deque(maxlen=128)
        # Coalesce streamed model audio into larger buffers before scheduling: the
        # player renders one big buffer fine but drops out on a rapid stream of tiny
        # ones. Accumulate render-rate int16 and flush in fixed chunks.
        self._play_accum = bytearray()
        self._flush_bytes = 0  # set in start() once the render rate is known
        self._extract_path_logged = False  # log the chosen mono-extraction path once

    @property
    def capture_config(self) -> CaptureFormat:
        return self._fmt

    @property
    def stats(self) -> dict[str, int]:
        return {"engine_restarts": self._restarts}

    async def start(
        self,
        on_frame: OnFrame,
        session: DuplexSession,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        if self._running:
            return True
        av = _import_avfaudio()
        if av is None:
            logger.warning("[vpio] AVFoundation unavailable; install duplex-bridge[vpio]")
            return False
        self._av = av

        from duplex_bridge.dsp.framing import Reframer

        self._on_frame = on_frame
        self._loop = loop
        self._reframer = Reframer(_FRAME_SAMPLES)

        engine = av["AVAudioEngine"].alloc().init()
        input_node = engine.inputNode()
        ok, err = _unpack(input_node.setVoiceProcessingEnabled_error_(True, None))
        if not ok:
            logger.warning("[vpio] setVoiceProcessingEnabled failed: %s", err)
            return False

        in_fmt = input_node.outputFormatForBus_(0)
        if in_fmt is None or in_fmt.sampleRate() <= 0:
            logger.warning("[vpio] input format unavailable (sampleRate=0)")
            return False
        self._in_rate = int(in_fmt.sampleRate())
        logger.info("[vpio] input %d Hz / %d ch (VPIO)", self._in_rate, in_fmt.channelCount())

        # Player node for model playback — this is VPIO's echo reference.
        player = av["AVAudioPlayerNode"].alloc().init()
        engine.attachNode_(player)
        # Connect to the OUTPUT node (not mainMixer): VPIO forces the output to the
        # input's rate (48 kHz); the mixer's default 44.1 kHz makes start() fail.
        render_fmt = engine.outputNode().inputFormatForBus_(0)
        engine.connect_to_format_(player, engine.outputNode(), render_fmt)
        self._render_fmt = render_fmt
        self._render_rate = int(render_fmt.sampleRate())
        self._render_channels = int(render_fmt.channelCount())
        self._flush_bytes = int(0.2 * self._render_rate) * 2  # ~200 ms of int16
        self._player = player
        self._input_node = input_node
        self._engine = engine

        self._tap_block = self._make_tap_block()
        input_node.installTapOnBus_bufferSize_format_block_(0, 1024, in_fmt, self._tap_block)

        session.on_audio_out(self._on_model_audio)

        engine.prepare()
        ok, err = _unpack(engine.startAndReturnError_(None))
        if not ok:
            logger.warning("[vpio] engine start failed: %s", err)
            await self.stop()
            return False
        self._running = True
        player.play()
        self._watchdog = loop.create_task(self._restart_watchdog())
        logger.info(
            "[vpio] echo-cancelling capture + playback started (ear-verified on macOS 15.6 / "
            "pyobjc 12.2, 2026-07-03 — the historical silent-playback issue is fixed by the "
            "buffer-retention + coalescing + watchdog re-arm combination; see "
            "docs/vpio-backend-status.md). If playback is silent on YOUR setup, use "
            "--audio-backend native-vpio (just build-native) or software-aec."
        )
        return True

    async def stop(self) -> None:
        self._running = False
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if self._input_node is not None:
            with contextlib.suppress(Exception):
                self._input_node.removeTapOnBus_(0)
                # Clean teardown: disable VP so coreaudiod's aggregate device isn't wedged.
                self._input_node.setVoiceProcessingEnabled_error_(False, None)
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.stop()
        self._engine = self._input_node = self._player = None
        self._tap_block = None
        self._scheduled.clear()
        self._play_accum.clear()
        self._on_frame = None
        self._loop = None

    async def _restart_watchdog(self) -> None:
        """Restart the engine if VPIO's configuration-change stops it (the key fix)."""
        try:
            while self._running:
                await asyncio.sleep(_WATCHDOG_INTERVAL_S)
                if self._engine is not None and not self._engine.isRunning():
                    self._restarts += 1
                    _unpack(self._engine.startAndReturnError_(None))
                    # Restarting the engine stops the player node — re-arm playback,
                    # else model-audio buffers scheduled afterward are silent.
                    if self._player is not None:
                        self._player.play()
        except asyncio.CancelledError:
            raise

    def _make_tap_block(self) -> Any:
        def tap(buf: Any, _when: Any) -> None:
            try:
                self._handle_tap_buffer(buf)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[vpio] tap error: %s", exc)

        return tap

    def _handle_tap_buffer(self, buf: Any) -> None:
        """Read the (echo-cancelled) tap buffer, downmix+resample to 16 kHz int16.

        Channel 0 is VPIO's processed (echo-cancelled) output. The tap buffer is
        USUALLY float32 deinterleaved (planar) at the hardware rate, N channels, in
        which case ``floatChannelData()[0]`` is already channel 0's own block.
        But on some devices (observed: a 9-channel aggregate input) the buffer's
        format is interleaved instead — AVAudioPCMBuffer then packs ALL channels'
        samples into that same single pointer, frame-major. Blindly reading the
        first ``frameLength()`` samples off an interleaved buffer reads a stride
        across channels rather than one channel's audio, which is heard as static.
        We read the buffer's actual format (``isInterleaved()`` / ``channelCount()``)
        and size the raw read accordingly, then hand off to the pure, unit-tested
        ``_extract_mono`` to pick out channel 0. Resampling is done in numpy via the
        M1 resampler.
        """
        n_in = int(buf.frameLength())
        if n_in <= 0:
            return
        fcd = buf.floatChannelData()
        if fcd is None:
            return
        fmt = buf.format()
        channels = int(fmt.channelCount()) if fmt is not None else 1
        interleaved = bool(fmt.isInterleaved()) if fmt is not None else False
        # Defensive stride check where the binding exposes it: a deinterleaved
        # buffer's per-channel stride should be 1 (one float per audio sample); a
        # mismatch means our layout assumption doesn't hold and we fall back to
        # treating the buffer as interleaved (the safer, bounds-checked read).
        stride = getattr(buf, "stride", None)
        if not interleaved and callable(stride):
            with contextlib.suppress(Exception):
                if int(stride()) not in (0, 1):
                    interleaved = True
        raw_count = n_in * channels if interleaved else n_in
        raw = np.frombuffer(fcd[0].as_buffer(raw_count), dtype=np.float32, count=raw_count)
        if not self._extract_path_logged:
            self._extract_path_logged = True
            logger.info(
                "[vpio] tap buffer format: channels=%d interleaved=%s "
                "(mono extraction path chosen once)",
                channels,
                interleaved,
            )
        mono = _extract_mono(raw, n_in, channels, interleaved)
        pcm = resample_f32_to_int16(np.ascontiguousarray(mono), self._in_rate, _CAPTURE_RATE)
        for frame in self._reframer.push(pcm):
            if self._loop is not None and self._on_frame is not None:
                self._loop.call_soon_threadsafe(self._on_frame, frame)

    def _on_model_audio(self, audio: bytes) -> None:
        """Resample + accumulate model audio, scheduling it in coalesced buffers."""
        if not self._running or self._player is None:
            return
        try:
            if len(audio) >= 2:
                # 24 kHz int16 -> render-rate int16, appended to the playback accumulator.
                self._play_accum.extend(resample_int16(audio, self._model_rate, self._render_rate))
            while len(self._play_accum) >= self._flush_bytes:
                chunk = bytes(self._play_accum[: self._flush_bytes])
                del self._play_accum[: self._flush_bytes]
                self._schedule_playback(chunk)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vpio] playback error: %s", exc)

    def _schedule_playback(self, rendered: bytes) -> None:
        """Schedule one render-rate int16 PCM chunk on the player node.

        Writes float samples directly into the buffer's channels — avoiding
        AVAudioConverter, whose alloc'd output buffers don't bridge cleanly.
        """
        av = self._av
        mono = np.frombuffer(rendered, dtype="<i2").astype(np.float32) / 32768.0
        n = mono.shape[0]
        if n == 0:
            return
        buf = av["AVAudioPCMBuffer"].alloc().initWithPCMFormat_frameCapacity_(self._render_fmt, n)
        buf.setFrameLength_(n)
        channels = buf.floatChannelData()
        payload = mono.tobytes()
        # as_buffer(count) takes an ELEMENT count (floats), not bytes.
        for ch in range(self._render_channels):
            channels[ch].as_buffer(n).cast("B")[:] = payload
        self._scheduled.append(buf)  # keep alive until rendered (see __init__)
        self._player.scheduleBuffer_completionHandler_(buf, None)


def _import_avfaudio() -> dict[str, Any] | None:
    """Resolve the audio symbols into a namespace dict.

    Prefer ``AVFoundation`` over the ``AVFAudio`` split module: on pyobjc 12 the
    ``AVFAudio`` module's ``AVAudioPCMBuffer.floatChannelData`` returns an opaque,
    non-subscriptable pointer, while ``AVFoundation`` has the correct manual
    bindings (returns a tuple of channel pointers).
    """
    names = ["AVAudioEngine", "AVAudioPlayerNode", "AVAudioPCMBuffer"]
    for module in ("AVFoundation", "AVFAudio"):
        try:
            mod = __import__(module)
        except ImportError:
            continue
        if all(hasattr(mod, n) for n in names):
            return {n: getattr(mod, n) for n in names}
    return None
