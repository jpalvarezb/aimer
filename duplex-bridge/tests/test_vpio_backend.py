"""Headless contract tests for the VPIO backend.

VPIO echo-cancellation *efficacy* needs a real device + acoustic loopback and is
covered by the manual smoke checklist (docs/), not CI. These tests cover the
device-independent contract: format, fallback, the error-unpack helper, and the
AVFoundation symbol resolution (skipped where AVFoundation is unavailable).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from duplex_bridge.audio_backends.base import CaptureFormat
from duplex_bridge.audio_backends.vpio_backend import VpioBackend, _import_avfaudio, _unpack


class FakeSession:
    def on_audio_out(self, callback) -> None:  # noqa: ANN001
        self._cb = callback

    async def send_audio(self, frames: bytes) -> None:
        pass

    async def send_activity_start(self) -> None:
        pass

    async def send_activity_end(self) -> None:
        pass


def test_unpack_normalizes_bool_and_tuple() -> None:
    # PyObjC may return a bare bool or a (result, error) tuple for error-out selectors.
    assert _unpack(True) == (True, None)
    assert _unpack((False, "err")) == (False, "err")


def test_capture_config_is_100ms_at_16k() -> None:
    cfg = VpioBackend().capture_config
    assert cfg.sample_rate == 16_000
    assert cfg.channels == 1
    assert cfg.frame_samples == 1600
    assert isinstance(cfg, CaptureFormat)


def test_stats_shape() -> None:
    assert VpioBackend().stats == {"engine_restarts": 0}


@pytest.mark.asyncio
async def test_start_returns_false_without_avfoundation(monkeypatch: pytest.MonkeyPatch) -> None:
    import duplex_bridge.audio_backends.vpio_backend as vpio

    monkeypatch.setattr(vpio, "_import_avfaudio", lambda: None)
    backend = VpioBackend()
    ok = await backend.start(lambda _f: None, FakeSession(), asyncio.get_running_loop())
    assert ok is False


def test_import_avfaudio_resolves_on_macos() -> None:
    pytest.importorskip("AVFoundation")
    ns = _import_avfaudio()
    assert ns is not None
    assert {"AVAudioEngine", "AVAudioPlayerNode", "AVAudioPCMBuffer"} <= set(ns)


# --- live-fix (3): _extract_mono — the diagnosed mic-static bug, as a pure numpy function ---
#
# Live finding: on a 9-channel aggregate input device, _handle_tap_buffer's hardcoded
# floatChannelData()[0] read is only correct for a DEINTERLEAVED buffer; when the tap
# buffer is interleaved, [0] reads a stride of every-9th-sample garbage across channels,
# producing static. _extract_mono is factored out as a pure function over plain numpy
# arrays so both layouts are bit-exact testable without PyObjC.


def test_extract_mono_interleaved_9_channel_yields_exact_channel_0() -> None:
    from duplex_bridge.audio_backends.vpio_backend import _extract_mono

    n_frames = 5
    channels = 9
    # Frame-major interleaved layout: sample value = channel*1000 + frame index, so
    # channel 0's samples are exactly 0, 1, 2, 3, 4 — trivially distinguishable from any
    # other channel or from a wrong stride.
    flat = np.array(
        [ch * 1000 + frame for frame in range(n_frames) for ch in range(channels)],
        dtype=np.float32,
    )
    mono = _extract_mono(flat, n_frames, channels, interleaved=True)
    expected = np.arange(n_frames, dtype=np.float32)
    assert mono.dtype == np.float32
    assert mono.shape == (n_frames,)
    assert np.array_equal(mono, expected)


def test_extract_mono_deinterleaved_3_channel_yields_channel_0_with_stride_check() -> None:
    from duplex_bridge.audio_backends.vpio_backend import _extract_mono

    n_frames = 7
    channels = 3
    # Planar/deinterleaved layout: channel 0's whole block comes first.
    ch0 = np.arange(n_frames, dtype=np.float32) + 100.0
    ch1 = np.arange(n_frames, dtype=np.float32) + 200.0
    ch2 = np.arange(n_frames, dtype=np.float32) + 300.0
    flat = np.concatenate([ch0, ch1, ch2])
    mono = _extract_mono(flat, n_frames, channels, interleaved=False)
    assert mono.shape == (n_frames,)
    assert np.array_equal(mono, ch0)


def test_extract_mono_mono_passes_through_unchanged() -> None:
    from duplex_bridge.audio_backends.vpio_backend import _extract_mono

    n_frames = 4
    samples = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    assert np.array_equal(_extract_mono(samples, n_frames, 1, interleaved=True), samples)
    assert np.array_equal(_extract_mono(samples, n_frames, 1, interleaved=False), samples)
