"""Headless contract tests for the VPIO backend.

VPIO echo-cancellation *efficacy* needs a real device + acoustic loopback and is
covered by the manual smoke checklist (docs/), not CI. These tests cover the
device-independent contract: format, fallback, the error-unpack helper, and the
AVFoundation symbol resolution (skipped where AVFoundation is unavailable).
"""

from __future__ import annotations

import asyncio

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
