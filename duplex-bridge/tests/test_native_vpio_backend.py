"""Headless contract tests for the native-vpio backend.

VPIO echo-cancellation *efficacy* needs a real device + acoustic loopback and is
covered by the manual on-device smoke (docs/vpio-backend-status.md), not CI. These
tests cover the device-independent contract by spawning a *fake* helper
(``native_helper_fake.py``) that speaks the same length-prefixed stdio protocol:
capture frames reach ``on_frame``, model audio is framed onto the helper's stdin,
graceful fallback when the binary is missing/unspawnable, and clean teardown.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest
from duplex_bridge.audio_backends.base import CaptureFormat
from duplex_bridge.audio_backends.native_vpio_backend import (
    _FRAME_BYTES,
    NativeVpioBackend,
    _resolve_helper_argv,
)

_FAKE = Path(__file__).resolve().parent / "native_helper_fake.py"


def _fake_cmd(*args: str) -> str:
    parts = [sys.executable, str(_FAKE), *args]
    return " ".join(shlex.quote(p) for p in parts)


class FakeSession:
    """Minimal DuplexSession stand-in capturing the on_audio_out subscription."""

    def __init__(self) -> None:
        self._cb = None
        self._interrupt_cb = None

    def on_audio_out(self, callback) -> None:  # noqa: ANN001
        self._cb = callback

    def on_interrupt(self, callback) -> None:  # noqa: ANN001
        self._interrupt_cb = callback

    def emit_model_audio(self, audio: bytes) -> None:
        assert self._cb is not None, "backend did not subscribe to on_audio_out"
        self._cb(audio)

    def trigger_interrupt(self) -> None:
        assert self._interrupt_cb is not None, "backend did not subscribe to on_interrupt"
        self._interrupt_cb()


async def _wait_for(predicate, timeout: float = 5.0) -> None:  # noqa: ANN001
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met within timeout")


def test_capture_config_is_100ms_at_16k() -> None:
    cfg = NativeVpioBackend().capture_config
    assert isinstance(cfg, CaptureFormat)
    assert (cfg.sample_rate, cfg.channels, cfg.frame_samples) == (16_000, 1, 1600)
    assert cfg.frame_samples * 2 == _FRAME_BYTES


def test_stats_shape() -> None:
    assert NativeVpioBackend().stats == {
        "dropped_frames": 0,
        "model_chunks_dropped": 0,
        "interruptions": 0,
    }


@pytest.mark.asyncio
async def test_start_returns_false_when_helper_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import duplex_bridge.audio_backends.native_vpio_backend as mod

    monkeypatch.setattr(mod, "_resolve_helper_argv", lambda: None)
    backend = NativeVpioBackend()
    ok = await backend.start(lambda _f: None, FakeSession(), asyncio.get_running_loop())
    assert ok is False


@pytest.mark.asyncio
async def test_start_returns_false_when_helper_unspawnable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIMER_VPIO_HELPER", "/nonexistent/aimer-vpio-helper-xyz")
    backend = NativeVpioBackend()
    ok = await backend.start(lambda _f: None, FakeSession(), asyncio.get_running_loop())
    assert ok is False


@pytest.mark.asyncio
async def test_capture_frames_reach_on_frame_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMER_VPIO_HELPER", _fake_cmd("--frames", "3", "--hold"))
    frames: list[bytes] = []
    backend = NativeVpioBackend()
    ok = await backend.start(frames.append, FakeSession(), asyncio.get_running_loop())
    assert ok is True
    try:
        await _wait_for(lambda: len(frames) >= 3)
    finally:
        await backend.stop()

    assert all(len(f) == _FRAME_BYTES for f in frames[:3])
    # Fake fills frame i with byte i, so order is verifiable by the first byte.
    assert [f[0] for f in frames[:3]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_model_audio_is_framed_to_helper_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = tmp_path / "received.txt"
    monkeypatch.setenv("AIMER_VPIO_HELPER", _fake_cmd("--record", str(record), "--hold"))
    session = FakeSession()
    backend = NativeVpioBackend()
    ok = await backend.start(lambda _f: None, session, asyncio.get_running_loop())
    assert ok is True
    try:
        session.emit_model_audio(b"\x01\x02" * 100)  # 200 bytes
        session.emit_model_audio(b"\x03\x04" * 50)  # 100 bytes
        await _wait_for(lambda: record.exists() and len(record.read_text().split()) >= 2)
    finally:
        await backend.stop()

    lengths = [int(x) for x in record.read_text().split()]
    assert lengths[:2] == [200, 100]


@pytest.mark.asyncio
async def test_interrupt_sends_flush_sentinel_to_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Barge-in writes a zero-length frame (the flush sentinel) to the helper."""
    record = tmp_path / "received.txt"
    monkeypatch.setenv("AIMER_VPIO_HELPER", _fake_cmd("--record", str(record), "--hold"))
    session = FakeSession()
    backend = NativeVpioBackend()
    ok = await backend.start(lambda _f: None, session, asyncio.get_running_loop())
    assert ok is True
    try:
        session.trigger_interrupt()
        # The fake helper records each frame's payload length; flush is length 0.
        await _wait_for(lambda: record.exists() and "0" in record.read_text().split())
    finally:
        await backend.stop()

    assert "0" in record.read_text().split()
    assert backend.stats["interruptions"] == 1


@pytest.mark.asyncio
async def test_stop_terminates_helper_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMER_VPIO_HELPER", _fake_cmd("--hold"))
    backend = NativeVpioBackend()
    ok = await backend.start(lambda _f: None, FakeSession(), asyncio.get_running_loop())
    assert ok is True
    proc = backend._proc
    assert proc is not None and proc.returncode is None

    await backend.stop()
    assert proc.returncode is not None  # cleanly terminated, no orphan
    assert backend._proc is None


def test_resolve_helper_argv_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMER_VPIO_HELPER", "python helper.py --flag")
    assert _resolve_helper_argv() == ["python", "helper.py", "--flag"]
