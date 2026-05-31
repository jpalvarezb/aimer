from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from duplex_bridge.audio_backends.sounddevice_backend import SoundDeviceBackend


class FakeSession:
    def __init__(self) -> None:
        self.audio_callback = None

    def on_audio_out(self, callback) -> None:  # noqa: ANN001
        self.audio_callback = callback


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


class FakeOutputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeAudio:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


def _fake_sounddevice() -> SimpleNamespace:
    return SimpleNamespace(RawInputStream=FakeInputStream, RawOutputStream=FakeOutputStream)


@pytest.mark.asyncio
async def test_backend_captures_frames_and_subscribes_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeInputStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice())

    collected: list[bytes] = []
    session = FakeSession()
    backend = SoundDeviceBackend(blocksize=2)

    ok = await backend.start(collected.append, session, asyncio.get_running_loop())
    assert ok
    # Backend owns the playback subscription.
    assert session.audio_callback is not None

    stream = FakeInputStream.instances[0]
    stream.kwargs["callback"](FakeAudio(b"\x01\x00\x02\x00"), 2, None, None)
    await asyncio.sleep(0.05)

    assert collected == [b"\x01\x00\x02\x00"]

    await backend.stop()
    assert stream.closed


@pytest.mark.asyncio
async def test_backend_gracefully_handles_missing_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duplex_bridge.audio_backends.sounddevice_backend as sd_backend

    def fail_import(name: str):
        if name == "sounddevice":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(sd_backend.importlib, "import_module", fail_import)

    backend = SoundDeviceBackend()
    ok = await backend.start(lambda _f: None, FakeSession(), asyncio.get_running_loop())
    assert ok is False


def test_backend_capture_config_reflects_blocksize() -> None:
    backend = SoundDeviceBackend(blocksize=320, sample_rate=16_000)
    assert backend.capture_config.frame_samples == 320
    assert backend.capture_config.frame_ms == pytest.approx(20.0)
