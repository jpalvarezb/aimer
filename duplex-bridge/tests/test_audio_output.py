from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from duplex_bridge.audio_output import SpeakerOutput, SpeakerOutputConfig


class FakeSession:
    def __init__(self) -> None:
        self.audio_callback = None

    def on_audio_out(self, callback) -> None:
        self.audio_callback = callback


class FakeOutputStream:
    instances: list[FakeOutputStream] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        FakeOutputStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


def test_audio_output_plays_enqueued_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOutputStream.instances.clear()
    fake_sounddevice = SimpleNamespace(RawOutputStream=FakeOutputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    session = FakeSession()
    output = SpeakerOutput(SpeakerOutputConfig(blocksize=2, queue_maxsize=2))

    assert output.start(session)
    assert session.audio_callback is not None

    session.audio_callback(b"\x01\x00\x02\x00")
    outdata = bytearray(4)
    stream = FakeOutputStream.instances[0]
    stream.kwargs["callback"](outdata, 2, None, None)

    assert bytes(outdata) == b"\x01\x00\x02\x00"
    assert output.stats["played_frames"] == 1

    output.stop()
    assert stream.closed


def test_audio_output_underrun_returns_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOutputStream.instances.clear()
    fake_sounddevice = SimpleNamespace(RawOutputStream=FakeOutputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    output = SpeakerOutput(SpeakerOutputConfig(blocksize=2, queue_maxsize=2))
    assert output.start(FakeSession())

    outdata = bytearray(b"\xff\xff\xff\xff")
    stream = FakeOutputStream.instances[0]
    stream.kwargs["callback"](outdata, 2, None, None)

    assert bytes(outdata) == b"\x00\x00\x00\x00"
    assert output.stats["underruns"] == 1

    output.stop()


def test_audio_output_drops_oldest_when_queue_full() -> None:
    output = SpeakerOutput(SpeakerOutputConfig(queue_maxsize=1))
    output._running = True

    output._enqueue(b"old")
    output._enqueue(b"new")

    assert output.stats["dropped_frames"] == 1
    assert output._queue.get_nowait() == b"new"


def test_audio_output_flush_drops_buffered_audio() -> None:
    output = SpeakerOutput(SpeakerOutputConfig(queue_maxsize=4))
    output._running = True

    output._enqueue(b"a")
    output._enqueue(b"b")
    output._pending.extend(b"partial")

    output.flush()

    assert output._queue.empty()
    assert not output._pending


def test_audio_output_gracefully_handles_missing_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duplex_bridge.audio_output as audio_output

    def fail_import(name: str):
        if name == "sounddevice":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(audio_output.importlib, "import_module", fail_import)

    output = SpeakerOutput()

    assert not output.start(FakeSession())


def test_audio_output_logs_health(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    FakeOutputStream.instances.clear()
    fake_sounddevice = SimpleNamespace(RawOutputStream=FakeOutputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    caplog.set_level("INFO", logger="duplex_bridge.audio_output")

    output = SpeakerOutput(SpeakerOutputConfig(blocksize=2, health_log_interval_s=0.0))
    assert output.start(FakeSession())

    outdata = bytearray(4)
    stream = FakeOutputStream.instances[0]
    stream.kwargs["callback"](outdata, 2, None, None)

    assert "played_per_sec" in caplog.text
    assert "underruns=1" in caplog.text

    output.stop()
