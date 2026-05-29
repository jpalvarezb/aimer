"""Verify that MicCapture and SpeakerOutput are started when --no-audio is not set."""

from __future__ import annotations

import argparse

import duplex_bridge.__main__ as bridge_main
import pytest


class FakeSession:
    def __init__(self, **_kwargs) -> None:
        pass

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass


class FakeServer:
    def __init__(self, **_kwargs) -> None:
        self.port = 8765

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FakeEvent:
    async def wait(self) -> None:
        return None


class FakeMicCapture:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self, session: object, loop: object) -> bool:
        self.started = True
        return True

    async def stop(self) -> None:
        self.stopped = True


class FakeSpeakerOutput:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self, session: object) -> bool:
        self.started = True
        return True

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_audio_wired_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_mic = FakeMicCapture()
    fake_speaker = FakeSpeakerOutput()

    monkeypatch.setattr(bridge_main, "GeminiLiveSession", lambda **_: FakeSession())
    monkeypatch.setattr(bridge_main, "WebSocketContextServer", lambda **_: FakeServer())
    monkeypatch.setattr(bridge_main.asyncio, "Event", FakeEvent)

    # Patch at the module level so the local imports inside async_main resolve to fakes
    import duplex_bridge.audio_input as audio_input_mod
    import duplex_bridge.audio_output as audio_output_mod

    monkeypatch.setattr(audio_input_mod, "MicCapture", lambda *a, **k: fake_mic)
    monkeypatch.setattr(audio_output_mod, "SpeakerOutput", lambda *a, **k: fake_speaker)

    result = await bridge_main.async_main(
        argparse.Namespace(
            host="127.0.0.1",
            port=8765,
            gemini_model="test-model",
            api_key_env="GEMINI_API_KEY",
            no_audio=False,
            audio_activity_rms_threshold=300.0,
            vad_silence_ms=None,
            vad_start_sensitivity=None,
            turn_coverage=None,
            manual_vad=False,
            end_of_turn_silence_ms=400,
            thinking_level=None,
            push_to_talk=False,
            ptt_key="cmd_r",
        )
    )

    assert result == 0
    assert fake_mic.started, "MicCapture.start() was not called"
    assert fake_speaker.started, "SpeakerOutput.start() was not called"
    captured = capsys.readouterr()
    assert "audio=enabled" in captured.out
