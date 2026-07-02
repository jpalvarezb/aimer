from __future__ import annotations

import argparse
import builtins

import duplex_bridge.__main__ as bridge_main
import pytest


class FakeSession:
    def __init__(self, **_kwargs) -> None:
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    def on_tool_call(self, callback) -> None:
        pass

    def on_tool_call_cancellation(self, callback) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class FakeServer:
    def __init__(self, **_kwargs) -> None:
        self.port = 8765
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class FakeEvent:
    async def wait(self) -> None:
        return None


@pytest.mark.asyncio
async def test_main_no_audio_does_not_import_or_start_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        forbidden = {
            "sounddevice",
            "duplex_bridge.audio_input",
            "duplex_bridge.audio_output",
        }
        if name in forbidden:
            raise AssertionError(f"unexpected audio import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(bridge_main, "GeminiLiveSession", FakeSession)
    monkeypatch.setattr(bridge_main, "WebSocketContextServer", FakeServer)
    monkeypatch.setattr(bridge_main.asyncio, "Event", FakeEvent)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = await bridge_main.async_main(
        argparse.Namespace(
            host="127.0.0.1",
            port=8765,
            gemini_model="test-model",
            api_key_env="GEMINI_API_KEY",
            no_audio=True,
            audio_backend="sounddevice",
            audio_activity_rms_threshold=300.0,
            vad_silence_ms=None,
            vad_start_sensitivity=None,
            turn_coverage=None,
            manual_vad=False,
            end_of_turn_silence_ms=400,
            onset_speech_ms=250,
            thinking_level=None,
            push_to_talk=False,
            ptt_key="cmd_r",
            escalate_full_frame=False,
            deixis_model="gemini-flash-lite-latest",
            no_deixis_resolver=True,
            computer_use_model="gemini-3.5-flash",
            computer_use_max_steps=24,
        )
    )

    assert result == 0
