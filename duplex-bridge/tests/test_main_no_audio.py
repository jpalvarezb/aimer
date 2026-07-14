from __future__ import annotations

import argparse
import asyncio
import builtins
from typing import Any

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

    def set_resume_context_provider(self, provider) -> None:
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


def _base_namespace(**overrides: Any) -> argparse.Namespace:
    """The full argparse.Namespace async_main needs, with --no-audio so the audio backends
    are never touched. Mirrors test_main_no_audio_does_not_import_or_start_sounddevice's
    Namespace; individual wiring tests only need to override a couple of fields."""
    base = dict(
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
    base.update(overrides)
    return argparse.Namespace(**base)


class _RecordingDispatcher:
    """Stands in for ToolDispatcher; records every registered handler by name."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.registered: dict[str, Any] = {}

    def register(self, name: str, handler: Any) -> None:
        self.registered[name] = handler

    def dispatch(self, *args: Any, **kwargs: Any) -> None:
        pass

    def cancel(self, *args: Any, **kwargs: Any) -> None:
        pass


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


# --- live-fix (4c): click_pointer declared and registered with the live dispatcher ---------


@pytest.mark.asyncio
async def test_click_pointer_declared_and_registered_with_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_main, "GeminiLiveSession", FakeSession)
    monkeypatch.setattr(bridge_main, "WebSocketContextServer", FakeServer)
    monkeypatch.setattr(bridge_main.asyncio, "Event", FakeEvent)

    dispatcher = _RecordingDispatcher()
    monkeypatch.setattr(bridge_main, "ToolDispatcher", lambda *a, **kw: dispatcher)

    await bridge_main.async_main(_base_namespace())

    assert "click_pointer" in {d["name"] for d in bridge_main.TOOL_DECLARATIONS}
    assert "click_pointer" in dispatcher.registered


# --- live-fix (1): the delegate research browser is headless (no visible window flash) -----


@pytest.mark.asyncio
async def test_delegate_browser_constructed_headless_for_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_main, "GeminiLiveSession", FakeSession)
    monkeypatch.setattr(bridge_main, "WebSocketContextServer", FakeServer)
    monkeypatch.setattr(bridge_main.asyncio, "Event", FakeEvent)

    captured: dict[str, Any] = {}

    class _RecordingDelegateBrowser:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def handlers_for_task(self, task_id: str) -> dict[str, Any]:
            return {}

        async def close_page(self, task_id: str) -> None:
            pass

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(bridge_main, "DelegateBrowser", _RecordingDelegateBrowser)

    await bridge_main.async_main(_base_namespace())

    assert captured.get("headless") is True


# --- live-fix (2d): _confirm_task forwards approve_all to TaskManager.confirm --------------


@pytest.mark.asyncio
async def test_confirm_task_handler_forwards_approve_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_main, "GeminiLiveSession", FakeSession)
    monkeypatch.setattr(bridge_main, "WebSocketContextServer", FakeServer)
    monkeypatch.setattr(bridge_main.asyncio, "Event", FakeEvent)

    dispatcher = _RecordingDispatcher()
    monkeypatch.setattr(bridge_main, "ToolDispatcher", lambda *a, **kw: dispatcher)

    captured_confirm_args: dict[str, Any] = {}

    class _RecordingTaskManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def pending_note(self) -> str:
            return ""

        async def confirm(
            self, task_id: str, approved: bool, approve_all: bool = False
        ) -> dict[str, Any]:
            captured_confirm_args.update(
                {"task_id": task_id, "approved": approved, "approve_all": approve_all}
            )
            return {"status": "done"}

    monkeypatch.setattr(bridge_main, "TaskManager", _RecordingTaskManager)

    await bridge_main.async_main(_base_namespace())

    confirm_handler = dispatcher.registered["confirm_task"]
    await confirm_handler({"task_id": "task-1", "approved": True, "approve_all": True})

    assert captured_confirm_args == {
        "task_id": "task-1",
        "approved": True,
        "approve_all": True,
    }


# --- live-fix (4b): the live computer_use handler bounds its run with asyncio.wait_for -----


@pytest.mark.asyncio
async def test_live_computer_use_handler_bounds_its_run_with_asyncio_wait_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live finding: two Teams-click computer_use calls never returned a final result — no
    per-goal wall-clock timeout meant a dangling policy call could hang forever, leaving the
    tool call silent. The live handler must wrap its run in asyncio.wait_for with a real,
    bounded timeout (not None / not unbounded)."""
    from duplex_bridge.actions.computer import Action, FakeComputer

    monkeypatch.setattr(bridge_main, "GeminiLiveSession", FakeSession)
    monkeypatch.setattr(bridge_main, "WebSocketContextServer", FakeServer)
    monkeypatch.setattr(bridge_main.asyncio, "Event", FakeEvent)
    monkeypatch.setattr(bridge_main, "MacOSComputer", FakeComputer)
    monkeypatch.setattr(
        bridge_main,
        "_make_computer_policy",
        lambda model, api_key_env: lambda goal, shot, hist: Action("done", note="ok"),
    )

    dispatcher = _RecordingDispatcher()
    monkeypatch.setattr(bridge_main, "ToolDispatcher", lambda *a, **kw: dispatcher)

    real_wait_for = asyncio.wait_for
    seen_timeouts: list[float | None] = []

    async def _spy_wait_for(fut: Any, timeout: float | None = None, *a: Any, **kw: Any) -> Any:
        seen_timeouts.append(timeout)
        return await real_wait_for(fut, timeout, *a, **kw)

    monkeypatch.setattr(bridge_main.asyncio, "wait_for", _spy_wait_for)

    await bridge_main.async_main(_base_namespace())

    handler = dispatcher.registered["computer_use"]
    await handler({"goal": "click the button"})

    assert seen_timeouts, "computer_use handler must bound its run with asyncio.wait_for"
    assert seen_timeouts[0] is not None and seen_timeouts[0] >= 30
