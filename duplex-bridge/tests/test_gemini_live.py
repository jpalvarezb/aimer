"""Tests for GeminiLiveSession."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aimer_core import ContextPacket, CursorPosition, FocusWindow, HoverRegion, SemanticContext
from duplex_bridge.providers.gemini_live import GeminiLiveSession
from google.genai import types


class _AsyncIter:
    """Helper to create proper async iterators for testing."""

    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class _ErrorIter:
    """Async iterator that raises once entered."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise RuntimeError("Simulated session error")


class _PendingIter:
    """Async iterator that stays open until the receive task is cancelled."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()
        raise StopAsyncIteration


class _QueueIter:
    """Async iterator that yields messages pushed by the test."""

    def __init__(self):
        self.queue = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()
        if item is StopAsyncIteration:
            raise StopAsyncIteration
        return item


@pytest.fixture
def mock_genai_client():
    """Mock the google.genai.Client."""
    with patch("duplex_bridge.providers.gemini_live.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the session context manager
        mock_session = MagicMock()
        mock_session.send_realtime_input = AsyncMock()
        mock_session.receive = MagicMock(return_value=_PendingIter())

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock()

        mock_client.aio.live.connect.return_value = mock_session_ctx

        yield mock_client, mock_session, mock_session_ctx


@pytest.mark.asyncio
async def test_open_starts_session_and_recv_loop(mock_genai_client, monkeypatch):
    """Verify open() initializes client and starts recv loop."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()

    try:
        # Verify client.aio.live.connect was called
        mock_client.aio.live.connect.assert_called_once()
        call_kwargs = mock_client.aio.live.connect.call_args[1]
        assert call_kwargs["model"] == "gemini-3.1-flash-live-preview"
        assert "config" in call_kwargs
        assert call_kwargs["config"].response_modalities == ["AUDIO"]
        assert call_kwargs["config"].realtime_input_config is None

        # Verify session loop started
        assert session._session_task is not None
        assert not session._session_task.done()

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_with_tile_streams_video_only(mock_genai_client, monkeypatch):
    """In auto-VAD, a packet with a tile sends ONLY the tile on realtime-video; the text
    annotation is never blasted per-packet (it would interrupt) — it is cached for the turn."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")  # auto-VAD
    await session.open()

    try:
        tile_bytes = b"\xff\xd8\xff\xe0"  # JPEG header
        packet = ContextPacket(
            cursor=CursorPosition(x=100, y=200),
            focus_window=FocusWindow(app="TestApp", title="TestWindow"),
            hover_region=HoverRegion(tile_b64=base64.b64encode(tile_bytes).decode()),
            semantic=SemanticContext(selected_text="test selection"),
        )

        await session.send_visual_context(packet)
        await asyncio.sleep(0.1)

        # Exactly one realtime send — the tile on the video channel, no text per-packet.
        assert mock_session.send_realtime_input.call_count == 1
        call = mock_session.send_realtime_input.call_args
        assert "video" in call[1]
        assert "text" not in call[1]
        # The packet is cached for per-turn injection.
        assert session._latest_packet is packet

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_without_tile_sends_nothing(mock_genai_client, monkeypatch):
    """A packet with no tile sends nothing per-packet (the annotation is never blasted);
    it is only cached for per-turn injection."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")  # auto-VAD
    await session.open()

    try:
        packet = ContextPacket(
            cursor=CursorPosition(x=50, y=75),
            focus_window=FocusWindow(app="NoTileApp"),
            hover_region=None,
        )

        await session.send_visual_context(packet)
        await asyncio.sleep(0.1)

        mock_session.send_realtime_input.assert_not_called()
        assert session._latest_packet is packet

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_throttles_tile_to_1fps(mock_genai_client, monkeypatch):
    """In auto-VAD, tiles stream on realtime-video but are throttled to <=1 FPS, keyed on
    packet capture time (packet.t) so the test is deterministic without a fake clock."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")  # auto-VAD
    await session.open()

    try:
        tile = base64.b64encode(b"\xff\xd8\xff\xe0").decode()

        def packet_at(t: float) -> ContextPacket:
            return ContextPacket(
                t=t, cursor=CursorPosition(x=0, y=0), hover_region=HoverRegion(tile_b64=tile)
            )

        await session.send_visual_context(packet_at(0.0))  # first → sent
        await session.send_visual_context(packet_at(0.5))  # within 1s → throttled
        await session.send_visual_context(packet_at(1.01))  # >1s later → sent
        await asyncio.sleep(0.1)

        assert mock_session.send_realtime_input.call_count == 2
        assert all("video" in c[1] for c in mock_session.send_realtime_input.call_args_list)

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_streams_during_turn_only(mock_genai_client, monkeypatch):
    """In manual VAD, visual context is cached (not sent) between turns; during an active turn
    BOTH the tile (video) and the text annotation stream so the model tracks mid-sentence
    pointing/selection. Streaming stops when the turn ends."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview", manual_vad=True)
    await session.open()

    try:
        tile = base64.b64encode(b"\xff\xd8\xff\xe0").decode()

        def packet_at(t: float) -> ContextPacket:
            return ContextPacket(
                t=t, cursor=CursorPosition(x=0, y=0), hover_region=HoverRegion(tile_b64=tile)
            )

        # Between turns: cache only, nothing sent.
        await session.send_visual_context(packet_at(0.0))
        mock_session.send_realtime_input.assert_not_called()

        # Turn start force-sends fresh context and sets the throttle clock.
        await session.send_activity_start()
        mock_session.send_realtime_input.reset_mock()

        # During the turn, a newer packet (>1s later) streams tile AND text.
        await session.send_visual_context(packet_at(1.5))
        kinds = [next(iter(c[1])) for c in mock_session.send_realtime_input.call_args_list]
        assert kinds == ["video", "text"]

        # After the turn ends, streaming stops again.
        await session.send_activity_end()
        mock_session.send_realtime_input.reset_mock()
        await session.send_visual_context(packet_at(3.0))
        mock_session.send_realtime_input.assert_not_called()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_close_is_idempotent(mock_genai_client, monkeypatch):
    """Call close() twice and verify __aexit__ is called at most once."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()

    await session.close()
    await session.close()  # Should not raise

    # Verify __aexit__ was called exactly once
    mock_session_ctx.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_recv_loop_dispatches_audio_to_callback(mock_genai_client, monkeypatch):
    """Mock server message with audio and verify callback is invoked."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    # Create a mock message with audio data
    mock_message = MagicMock()
    mock_message.data = b"audio_pcm_data"
    mock_message.tool_call = None

    # Make receive() return an async iterator
    mock_session.receive = MagicMock(return_value=_AsyncIter([mock_message]))

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    # Register callback
    audio_received = []

    def audio_callback(data: bytes):
        audio_received.append(data)

    session.on_audio_out(audio_callback)

    await session.open()

    try:
        # Wait for recv loop to process
        await asyncio.sleep(0.2)

        # Verify callback was invoked
        assert len(audio_received) == 1
        assert audio_received[0] == b"audio_pcm_data"

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_open_twice_raises(mock_genai_client, monkeypatch):
    """Verify second open() raises RuntimeError."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()

    try:
        with pytest.raises(RuntimeError, match="already open"):
            await session.open()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_missing_api_key_raises(mock_genai_client, monkeypatch):
    """Verify open() raises if API key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        await session.open()


@pytest.mark.asyncio
async def test_recv_loop_dispatches_tool_call_to_callback(mock_genai_client, monkeypatch):
    """Mock server message with tool_call and verify callback is invoked."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    # Create a mock message with tool_call
    mock_message = MagicMock()
    mock_message.data = None
    mock_message.tool_call = {"name": "test_tool", "args": {"key": "value"}}

    # Make receive() return an async iterator
    mock_session.receive = MagicMock(return_value=_AsyncIter([mock_message]))

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    # Register callback
    tool_calls_received = []

    def tool_callback(call):
        tool_calls_received.append(call)

    session.on_tool_call(tool_callback)

    await session.open()

    try:
        # Wait for recv loop to process
        await asyncio.sleep(0.2)

        # Verify callback was invoked
        assert len(tool_calls_received) == 1
        assert tool_calls_received[0]["name"] == "test_tool"

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_recv_loop_dispatches_interrupt_to_callback(mock_genai_client, monkeypatch):
    """A server message flagged interrupted fires the registered interrupt callback."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_message = MagicMock()
    mock_message.data = None
    mock_message.tool_call = None
    mock_message.server_content.interrupted = True

    mock_session.receive = MagicMock(return_value=_AsyncIter([mock_message]))

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    interrupts = []
    session.on_interrupt(lambda: interrupts.append(1))

    await session.open()
    try:
        await asyncio.sleep(0.2)
        assert interrupts == [1]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_recv_loop_no_interrupt_when_not_flagged(mock_genai_client, monkeypatch):
    """A normal audio message does not fire the interrupt callback."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_message = MagicMock()
    mock_message.data = b"audio_pcm_data"
    mock_message.tool_call = None
    mock_message.server_content.interrupted = False

    mock_session.receive = MagicMock(return_value=_AsyncIter([mock_message]))

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")

    interrupts = []
    session.on_interrupt(lambda: interrupts.append(1))

    await session.open()
    try:
        await asyncio.sleep(0.2)
        assert interrupts == []
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_session_reconnects_on_recv_error(mock_genai_client, monkeypatch):
    """Verify recv loop errors trigger a fresh Gemini Live session."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_session.receive = MagicMock(
        side_effect=[
            _ErrorIter(),
            _PendingIter(),
        ]
    )

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()

    try:
        await asyncio.sleep(0.8)

        assert session.stats["reconnects"] >= 1
        assert session._connected

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_during_reconnect_drops(mock_genai_client, monkeypatch):
    """Visual context sent during Gemini reconnect is dropped without raising."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()

    try:
        session._connected = False

        await session.send_visual_context(ContextPacket(cursor=CursorPosition(x=10, y=20)))

        assert session.stats["dropped_during_reconnect"] == 1
        mock_session.send_realtime_input.assert_not_called()

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_ttfb_tracks_first_audio_separately_from_tool_call(
    mock_genai_client,
    monkeypatch,
):
    """An early tool call must not satisfy the first-audio latency metric."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    receive_iter = _QueueIter()
    mock_session.receive = MagicMock(return_value=receive_iter)

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()

    try:
        await session.send_audio(b"\x00\x00" * 1600)
        assert session._first_audio_chunk_send_at is not None
        assert session._first_audio_activity_send_at is None

        await session.send_audio(b"\x00\x04" * 1600)
        assert session._first_audio_activity_send_at is not None

        tool_message = MagicMock()
        tool_message.data = None
        tool_message.tool_call = {"name": "early_tool", "args": {}}
        await receive_iter.queue.put(tool_message)
        await asyncio.sleep(0.05)

        assert session.stats["first_any_response_after_any_send_ms"] is not None
        assert session.stats["first_audio_out_after_any_send_ms"] is None

        audio_message = MagicMock()
        audio_message.data = b"audio-pcm"
        audio_message.mime_type = "audio/pcm;rate=24000"
        audio_message.tool_call = None
        await receive_iter.queue.put(audio_message)
        await asyncio.sleep(0.05)

        assert session.stats["first_audio_out_after_any_send_ms"] is not None
        assert session.stats["first_audio_out_after_first_audio_chunk_send_ms"] is not None
        assert session.stats["first_audio_out_after_first_audio_activity_send_ms"] is not None
        assert session.stats["first_audio_out_after_last_audio_activity_ms"] is not None
        assert (
            session.stats["first_audio_out_after_any_send_ms"]
            >= session.stats["first_any_response_after_any_send_ms"]
        )
        assert (
            session.stats["first_audio_out_after_first_audio_chunk_send_ms"]
            >= session.stats["first_audio_out_after_first_audio_activity_send_ms"]
        )

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_live_config_includes_realtime_input_when_requested(
    mock_genai_client,
    monkeypatch,
):
    """VAD tuning is opt-in and passes through google-genai enum values."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        turn_coverage="activity_only",
        vad_start_sensitivity="high",
        vad_silence_ms=500,
    )
    await session.open()

    try:
        config = mock_client.aio.live.connect.call_args[1]["config"]
        realtime = config.realtime_input_config
        assert realtime is not None
        assert str(realtime.turn_coverage).endswith("TURN_INCLUDES_ONLY_ACTIVITY")
        assert realtime.automatic_activity_detection is not None
        assert realtime.automatic_activity_detection.silence_duration_ms == 500
        assert str(realtime.automatic_activity_detection.start_of_speech_sensitivity).endswith(
            "START_SENSITIVITY_HIGH"
        )

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_manual_vad_disables_automatic_detection(mock_genai_client, monkeypatch):
    """manual_vad sets automatic_activity_detection.disabled=True so the caller drives turns."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview", manual_vad=True)
    await session.open()

    try:
        config = mock_client.aio.live.connect.call_args[1]["config"]
        realtime = config.realtime_input_config
        assert realtime is not None
        assert realtime.automatic_activity_detection.disabled is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_thinking_level_and_end_sensitivity_pass_through(mock_genai_client, monkeypatch):
    """thinking_level and vad_end_sensitivity map to google-genai enum values."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        thinking_level="minimal",
        vad_end_sensitivity="high",
    )
    await session.open()

    try:
        config = mock_client.aio.live.connect.call_args[1]["config"]
        assert config.thinking_config is not None
        assert str(config.thinking_config.thinking_level).endswith("MINIMAL")
        end_sens = (
            config.realtime_input_config.automatic_activity_detection.end_of_speech_sensitivity
        )
        assert str(end_sens).endswith("END_SENSITIVITY_HIGH")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_activity_markers_noop_without_manual_vad(mock_genai_client, monkeypatch):
    """send_activity_start/end are no-ops unless manual_vad is enabled."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")  # manual_vad=False
    await session.open()

    try:
        await session.send_activity_start()
        await session.send_activity_end()
        mock_session.send_realtime_input.assert_not_called()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_activity_end_signals_turn_end_in_manual_vad(mock_genai_client, monkeypatch):
    """In manual_vad mode, send_activity_end forwards an ActivityEnd to the session."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview", manual_vad=True)
    await session.open()

    try:
        await session.send_activity_start()
        await session.send_activity_end()
        kwargs = [c[1] for c in mock_session.send_realtime_input.call_args_list]
        assert any("activity_start" in k for k in kwargs)
        assert any("activity_end" in k for k in kwargs)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_activity_start_injects_cached_context(mock_genai_client, monkeypatch):
    """In manual VAD, a turn start injects the cached tile (video) + annotation (text)
    exactly once, after the activity_start marker — the safe turn-start injection point."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview", manual_vad=True)
    await session.open()

    try:
        tile = base64.b64encode(b"\xff\xd8\xff\xe0").decode()
        packet = ContextPacket(
            cursor=CursorPosition(x=10, y=20),
            focus_window=FocusWindow(app="Editor", title="main.py"),
            hover_region=HoverRegion(tile_b64=tile),
            semantic=SemanticContext(selected_text="def foo"),
        )

        # Caching in manual VAD must NOT send anything per-packet.
        await session.send_visual_context(packet)
        mock_session.send_realtime_input.assert_not_called()

        # Turn start injects: activity_start marker, then tile (video), then annotation (text).
        await session.send_activity_start()
        kinds = [next(iter(c[1])) for c in mock_session.send_realtime_input.call_args_list]
        assert kinds == ["activity_start", "video", "text"]
        text = mock_session.send_realtime_input.call_args_list[-1][1]["text"]
        assert "app=Editor" in text
        assert "title=main.py" in text
        assert "cursor=(10,20)" in text
        assert "selected=def foo" in text
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_activity_start_without_cached_packet_sends_only_marker(
    mock_genai_client, monkeypatch
):
    """With no visual context cached yet, send_activity_start sends only the marker."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview", manual_vad=True)
    await session.open()

    try:
        await session.send_activity_start()
        kwargs = [c[1] for c in mock_session.send_realtime_input.call_args_list]
        assert any("activity_start" in k for k in kwargs)
        assert all("text" not in k and "video" not in k for k in kwargs)
    finally:
        await session.close()


# --- Week 9: tool responses + cancellation ---------------------------------------


@pytest.mark.asyncio
async def test_send_tool_response_builds_final_function_response(mock_genai_client, monkeypatch):
    """A final tool response carries the call id, WHEN_IDLE scheduling, will_continue=False."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_session.send_tool_response = AsyncMock()

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()
    try:
        await session.send_tool_response(
            name="computer_use", call_id="fc_1", response={"status": "ok"}
        )
        mock_session.send_tool_response.assert_awaited_once()
        fr = mock_session.send_tool_response.await_args.kwargs["function_responses"]
        assert fr.id == "fc_1"
        assert fr.name == "computer_use"
        assert fr.response == {"status": "ok"}
        assert fr.scheduling == types.FunctionResponseScheduling.WHEN_IDLE
        assert fr.will_continue is False
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_tool_response_nonfinal_is_silent_ack(mock_genai_client, monkeypatch):
    """final=False is the NON_BLOCKING ack: SILENT scheduling + will_continue=True."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mock_session.send_tool_response = AsyncMock()

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    await session.open()
    try:
        await session.send_tool_response(
            name="computer_use", call_id="fc_2", response={"status": "started"}, final=False
        )
        fr = mock_session.send_tool_response.await_args.kwargs["function_responses"]
        assert fr.scheduling == types.FunctionResponseScheduling.SILENT
        assert fr.will_continue is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_recv_loop_dispatches_tool_call_cancellation(mock_genai_client, monkeypatch):
    """A tool_call_cancellation message fires the registered callback with the ids."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_message = MagicMock()
    mock_message.data = None
    mock_message.tool_call = None
    mock_message.tool_call_cancellation.ids = ["fc_1", "fc_2"]
    mock_session.receive = MagicMock(return_value=_AsyncIter([mock_message]))

    session = GeminiLiveSession(model="gemini-3.1-flash-live-preview")
    cancelled: list[list[str]] = []
    session.on_tool_call_cancellation(cancelled.append)

    await session.open()
    try:
        await asyncio.sleep(0.2)
        assert cancelled == [["fc_1", "fc_2"]]
    finally:
        await session.close()


def test_build_tools_maps_nonblocking_behavior():
    """An optional 'behavior' key on a declaration maps to types.Behavior."""
    session = GeminiLiveSession(
        model="gemini-3.1-flash-live-preview",
        tools=[
            {"name": "long_op", "behavior": "NON_BLOCKING"},
            {"name": "quick_op"},
        ],
    )
    (tool,) = session._build_tools()
    long_op, quick_op = tool.function_declarations
    assert long_op.behavior == types.Behavior.NON_BLOCKING
    assert quick_op.behavior is None
