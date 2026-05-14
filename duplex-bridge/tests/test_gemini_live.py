"""Tests for GeminiLiveSession."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aimer_core import ContextPacket, CursorPosition, FocusWindow, HoverRegion, SemanticContext
from duplex_bridge.providers.gemini_live import GeminiLiveSession


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


@pytest.fixture
def mock_genai_client():
    """Mock the google.genai.Client."""
    with patch("duplex_bridge.providers.gemini_live.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the session context manager
        mock_session = MagicMock()
        mock_session.send_realtime_input = AsyncMock()
        mock_session.receive = MagicMock(return_value=_AsyncIter([]))

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

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")
    await session.open()

    try:
        # Verify client.aio.live.connect was called
        mock_client.aio.live.connect.assert_called_once()
        call_kwargs = mock_client.aio.live.connect.call_args[1]
        assert call_kwargs["model"] == "gemini-live-2.5-flash-preview"
        assert "config" in call_kwargs

        # Verify recv loop started
        assert session._recv_task is not None
        assert not session._recv_task.done()

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_with_tile(mock_genai_client, monkeypatch):
    """Send ContextPacket with tile_b64 and verify image and text are sent."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")
    await session.open()

    try:
        # Create packet with tile
        tile_bytes = b"\xff\xd8\xff\xe0"  # JPEG header
        packet = ContextPacket(
            cursor=CursorPosition(x=100, y=200),
            focus_window=FocusWindow(app="TestApp", title="TestWindow"),
            hover_region=HoverRegion(tile_b64=base64.b64encode(tile_bytes).decode()),
            semantic=SemanticContext(selected_text="test selection"),
        )

        await session.send_visual_context(packet)
        await asyncio.sleep(0.1)

        # Verify two calls: one for image, one for text
        assert mock_session.send_realtime_input.call_count == 2

        # First call should be image
        first_call = mock_session.send_realtime_input.call_args_list[0]
        assert "media" in first_call[1]

        # Second call should be text annotation
        second_call = mock_session.send_realtime_input.call_args_list[1]
        assert "text" in second_call[1]
        text = second_call[1]["text"]
        assert "app=TestApp" in text
        assert "title=TestWindow" in text
        assert "cursor=(100,200)" in text
        assert "selected=test selection" in text

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_visual_context_without_tile_skips_image(mock_genai_client, monkeypatch):
    """Send ContextPacket without tile and verify only text annotation is sent."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")
    await session.open()

    try:
        # Create packet without tile
        packet = ContextPacket(
            cursor=CursorPosition(x=50, y=75),
            focus_window=FocusWindow(app="NoTileApp"),
            hover_region=None,
        )

        await session.send_visual_context(packet)
        await asyncio.sleep(0.1)

        # Verify only one call for text
        assert mock_session.send_realtime_input.call_count == 1
        call = mock_session.send_realtime_input.call_args
        assert "text" in call[1]
        assert "media" not in call[1]

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_close_is_idempotent(mock_genai_client, monkeypatch):
    """Call close() twice and verify __aexit__ is called at most once."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")
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

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")

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

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")
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

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")

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

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")

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
async def test_recv_loop_exits_on_session_error(mock_genai_client, monkeypatch):
    """Verify recv loop exits cleanly when session.receive() raises."""
    mock_client, mock_session, mock_session_ctx = mock_genai_client
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    # Create an async iterator that raises
    class _ErrorIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("Simulated session error")

    mock_session.receive = MagicMock(return_value=_ErrorIter())

    session = GeminiLiveSession(model="gemini-live-2.5-flash-preview")
    await session.open()

    try:
        # Wait for recv loop to hit error and exit
        await asyncio.sleep(0.2)

        # Verify recv task completed (exited due to error)
        assert session._recv_task.done()

    finally:
        await session.close()
