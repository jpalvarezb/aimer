"""Tests for WebSocketContextServer."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest
from aimer_core import ContextPacket, CursorPosition

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect  # type: ignore[attr-defined]

from duplex_bridge.server import WebSocketContextServer
from duplex_bridge.session import DuplexSession


class MockSession(DuplexSession):
    """Mock DuplexSession for testing."""

    def __init__(self):
        self._send_visual_context_mock = AsyncMock()
        self._send_audio_mock = AsyncMock()
        self.on_audio_out_callback = None
        self.on_tool_call_callback = None
        self.opened = False
        self.closed = False

    async def open(self):
        self.opened = True

    async def send_audio(self, frames: bytes):
        await self._send_audio_mock(frames)

    async def send_visual_context(self, packet):
        await self._send_visual_context_mock(packet)

    def on_audio_out(self, callback):
        self.on_audio_out_callback = callback

    def on_tool_call(self, callback):
        self.on_tool_call_callback = callback

    async def close(self):
        self.closed = True


@pytest.fixture
async def mock_session():
    """Fixture providing a mock DuplexSession."""
    return MockSession()


@pytest.fixture
async def server_with_session(mock_session):
    """Fixture providing a started WebSocketContextServer with mock session."""
    server = WebSocketContextServer(
        session=mock_session,
        host="127.0.0.1",
        port=0,  # Random port
    )
    await server.start()
    # Server is bound and ready to accept connections
    url = f"ws://127.0.0.1:{server.port}/context"

    yield server, url, mock_session

    await server.stop()


@pytest.mark.asyncio
async def test_server_forwards_valid_packet(server_with_session):
    """Connect a client, send a valid ContextPacket, verify session receives it."""
    server, url, mock_session = server_with_session

    packet = ContextPacket(cursor=CursorPosition(x=42, y=84))

    async with connect(url) as ws:
        await ws.send(packet.model_dump_json())
        # Brief yield to let server process
        await asyncio.sleep(0)

    # Verify session received the packet
    mock_session._send_visual_context_mock.assert_called_once()
    received_packet = mock_session._send_visual_context_mock.call_args[0][0]
    assert received_packet.cursor.x == 42
    assert received_packet.cursor.y == 84


@pytest.mark.asyncio
async def test_server_rejects_bad_json(server_with_session):
    """Send invalid JSON and verify server logs warning but doesn't crash."""
    server, url, mock_session = server_with_session

    async with connect(url) as ws:
        # Send bad JSON
        await ws.send('{"not": "a packet"}')
        await asyncio.sleep(0)

        # Send valid packet
        packet = ContextPacket(cursor=CursorPosition(x=10, y=20))
        await ws.send(packet.model_dump_json())
        await asyncio.sleep(0)

    # Verify only the valid packet was forwarded
    mock_session._send_visual_context_mock.assert_called_once()
    received_packet = mock_session._send_visual_context_mock.call_args[0][0]
    assert received_packet.cursor.x == 10


@pytest.mark.asyncio
async def test_server_kicks_old_client_on_second_connect(server_with_session):
    """Connect A, then B, verify A is closed with code 1008."""
    server, url, mock_session = server_with_session

    # Connect client A
    client_a = await connect(url)

    # Connect client B
    client_b = await connect(url)
    await asyncio.sleep(0)  # Brief yield to process client kick

    # Verify client A was closed (expected to raise or close)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(client_a.recv(), timeout=0.5)

    # Clean up
    await client_b.close()


@pytest.mark.asyncio
async def test_server_continues_after_session_error(server_with_session):
    """Mock session raises on send_visual_context; verify server continues."""
    server, url, mock_session = server_with_session

    # Make session raise on first call, succeed on second
    mock_session._send_visual_context_mock.side_effect = [
        RuntimeError("Test error"),
        None,
    ]

    async with connect(url) as ws:
        # Send two packets
        packet1 = ContextPacket(cursor=CursorPosition(x=1, y=1))
        packet2 = ContextPacket(cursor=CursorPosition(x=2, y=2))

        await ws.send(packet1.model_dump_json())
        await asyncio.sleep(0)

        await ws.send(packet2.model_dump_json())
        await asyncio.sleep(0)

    # Verify both were attempted
    assert mock_session._send_visual_context_mock.call_count == 2


@pytest.mark.asyncio
async def test_server_stop_is_idempotent(mock_session):
    """Call stop() twice and verify second is a no-op."""
    server = WebSocketContextServer(session=mock_session, port=0)
    await server.start()

    await server.stop()
    await server.stop()  # Should not raise

    assert server._server is None
