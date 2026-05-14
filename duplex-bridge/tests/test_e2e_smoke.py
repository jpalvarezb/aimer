"""End-to-end smoke test for WebSocket transport.

This test spins up both WebSocketContextServer (duplex-bridge) and
WebSocketPacketSink (pointer-agent) in-process to verify the full pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aimer_core import ContextPacket, CursorPosition

# Import from both packages
from pointer_agent.transport import WebSocketPacketSink, WebSocketTransportConfig

from duplex_bridge.server import WebSocketContextServer
from duplex_bridge.session import AudioOutCallback, DuplexSession, ToolCallCallback


class FakeSession(DuplexSession):
    """Fake DuplexSession that records send_visual_context calls."""

    def __init__(self):
        self.received_packets: list[ContextPacket] = []
        self._open = False

    async def open(self):
        self._open = True

    async def send_audio(self, frames: bytes):
        pass

    async def send_visual_context(self, packet: ContextPacket):
        self.received_packets.append(packet)

    def on_audio_out(self, callback: AudioOutCallback):
        pass

    def on_tool_call(self, callback: ToolCallCallback):
        pass

    async def close(self):
        self._open = False


@pytest.mark.asyncio
async def test_e2e_websocket_transport():
    """Push 5 ContextPackets through sink -> server -> session and verify all arrive."""
    # Create fake session
    fake_session = FakeSession()
    await fake_session.open()

    # Create server on random port
    server = WebSocketContextServer(
        session=fake_session,
        host="127.0.0.1",
        port=0,  # Random port
        path="/context",
    )
    await server.start()
    await asyncio.sleep(0.1)

    # Get actual port
    port = server._server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}/context"

    # Create sink
    config = WebSocketTransportConfig(url=url, reconnect_cap_s=0.1)
    sink = WebSocketPacketSink(config)

    try:
        # Push 5 packets
        expected_positions = [(10, 20), (30, 40), (50, 60), (70, 80), (90, 100)]

        for x, y in expected_positions:
            packet = ContextPacket(cursor=CursorPosition(x=x, y=y))
            await sink(packet)

        # Wait for all packets to arrive
        await asyncio.sleep(0.5)

        # Verify all packets were received
        assert len(fake_session.received_packets) == 5

        for i, (x, y) in enumerate(expected_positions):
            received = fake_session.received_packets[i]
            assert received.cursor.x == x
            assert received.cursor.y == y

        # Verify stats
        stats = sink.stats
        assert stats["sent"] == 5
        assert stats["dropped"] == 0

    finally:
        # Clean up
        await sink.close()
        await server.stop()
        await fake_session.close()
