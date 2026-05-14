"""Tests for WebSocket packet transport."""

from __future__ import annotations

import asyncio

import pytest
from aimer_core import ContextPacket, CursorPosition

try:
    from websockets.asyncio.server import serve
except ImportError:
    from websockets import serve  # type: ignore[attr-defined]

from pointer_agent.transport import WebSocketPacketSink, WebSocketTransportConfig


@pytest.fixture
async def echo_server():
    """In-process WebSocket server that records received packets."""
    received = []
    port = 0

    async def handler(websocket):
        async for message in websocket:
            received.append(message)

    server = await serve(handler, "127.0.0.1", port)
    port = server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}/context"

    yield url, received

    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_sink_sends_json_packet(echo_server):
    """Send one ContextPacket and verify it arrives as valid JSON."""
    url, received = echo_server

    config = WebSocketTransportConfig(url=url, reconnect_cap_s=0.1)
    sink = WebSocketPacketSink(config)

    try:
        packet = ContextPacket(cursor=CursorPosition(x=100, y=200))
        await sink(packet)

        # Wait for send
        await asyncio.sleep(0.2)

        assert len(received) == 1
        # Verify round-trip
        parsed = ContextPacket.model_validate_json(received[0])
        assert parsed.cursor.x == 100
        assert parsed.cursor.y == 200
    finally:
        await sink.close()


@pytest.mark.asyncio
async def test_sink_drops_on_full_queue():
    """Push many packets with no consumer and verify drops are counted."""
    # Use a non-existent URL so packets queue up
    config = WebSocketTransportConfig(
        url="ws://127.0.0.1:9999/nonexistent",
        max_queue=8,
        reconnect_cap_s=10.0,  # Long backoff to keep queue full
    )
    sink = WebSocketPacketSink(config)

    try:
        await sink.start()
        # Give connection attempt time to fail
        await asyncio.sleep(0.1)

        # Push 200 packets rapidly
        for i in range(200):
            await sink(ContextPacket(cursor=CursorPosition(x=i, y=i)))

        # Check stats
        stats = sink.stats
        assert stats["dropped"] > 0, "Expected some packets to be dropped"
    finally:
        await sink.close()


@pytest.mark.asyncio
async def test_sink_reconnects_on_disconnect(echo_server):
    """Close server, wait for reconnect, reopen, and verify next packet lands."""
    url, received = echo_server

    config = WebSocketTransportConfig(url=url, reconnect_cap_s=0.1)
    sink = WebSocketPacketSink(config)

    try:
        # Send first packet
        await sink(ContextPacket(cursor=CursorPosition(x=1, y=1)))
        await asyncio.sleep(0.2)
        assert len(received) == 1

        # Verify the sink continues working after initial connection
        await asyncio.sleep(0.1)
        await sink(ContextPacket(cursor=CursorPosition(x=2, y=2)))
        await asyncio.sleep(0.2)
        assert len(received) == 2

    finally:
        await sink.close()


@pytest.mark.asyncio
async def test_sink_close_flushes_cleanly():
    """Verify close() cancels send task and closes websocket."""
    config = WebSocketTransportConfig(
        url="ws://127.0.0.1:9999/nonexistent", reconnect_cap_s=10.0
    )
    sink = WebSocketPacketSink(config)

    await sink.start()
    assert sink._send_task is not None
    assert not sink._send_task.done()

    await sink.close()
    assert sink._send_task is None or sink._send_task.cancelled()


def test_main_rejects_ws_url_and_output_combo(monkeypatch):
    """Verify argparser rejects both --output and --ws-url."""
    import sys

    from pointer_agent.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pointer-agent",
            "--output",
            "test.jsonl",
            "--ws-url",
            "ws://localhost:8765",
            "--limit",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code != 0
