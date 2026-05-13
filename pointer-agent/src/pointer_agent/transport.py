"""Transport sinks for forwarding context packets.

Week 1 emits to stdout or JSONL. The WebSocket sink is reserved for the Week 3
Gemini Live bridge integration.
"""

from __future__ import annotations

from dataclasses import dataclass

from jointer_core import ContextPacket


@dataclass(frozen=True)
class WebSocketTransportConfig:
    url: str = "ws://127.0.0.1:8765/context"


class WebSocketPacketSink:
    """Week 3 placeholder for streaming packets to duplex-bridge."""

    def __init__(self, config: WebSocketTransportConfig | None = None) -> None:
        self.config = config or WebSocketTransportConfig()

    async def __call__(self, _packet: ContextPacket) -> None:
        raise NotImplementedError("WebSocket context transport is planned for Week 3.")
