"""Transport sinks for forwarding context packets.

Week 1 emits to stdout or JSONL. Week 3 adds WebSocket transport to duplex-bridge.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from aimer_core import ContextPacket

try:
    from websockets.asyncio.client import connect
except ImportError:
    # TODO(week-4): migrate to asyncio.client when all environments have websockets v15+
    from websockets import connect  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

_SEND_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class WebSocketTransportConfig:
    """Configuration for WebSocket packet sink."""

    url: str = "ws://127.0.0.1:8765/context"
    max_queue: int = 64
    send_timeout_s: float = 1.0
    reconnect_cap_s: float = 4.0


@dataclass
class _Stats:
    """Internal statistics for debugging."""

    sent: int = 0
    dropped: int = 0
    reconnects: int = 0


class WebSocketPacketSink:
    """Async WebSocket sink for streaming packets to duplex-bridge.

    Uses a non-blocking queue to ensure the telemetry loop never blocks on a slow
    consumer. Dropped packets are counted but not retried (freshness > completeness).
    Automatically reconnects with exponential backoff on disconnect.
    """

    def __init__(self, config: WebSocketTransportConfig | None = None) -> None:
        self.config = config or WebSocketTransportConfig()
        self._queue: asyncio.Queue[ContextPacket] = asyncio.Queue(maxsize=self.config.max_queue)
        self._stats = _Stats()
        self._send_task: asyncio.Task[None] | None = None
        self._close_event = asyncio.Event()

    async def start(self) -> None:
        """Start the background send loop."""
        if self._send_task is not None:
            raise RuntimeError("WebSocketPacketSink is already started")
        self._send_task = asyncio.create_task(self._send_loop())

    async def __call__(self, packet: ContextPacket) -> None:
        """Queue a packet for sending. Drops if queue is full (non-blocking)."""
        if self._send_task is None:
            await self.start()
        try:
            self._queue.put_nowait(packet)
        except asyncio.QueueFull:
            self._stats.dropped += 1

    @property
    def stats(self) -> dict[str, Any]:
        """Return current statistics for debugging."""
        return {
            "sent": self._stats.sent,
            "dropped": self._stats.dropped,
            "reconnects": self._stats.reconnects,
        }

    async def close(self) -> None:
        """Close the sink and flush the queue."""
        self._close_event.set()
        if self._send_task is not None:
            self._send_task.cancel()
            try:
                await asyncio.wait_for(self._send_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._send_task = None

    async def _send_loop(self) -> None:
        """Background task that connects, sends packets, and reconnects on failure."""
        backoff = 0.5
        ws = None

        while not self._close_event.is_set():
            try:
                # Connect
                ws = await connect(self.config.url)
                logger.info(f"[transport] connected to {self.config.url}")
                backoff = 0.5

                # Send loop
                while not self._close_event.is_set():
                    try:
                        packet = await asyncio.wait_for(
                            self._queue.get(), timeout=self.config.send_timeout_s
                        )
                    except asyncio.TimeoutError:
                        continue

                    try:
                        await asyncio.wait_for(
                            ws.send(packet.model_dump_json()),
                            timeout=self.config.send_timeout_s,
                        )
                        self._stats.sent += 1
                    except asyncio.TimeoutError:
                        logger.warning("[transport] send timeout, dropping packet")
                        self._stats.dropped += 1

            except Exception as e:
                if self._close_event.is_set():
                    break

                logger.info(f"[transport] disconnected: {e}")
                self._stats.reconnects += 1

                # Flush stale packets
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                        self._stats.dropped += 1
                    except asyncio.QueueEmpty:
                        break

                # Reconnect with exponential backoff
                logger.info(f"[transport] reconnecting in {backoff:.1f}s")
                try:
                    await asyncio.wait_for(
                        self._close_event.wait(), timeout=backoff
                    )
                    break  # Close event was set during backoff
                except asyncio.TimeoutError:
                    pass

                backoff = min(backoff * 2, self.config.reconnect_cap_s)

            finally:
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    ws = None
