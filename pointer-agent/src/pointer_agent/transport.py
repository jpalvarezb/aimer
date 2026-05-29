"""Transport sinks for forwarding context packets.

Week 1 emits to stdout or JSONL. Week 3 adds WebSocket transport to duplex-bridge.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from aimer_core import ContextPacket

try:
    from websockets.asyncio.client import connect
except ImportError:
    # TODO(week-4): migrate to asyncio.client when all environments have websockets v15+
    from websockets import connect  # noqa: F811

logger = logging.getLogger(__name__)

_QUEUE_POLL_S = 0.1


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


@dataclass(frozen=True)
class PacketSendTiming:
    """Phase timings for a packet that reached ws.send completion."""

    packet_build_ms: float
    ws_send_ms: float
    total_ms: float


@dataclass
class _QueuedPacket:
    packet: ContextPacket
    timing_future: asyncio.Future[PacketSendTiming] | None = None


class WebSocketPacketSink:
    """Async WebSocket sink for streaming packets to duplex-bridge.

    Uses a non-blocking queue to ensure the telemetry loop never blocks on a slow
    consumer. Dropped packets are counted but not retried (freshness > completeness).
    Automatically reconnects with exponential backoff on disconnect.
    """

    def __init__(self, config: WebSocketTransportConfig | None = None) -> None:
        self.config = config or WebSocketTransportConfig()
        self._queue: asyncio.Queue[_QueuedPacket] = asyncio.Queue(maxsize=self.config.max_queue)
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
            self._queue.put_nowait(_QueuedPacket(packet=packet))
        except asyncio.QueueFull:
            self._stats.dropped += 1

    async def send_with_latency(self, packet: ContextPacket) -> PacketSendTiming:
        """Send a packet and wait until the underlying ws.send call completes.

        Normal telemetry remains freshness-first and non-blocking through
        ``__call__``. Latency profiling opts into this method so total timing can
        include a wire-ish completion point instead of queue insertion.
        """
        if self._send_task is None:
            await self.start()

        loop = asyncio.get_running_loop()
        timing_future: asyncio.Future[PacketSendTiming] = loop.create_future()
        try:
            self._queue.put_nowait(_QueuedPacket(packet=packet, timing_future=timing_future))
        except asyncio.QueueFull as exc:
            self._stats.dropped += 1
            raise RuntimeError("WebSocket packet queue is full") from exc

        return await asyncio.wait_for(
            timing_future, timeout=self.config.send_timeout_s + self.config.reconnect_cap_s
        )

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
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self._send_task, timeout=1.0)
            self._send_task = None

    async def _send_loop(self) -> None:
        """Background task that connects, sends packets, and reconnects on failure."""
        backoff = 0.5
        ws = None

        while not self._close_event.is_set():
            try:
                # Connect
                ws = await connect(self.config.url)
                logger.info("[transport] connected to %s", self.config.url)
                backoff = 0.5

                # Send loop
                while not self._close_event.is_set():
                    try:
                        queued = await asyncio.wait_for(self._queue.get(), timeout=_QUEUE_POLL_S)
                    except asyncio.TimeoutError:
                        continue

                    try:
                        started_at = time.perf_counter()
                        build_started_at = time.perf_counter()
                        payload = queued.packet.model_dump_json()
                        packet_build_ms = (time.perf_counter() - build_started_at) * 1000.0

                        send_started_at = time.perf_counter()
                        await asyncio.wait_for(
                            ws.send(payload),
                            timeout=self.config.send_timeout_s,
                        )
                        ws_send_ms = (time.perf_counter() - send_started_at) * 1000.0
                        total_ms = (time.perf_counter() - started_at) * 1000.0
                        self._stats.sent += 1
                        if queued.timing_future is not None and not queued.timing_future.done():
                            queued.timing_future.set_result(
                                PacketSendTiming(
                                    packet_build_ms=packet_build_ms,
                                    ws_send_ms=ws_send_ms,
                                    total_ms=total_ms,
                                )
                            )
                    except asyncio.TimeoutError:
                        logger.warning("[transport] send timeout, dropping packet")
                        self._stats.dropped += 1
                        if queued.timing_future is not None and not queued.timing_future.done():
                            queued.timing_future.set_exception(
                                TimeoutError("WebSocket send timed out")
                            )
                        raise
                    except Exception as exc:
                        self._stats.dropped += 1
                        if queued.timing_future is not None and not queued.timing_future.done():
                            queued.timing_future.set_exception(exc)
                        raise

            except Exception as e:
                if self._close_event.is_set():
                    break

                logger.info("[transport] disconnected: %s", e)
                self._stats.reconnects += 1

                # Flush stale packets
                while not self._queue.empty():
                    try:
                        queued = self._queue.get_nowait()
                        if queued.timing_future is not None and not queued.timing_future.done():
                            queued.timing_future.set_exception(
                                RuntimeError("WebSocket disconnected before send")
                            )
                        self._stats.dropped += 1
                    except asyncio.QueueEmpty:
                        break

                # Reconnect with exponential backoff
                logger.info("[transport] reconnecting in %.1fs", backoff)
                try:
                    await asyncio.wait_for(self._close_event.wait(), timeout=backoff)
                    break  # Close event was set during backoff
                except asyncio.TimeoutError:
                    pass

                backoff = min(backoff * 2, self.config.reconnect_cap_s)

            finally:
                if ws is not None:
                    with contextlib.suppress(Exception):
                        await ws.close()
                    ws = None
