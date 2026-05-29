"""Telemetry loop for emitting Aimer context packets."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aimer_core import ContextPacket

from pointer_agent.capture.base import CaptureProvider

PacketSink = Callable[[ContextPacket], Awaitable[None]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencySample:
    """One tile-to-wire timing sample in milliseconds."""

    capture_ms: float
    jpeg_encode_ms: float | None
    base64_ms: float | None
    packet_build_ms: float | None
    ws_send_ms: float | None
    total_tile_to_wire_ms: float


@dataclass
class _LatencyWindow:
    """Accumulate and periodically log tile-to-wire latency percentiles."""

    log_interval_s: float
    samples: list[LatencySample] = field(default_factory=list)
    last_log_at: float = field(default_factory=time.perf_counter)

    def add(self, sample: LatencySample) -> None:
        self.samples.append(sample)

    def maybe_log(self) -> None:
        now = time.perf_counter()
        if not self.samples or (now - self.last_log_at) < self.log_interval_s:
            return

        logger.info(
            "[latency] tile-to-wire %s",
            " ".join(
                [
                    _format_percentiles("total", [s.total_tile_to_wire_ms for s in self.samples]),
                    _format_percentiles("capture", [s.capture_ms for s in self.samples]),
                    _format_percentiles(
                        "jpeg",
                        [s.jpeg_encode_ms for s in self.samples if s.jpeg_encode_ms is not None],
                    ),
                    _format_percentiles(
                        "base64", [s.base64_ms for s in self.samples if s.base64_ms is not None]
                    ),
                    _format_percentiles(
                        "packet_build",
                        [s.packet_build_ms for s in self.samples if s.packet_build_ms is not None],
                    ),
                    _format_percentiles(
                        "ws_send", [s.ws_send_ms for s in self.samples if s.ws_send_ms is not None]
                    ),
                ]
            ),
        )
        self.samples.clear()
        self.last_log_at = now


async def stdout_sink(packet: ContextPacket) -> None:
    """Write a packet to stdout as newline-delimited JSON."""

    print(packet.model_dump_json(), flush=True)


class JsonlFileSink:
    """Append packets to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    async def __call__(self, packet: ContextPacket) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as output:
            output.write(packet.model_dump_json())
            output.write("\n")


async def run_telemetry(
    provider: CaptureProvider,
    *,
    interval_hz: float = 10.0,
    sink: PacketSink = stdout_sink,
    limit: int | None = None,
    log_latency: bool = False,
    latency_log_interval_s: float = 10.0,
) -> None:
    """Run the capture loop at a fixed cadence until cancelled or limited."""

    if interval_hz <= 0:
        raise ValueError("interval_hz must be greater than zero")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if latency_log_interval_s <= 0:
        raise ValueError("latency_log_interval_s must be greater than zero")

    interval = 1.0 / interval_hz
    count = 0
    latency_window = _LatencyWindow(log_interval_s=latency_log_interval_s) if log_latency else None

    while limit is None or count < limit:
        started_at = time.perf_counter()
        _reset_hover_region_timing()
        capture_started_at = time.perf_counter()
        packet = provider.capture()
        captured_at = time.perf_counter()
        send_timing = await _send_packet(sink, packet, log_latency=log_latency)
        sent_at = time.perf_counter()
        count += 1

        if latency_window is not None and _packet_has_tile(packet):
            hover_timing = _hover_region_timing()
            latency_window.add(
                LatencySample(
                    capture_ms=(captured_at - capture_started_at) * 1000.0,
                    jpeg_encode_ms=getattr(hover_timing, "jpeg_encode_ms", None),
                    base64_ms=getattr(hover_timing, "base64_ms", None),
                    packet_build_ms=getattr(send_timing, "packet_build_ms", None),
                    ws_send_ms=getattr(send_timing, "ws_send_ms", None),
                    total_tile_to_wire_ms=(sent_at - started_at) * 1000.0,
                )
            )
            latency_window.maybe_log()

        elapsed = sent_at - started_at
        await asyncio.sleep(max(0.0, interval - elapsed))


def run_blocking(
    provider: CaptureProvider,
    *,
    interval_hz: float = 10.0,
    sink: PacketSink = stdout_sink,
    limit: int | None = None,
    log_latency: bool = False,
    latency_log_interval_s: float = 10.0,
) -> int:
    """Synchronous wrapper used by the CLI."""

    try:
        asyncio.run(
            run_telemetry(
                provider,
                interval_hz=interval_hz,
                sink=sink,
                limit=limit,
                log_latency=log_latency,
                latency_log_interval_s=latency_log_interval_s,
            )
        )
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0
    except Exception as exc:
        print(f"pointer-agent failed: {exc}", file=sys.stderr)
        return 1
    return 0


async def _send_packet(
    sink: PacketSink,
    packet: ContextPacket,
    *,
    log_latency: bool,
) -> Any:
    if log_latency:
        send_with_latency = getattr(sink, "send_with_latency", None)
        if send_with_latency is not None:
            return await send_with_latency(packet)

    await sink(packet)
    return None


def _packet_has_tile(packet: ContextPacket) -> bool:
    return bool(packet.hover_region and packet.hover_region.tile_b64)


def _reset_hover_region_timing() -> None:
    try:
        from pointer_agent.capture.macos import screen
    except Exception:
        return
    screen.reset_last_capture_timing()


def _hover_region_timing() -> Any:
    try:
        from pointer_agent.capture.macos import screen
    except Exception:
        return None
    return screen.get_last_capture_timing()


def _format_percentiles(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}=n/a"
    sorted_values = sorted(values)
    return (
        f"{name}_p50={_percentile(sorted_values, 50):.1f}ms "
        f"{name}_p95={_percentile(sorted_values, 95):.1f}ms"
    )


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = round((percentile / 100.0) * (len(sorted_values) - 1))
    return sorted_values[index]
