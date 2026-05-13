"""Telemetry loop for emitting Aimer context packets."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from aimer_core import ContextPacket

from pointer_agent.capture.base import CaptureProvider

PacketSink = Callable[[ContextPacket], Awaitable[None]]


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
) -> None:
    """Run the capture loop at a fixed cadence until cancelled or limited."""

    if interval_hz <= 0:
        raise ValueError("interval_hz must be greater than zero")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    interval = 1.0 / interval_hz
    count = 0

    while limit is None or count < limit:
        started_at = time.perf_counter()
        packet = provider.capture()
        await sink(packet)
        count += 1

        elapsed = time.perf_counter() - started_at
        await asyncio.sleep(max(0.0, interval - elapsed))


def run_blocking(
    provider: CaptureProvider,
    *,
    interval_hz: float = 10.0,
    sink: PacketSink = stdout_sink,
    limit: int | None = None,
) -> int:
    """Synchronous wrapper used by the CLI."""

    try:
        asyncio.run(run_telemetry(provider, interval_hz=interval_hz, sink=sink, limit=limit))
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0
    except Exception as exc:
        print(f"pointer-agent failed: {exc}", file=sys.stderr)
        return 1
    return 0
