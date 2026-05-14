"""Command line entry point for pointer-agent."""

from __future__ import annotations

import argparse
import asyncio
import os

from pointer_agent.capture import PlatformCaptureProvider
from pointer_agent.telemetry import JsonlFileSink, run_blocking, stdout_sink
from pointer_agent.transport import WebSocketPacketSink, WebSocketTransportConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pointer-agent",
        description="Emit Aimer pointer context packets as newline-delimited JSON.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=float(os.environ.get("AIMER_TELEMETRY_HZ", "10")),
        help="Telemetry capture frequency. Defaults to 10 Hz.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("AIMER_TELEMETRY_OUTPUT") or None,
        help="Optional JSONL output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--ws-url",
        default=None,
        help="Optional WebSocket URL for streaming to duplex-bridge.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional packet count for tests or one-shot diagnostics.",
    )
    parser.add_argument(
        "--tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cursor-settled screen tiles. Use --no-tiles to emit Week 1 context only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Validate mutually exclusive flags
    if args.output and args.ws_url:
        raise SystemExit("Error: --output and --ws-url are mutually exclusive")

    # Choose sink
    if args.ws_url:
        ws_sink = WebSocketPacketSink(WebSocketTransportConfig(url=args.ws_url))
        try:
            return run_blocking(
                PlatformCaptureProvider(tiles_enabled=args.tiles),  # type: ignore[call-arg]
                interval_hz=args.hz,
                sink=ws_sink,
                limit=args.limit,
            )
        finally:
            asyncio.run(ws_sink.close())
    else:
        sink = JsonlFileSink(args.output) if args.output else stdout_sink
        return run_blocking(
            PlatformCaptureProvider(tiles_enabled=args.tiles),  # type: ignore[call-arg]
            interval_hz=args.hz,
            sink=sink,
            limit=args.limit,
        )


if __name__ == "__main__":
    raise SystemExit(main())
