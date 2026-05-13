"""Command line entry point for pointer-agent."""

from __future__ import annotations

import argparse
import os

from pointer_agent.capture import PlatformCaptureProvider
from pointer_agent.telemetry import JsonlFileSink, run_blocking, stdout_sink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pointer-agent",
        description="Emit Jointer pointer context packets as newline-delimited JSON.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=float(os.environ.get("JOINTER_TELEMETRY_HZ", "10")),
        help="Telemetry capture frequency. Defaults to 10 Hz.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("JOINTER_TELEMETRY_OUTPUT") or None,
        help="Optional JSONL output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional packet count for tests or one-shot diagnostics.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sink = JsonlFileSink(args.output) if args.output else stdout_sink
    return run_blocking(
        PlatformCaptureProvider(),
        interval_hz=args.hz,
        sink=sink,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
