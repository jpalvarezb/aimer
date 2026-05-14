"""Command line entry point for duplex-bridge."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from duplex_bridge.providers.gemini_live import GeminiLiveSession
from duplex_bridge.server import WebSocketContextServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duplex-bridge",
        description="Aimer bridge from visual context packets to duplex model sessions.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="WebSocket server host. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port. Defaults to 8765.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash-exp",
        help="Gemini model name. Defaults to gemini-2.0-flash-exp (current Live-capable model).",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable for Gemini API key. Defaults to GEMINI_API_KEY.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    # Create Gemini Live session
    session = GeminiLiveSession(
        model=args.gemini_model,
        api_key_env=args.api_key_env,
    )

    # Create WebSocket server
    server = WebSocketContextServer(
        session=session,
        host=args.host,
        port=args.port,
        path="/context",
    )

    try:
        # Open session and start server
        await session.open()
        await server.start()

        print(
            f"[duplex-bridge] listening on ws://{args.host}:{args.port}/context, "
            f"gemini model={args.gemini_model}"
        )

        # Run until Ctrl+C
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("\n[duplex-bridge] shutting down")
    finally:
        # Clean shutdown
        await server.stop()
        await session.close()

    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
