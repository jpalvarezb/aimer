"""WebSocket server for receiving visual context from pointer-agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aimer_core import ContextPacket
from pydantic import ValidationError

try:
    from websockets.asyncio.server import serve
except ImportError:
    from websockets import serve  # type: ignore[attr-defined]

from duplex_bridge.session import DuplexSession

logger = logging.getLogger(__name__)


class WebSocketContextServer:
    """WebSocket server that forwards visual context packets to a DuplexSession.

    Accepts one client at a time. If a second client connects, the older connection
    is closed with a policy_violation code. Parse errors are logged but do not crash
    the server. Session errors are logged and the server continues.

    Note: The session lifecycle is OWNED by the caller, not this server. The server
    will NOT call session.close().
    """

    def __init__(
        self,
        session: DuplexSession,
        host: str = "127.0.0.1",
        port: int = 8765,
        path: str = "/context",
    ) -> None:
        self.session = session
        self.host = host
        self.port = port
        self.path = path
        self._server: Any = None
        self._server_task: asyncio.Task[None] | None = None
        self._current_client: Any = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._server_task is not None:
            raise RuntimeError("WebSocketContextServer is already started")

        self._server_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server_task is None:
            return  # Idempotent

        self._stop_event.set()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._server_task is not None:
            try:
                await asyncio.wait_for(self._server_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
            self._server_task = None

    async def _run(self) -> None:
        """Internal server task."""
        self._server = await serve(self._handle_client, self.host, self.port)
        logger.info(f"[server] listening on ws://{self.host}:{self.port}{self.path}")
        await self._stop_event.wait()

    async def _handle_client(self, websocket: Any) -> None:
        """Handle incoming WebSocket connection."""
        # Single-client-at-a-time policy
        if self._current_client is not None:
            logger.info("[server] new client connected, closing old client")
            try:
                await self._current_client.close(code=1008, reason="Policy violation")
            except Exception:
                pass

        self._current_client = websocket

        try:
            async for message in websocket:
                try:
                    # Parse packet
                    packet = ContextPacket.model_validate_json(message)

                    # Forward to session
                    try:
                        await self.session.send_visual_context(packet)
                    except Exception as e:
                        logger.error(f"[server] session error: {e}")
                        # Continue - don't let session errors crash the server

                except ValidationError as e:
                    logger.warning(
                        f"[server] invalid packet (first 200 chars): {str(message)[:200]}"
                    )
                    logger.debug(f"[server] validation error: {e}")
                    # Continue - don't crash on bad packets

        except Exception as e:
            logger.debug(f"[server] client disconnected: {e}")
        finally:
            if self._current_client is websocket:
                self._current_client = None
