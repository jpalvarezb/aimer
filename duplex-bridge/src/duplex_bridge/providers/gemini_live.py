"""Gemini Live DuplexSession implementation.

The Aimer brief picks Gemini Live for the pragmatic v1 duplex model. This adapter
uses google-genai to connect to Gemini's multimodal Live API.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any

from aimer_core import ContextPacket
from google import genai
from google.genai import types

from duplex_bridge.session import AudioOutCallback, DuplexSession, ToolCallCallback

logger = logging.getLogger(__name__)

# System instruction explaining Aimer's role
_SYSTEM_INSTRUCTION = (
    "You are Aimer, a pointer-grounded assistant. The user points at things on screen "
    "and speaks. You receive cursor position, window context, selected text, and screen tiles. "
    "Respond naturally and concisely."
)

_INITIAL_CONNECT_TIMEOUT_S = 5.0
_RECONNECT_INITIAL_S = 0.5
_RECONNECT_CAP_S = 4.0


@dataclass
class _Stats:
    """Internal session statistics for debugging."""

    reconnects: int = 0
    dropped_during_reconnect: int = 0


class GeminiLiveSession(DuplexSession):
    """Adapter for the Gemini Live API.

    Uses google-genai to establish a bidirectional session with visual context and
    audio streams. Audio output is 24 kHz PCM by default.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "GEMINI_API_KEY",
        response_modalities: list[str] | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.response_modalities = response_modalities or ["AUDIO", "TEXT"]

        self._client: genai.Client | None = None
        self._session: Any = None
        self._session_ctx: Any = None
        self._session_task: asyncio.Task[None] | None = None
        self._open = False
        self._connected = False
        self._close_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._api_key: str | None = None
        self._stats = _Stats()

        self._audio_callbacks: list[AudioOutCallback] = []
        self._tool_callbacks: list[ToolCallCallback] = []

    @property
    def stats(self) -> dict[str, int]:
        """Return current session statistics for debugging."""
        return {
            "reconnects": self._stats.reconnects,
            "dropped_during_reconnect": self._stats.dropped_during_reconnect,
        }

    async def open(self) -> None:
        """Open the Gemini Live session and start receiving."""
        if self._open:
            raise RuntimeError("GeminiLiveSession is already open")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        self._api_key = api_key

        self._close_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._connected = False
        self._open = True
        self._session_task = asyncio.create_task(self._session_loop())

        try:
            await asyncio.wait_for(
                self._connected_event.wait(), timeout=_INITIAL_CONNECT_TIMEOUT_S
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError("failed to connect to Gemini within 5s") from exc

    async def _session_loop(self) -> None:
        """Connect to Gemini Live, receive messages, and reconnect on failure.

        Reconnects create a fresh Live session. The system instruction is resent via
        LiveConnectConfig, but in-flight visual context is intentionally lost.
        """
        backoff = _RECONNECT_INITIAL_S

        while not self._close_event.is_set():
            try:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(api_version="v1beta"),
                )
                config = types.LiveConnectConfig(
                    response_modalities=self.response_modalities,
                    system_instruction=types.Content(
                        parts=[types.Part(text=_SYSTEM_INSTRUCTION)]
                    ),
                )

                self._session_ctx = self._client.aio.live.connect(
                    model=self.model, config=config
                )
                self._session = await self._session_ctx.__aenter__()
                self._connected = True
                self._connected_event.set()
                backoff = _RECONNECT_INITIAL_S

                logger.info("[gemini] connected to %s", self.model)
                await self._recv_loop()

                if not self._close_event.is_set():
                    raise RuntimeError("Gemini receive loop ended")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._connected = False
                if self._close_event.is_set():
                    break

                logger.warning("[gemini] session disconnected: %s", e)
                self._stats.reconnects += 1

                logger.info("[gemini] reconnecting in %.1fs", backoff)
                try:
                    await asyncio.wait_for(
                        self._close_event.wait(), timeout=backoff
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                backoff = min(backoff * 2, _RECONNECT_CAP_S)
            finally:
                self._connected = False
                if self._session_ctx is not None:
                    try:
                        await self._session_ctx.__aexit__(None, None, None)
                    except Exception as e:
                        logger.warning("[gemini] error closing session: %s", e)
                    self._session_ctx = None
                    self._session = None

                if self._client is not None:
                    with contextlib.suppress(Exception):
                        self._client.close()
                    self._client = None

        self._connected = False

    def _drop_if_reconnecting(self) -> bool:
        """Return True when a send should be dropped during reconnect."""
        if self._connected and self._session is not None:
            return False
        self._stats.dropped_during_reconnect += 1
        return True

    async def send_audio(self, frames: bytes) -> None:
        """Send raw PCM audio frames to Gemini Live.

        Note: Microphone capture is Week 4. This method is functional but unused for now.
        """
        if not self._open:
            raise RuntimeError("Session is not open")
        if self._drop_if_reconnecting():
            return

        await self._session.send_realtime_input(
            audio=types.Blob(mime_type="audio/pcm;rate=16000", data=frames)
        )

    async def send_visual_context(self, packet: ContextPacket) -> None:
        """Send visual context from a ContextPacket to Gemini Live.

        Sends the screen tile as an inline JPEG image (if present) and a concise
        text annotation with cursor, window, and selected text context.
        """
        if not self._open:
            raise RuntimeError("Session is not open")
        if self._drop_if_reconnecting():
            return

        # Send tile as inline image if present
        if packet.hover_region and packet.hover_region.tile_b64:
            tile_bytes = base64.b64decode(packet.hover_region.tile_b64)
            await self._session.send_realtime_input(
                media=types.Blob(mime_type="image/jpeg", data=tile_bytes)
            )

        # Build text annotation
        parts = ["[context]"]

        if packet.focus_window:
            app = packet.focus_window.app
            if app:
                parts.append(f"app={app}")
            title = packet.focus_window.title
            if title:
                parts.append(f"title={title}")

        parts.append(f"cursor=({packet.cursor.x:.0f},{packet.cursor.y:.0f})")

        if packet.semantic:
            selected = packet.semantic.selected_text
            if selected:
                parts.append(f"selected={selected[:80]}")

        text_annotation = " ".join(parts)
        await self._session.send_realtime_input(text=text_annotation)

    def on_audio_out(self, callback: AudioOutCallback) -> None:
        """Register a callback for streaming audio output (24 kHz PCM)."""
        self._audio_callbacks.append(callback)

    def on_tool_call(self, callback: ToolCallCallback) -> None:
        """Register a callback for model-emitted tool calls."""
        self._tool_callbacks.append(callback)

    async def close(self) -> None:
        """Close the Gemini Live session."""
        if not self._open:
            return  # Idempotent

        self._open = False
        self._connected = False
        self._close_event.set()

        if self._session_task is not None:
            self._session_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._session_task
            self._session_task = None

        logger.info("[gemini] closed")

    async def _recv_loop(self) -> None:
        """Background task that receives messages from Gemini Live and dispatches to callbacks."""
        async for message in self._session.receive():
            # Dispatch audio output
            if hasattr(message, "data") and message.data:
                audio_data: bytes = message.data
                for callback in self._audio_callbacks:
                    audio_result = callback(audio_data)
                    if asyncio.iscoroutine(audio_result):
                        await audio_result

            # Dispatch tool calls
            if hasattr(message, "tool_call") and message.tool_call:
                tool_call: Any = message.tool_call
                for tool_callback in self._tool_callbacks:
                    tool_result = tool_callback(tool_call)
                    if asyncio.iscoroutine(tool_result):
                        await tool_result
