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
        self._recv_task: asyncio.Task[None] | None = None
        self._open = False

        self._audio_callbacks: list[AudioOutCallback] = []
        self._tool_callbacks: list[ToolCallCallback] = []

    async def open(self) -> None:
        """Open the Gemini Live session and start receiving."""
        if self._open:
            raise RuntimeError("GeminiLiveSession is already open")

        # Check API key
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")

        # Build client
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )

        # Build config
        config = types.LiveConnectConfig(
            response_modalities=self.response_modalities,
            system_instruction=types.Content(
                parts=[types.Part(text=_SYSTEM_INSTRUCTION)]
            ),
        )

        # Connect
        self._session_ctx = self._client.aio.live.connect(model=self.model, config=config)
        self._session = await self._session_ctx.__aenter__()

        # Start recv loop
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._open = True

        logger.info(f"[gemini] connected to {self.model}")

    async def send_audio(self, frames: bytes) -> None:
        """Send raw PCM audio frames to Gemini Live.

        Note: Microphone capture is Week 4. This method is functional but unused for now.
        """
        if not self._open:
            raise RuntimeError("Session is not open")

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

        # Send tile as inline image if present
        if (
            packet.hover_region
            and getattr(packet.hover_region, "tile_b64", None)
            and packet.hover_region.tile_b64
        ):
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

        # Cancel recv loop
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None

        # Close session
        if self._session_ctx is not None:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"[gemini] error closing session: {e}")
            self._session_ctx = None
            self._session = None

        logger.info("[gemini] closed")

    async def _recv_loop(self) -> None:
        """Background task that receives messages from Gemini Live and dispatches to callbacks."""
        try:
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

        except Exception as e:
            if self._open:
                logger.error(f"[gemini] recv loop error: {e}")
            # Exit loop on error; do NOT restart
