"""Provider-neutral duplex model session interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from jointer_core import ContextPacket

AudioOutCallback = Callable[[bytes], Awaitable[None] | None]
ToolCall = Mapping[str, Any]
ToolCallCallback = Callable[[ToolCall], Awaitable[None] | None]


class DuplexSession(ABC):
    """Anti-lock-in boundary for Gemini Live, Realtime, Moshi, or TML sessions."""

    @abstractmethod
    async def open(self) -> None:
        """Open the underlying bidirectional model session."""

    @abstractmethod
    async def send_audio(self, frames: bytes) -> None:
        """Send raw PCM audio frames to the duplex model."""

    @abstractmethod
    async def send_visual_context(self, packet: ContextPacket) -> None:
        """Send one visual/deictic context packet to the duplex model."""

    @abstractmethod
    def on_audio_out(self, callback: AudioOutCallback) -> None:
        """Register a callback for streaming audio output."""

    @abstractmethod
    def on_tool_call(self, callback: ToolCallCallback) -> None:
        """Register a callback for model-emitted tool calls."""

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying model session."""
