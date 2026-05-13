"""Gemini Live DuplexSession placeholder.

The Jointer brief picks Gemini Live for the pragmatic v1 duplex model. This
adapter is intentionally a Week 3 stub; Week 1 only establishes the provider
interface and shared context packet dependency.
"""

from __future__ import annotations

from jointer_core import ContextPacket

from duplex_bridge.session import AudioOutCallback, DuplexSession, ToolCallCallback


class GeminiLiveSession(DuplexSession):
    """Week 3 adapter for the Gemini Live API."""

    async def open(self) -> None:
        raise NotImplementedError("Gemini Live integration is planned for Week 3.")

    async def send_audio(self, _frames: bytes) -> None:
        raise NotImplementedError("Gemini Live integration is planned for Week 3.")

    async def send_visual_context(self, _packet: ContextPacket) -> None:
        raise NotImplementedError("Gemini Live integration is planned for Week 3.")

    def on_audio_out(self, _callback: AudioOutCallback) -> None:
        raise NotImplementedError("Gemini Live integration is planned for Week 3.")

    def on_tool_call(self, _callback: ToolCallCallback) -> None:
        raise NotImplementedError("Gemini Live integration is planned for Week 3.")

    async def close(self) -> None:
        raise NotImplementedError("Gemini Live integration is planned for Week 3.")
