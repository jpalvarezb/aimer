"""Provider-neutral duplex model session interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from aimer_core import ContextPacket

AudioOutCallback = Callable[[bytes], Awaitable[None] | None]
TextOutCallback = Callable[[str], Awaitable[None] | None]
ToolCall = Mapping[str, Any]
ToolCallCallback = Callable[[ToolCall], Awaitable[None] | None]
InterruptCallback = Callable[[], Awaitable[None] | None]
ToolCancellationCallback = Callable[[list[str]], Awaitable[None] | None]
TurnCompleteCallback = Callable[[], None]


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

    async def send_activity_start(self) -> None:  # noqa: B027
        """Signal the start of a user turn (manual VAD).

        Optional hook with a default no-op; providers that support client-driven
        turn detection override this. Callers may invoke it unconditionally.
        """

    async def send_activity_end(self) -> None:  # noqa: B027
        """Signal the end of a user turn (manual VAD).

        Optional hook with a default no-op; providers that support client-driven
        turn detection override this. Callers may invoke it unconditionally.
        """

    @abstractmethod
    def on_audio_out(self, callback: AudioOutCallback) -> None:
        """Register a callback for streaming audio output."""

    def on_interrupt(self, callback: InterruptCallback) -> None:  # noqa: B027
        """Register a callback fired when the model is interrupted (barge-in).

        Optional hook with a default no-op so providers without a native barge-in
        signal need not implement it. Providers that detect the user talking over
        the assistant invoke the registered callbacks so the audio backend can
        flush already-buffered playback (otherwise the assistant keeps talking for
        a beat after the user starts). Callers may register unconditionally.
        """

    def on_text_out(self, callback: TextOutCallback) -> None:  # noqa: B027
        """Register a callback for streaming text output from the model.

        Optional hook with a default no-op; not all providers or modalities emit
        text. When the model produces a text response, each text chunk is passed to
        the registered callbacks as a plain string. Needed by the deictic eval
        harness when running text-modality sessions. Callers may register
        unconditionally.
        """

    def on_turn_complete(self, callback: TurnCompleteCallback) -> None:  # noqa: B027
        """Register a callback fired when the model's current turn completes.

        Optional hook with a default no-op; not all providers expose an explicit
        end-of-turn signal. Providers that do (e.g. Gemini Live's
        ``turn_complete``, or the natural exhaustion of a one-turn receive
        stream) invoke the registered callbacks exactly once per turn so callers
        (e.g. the deictic eval harness) can stop waiting for a response promptly
        instead of relying on a fixed settle window. Callers may register
        unconditionally.
        """

    @abstractmethod
    def on_tool_call(self, callback: ToolCallCallback) -> None:
        """Register a callback for model-emitted tool calls."""

    async def send_tool_response(  # noqa: B027 — optional hook, default no-op
        self,
        *,
        name: str,
        call_id: str,
        response: Mapping[str, Any],
        is_error: bool = False,
        final: bool = True,
    ) -> None:
        """Deliver a tool result back to the model, correlated by ``call_id``.

        ``final=False`` sends a silent progress acknowledgement (more responses will
        follow for the same ``call_id``) — used to ACK long-running tools immediately
        so the model can keep conversing while the work runs. Default no-op so
        providers without a tool-response channel need not implement it; results are
        then simply not spoken.
        """

    def on_tool_call_cancellation(self, callback: ToolCancellationCallback) -> None:  # noqa: B027
        """Register a callback fired when the model cancels in-flight tool calls.

        Optional hook with a default no-op; providers with a native cancellation
        signal (Gemini's ``tool_call_cancellation``) invoke the callbacks with the
        cancelled function-call ids so dispatched work can be aborted.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying model session."""
