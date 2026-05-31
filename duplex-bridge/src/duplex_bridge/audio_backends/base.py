"""Audio backend seam.

An ``AudioBackend`` owns the platform audio I/O: it captures the microphone and
emits cleaned 16 kHz mono int16 frames via an ``on_frame`` callback, and it owns
playback (subscribing to ``session.on_audio_out``). One backend owning *both*
streams is required for OS-level echo cancellation (macOS Voice Processing I/O),
where a single ``AVAudioEngine`` must drive capture and render to derive its echo
reference. The turn-detection / VAD / push-to-talk layer (``MicCapture``) sits
above this seam and is backend-agnostic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from duplex_bridge.session import DuplexSession

# Called (on the event loop) once per captured frame with raw PCM bytes.
OnFrame = Callable[[bytes], None]


class BackendName(str, Enum):
    """Selectable audio backends."""

    SOUNDDEVICE = "sounddevice"
    SOFTWARE_AEC = "software-aec"
    VPIO = "vpio"
    NATIVE_VPIO = "native-vpio"


@dataclass(frozen=True)
class CaptureFormat:
    """Format of the frames a backend emits to the pipeline."""

    sample_rate: int = 16_000
    channels: int = 1
    frame_samples: int = 1_600  # per emitted frame

    @property
    def frame_ms(self) -> float:
        return self.frame_samples / self.sample_rate * 1000.0


@runtime_checkable
class AudioBackend(Protocol):
    """Platform audio I/O behind the capture pipeline."""

    async def start(
        self,
        on_frame: OnFrame,
        session: DuplexSession,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Start capture + playback. Returns False if audio is unavailable.

        Implementations marshal ``on_frame`` onto ``loop`` (never call it from a
        realtime audio thread directly) and own the ``session.on_audio_out``
        subscription for playback.
        """
        ...

    async def stop(self) -> None:
        """Stop capture and playback."""
        ...

    @property
    def capture_config(self) -> CaptureFormat:
        """Format of frames delivered to ``on_frame``."""
        ...

    @property
    def stats(self) -> dict[str, int]:
        """Backend-level counters (device drops, underruns, ...)."""
        ...
