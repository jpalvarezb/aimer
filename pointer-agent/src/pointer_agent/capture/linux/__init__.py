"""Linux capture provider placeholder."""

from __future__ import annotations

from jointer_core import ContextPacket

from pointer_agent.capture.base import CaptureProvider


class LinuxCaptureProvider(CaptureProvider):
    """Linux AT-SPI implementation placeholder."""

    def capture(self) -> ContextPacket:
        raise NotImplementedError("Linux capture is not implemented in Week 1.")
