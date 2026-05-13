"""Windows capture provider placeholder."""

from __future__ import annotations

from aimer_core import ContextPacket

from pointer_agent.capture.base import CaptureProvider


class WindowsCaptureProvider(CaptureProvider):
    """Windows UI Automation implementation placeholder."""

    def capture(self) -> ContextPacket:
        raise NotImplementedError("Windows capture is not implemented in Week 1.")
