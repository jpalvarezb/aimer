"""Capture provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jointer_core import ContextPacket


class CaptureError(RuntimeError):
    """Raised when a capture provider cannot read platform context."""


class CaptureProvider(ABC):
    """Platform-specific source for one Jointer context packet."""

    @abstractmethod
    def capture(self) -> ContextPacket:
        """Capture the current cursor, window, visual, and semantic context."""
