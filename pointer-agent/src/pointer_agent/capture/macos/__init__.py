"""macOS capture provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from aimer_core import ContextPacket, FocusWindow, SemanticContext

from pointer_agent.capture.base import CaptureProvider
from pointer_agent.capture.macos.accessibility import capture_semantic_context
from pointer_agent.capture.macos.cursor import capture_cursor
from pointer_agent.capture.macos.screen import capture_hover_region
from pointer_agent.capture.macos.window import capture_focus_window

T = TypeVar("T")


class MacOSCaptureProvider(CaptureProvider):
    """Capture Aimer context from macOS cursor, window, and AX APIs."""

    def capture(self) -> ContextPacket:
        cursor = capture_cursor()
        focus_window = _safe_capture(capture_focus_window, FocusWindow())
        semantic = _safe_capture(capture_semantic_context, SemanticContext())
        hover_region = _safe_capture(lambda: capture_hover_region(cursor), None)

        return ContextPacket(
            cursor=cursor,
            focus_window=focus_window,
            hover_region=hover_region,
            semantic=semantic,
        )


def _safe_capture(capture: Callable[[], T], default: T) -> T:
    try:
        return capture()
    except Exception:
        return default
