"""macOS capture provider."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from aimer_core import ContextPacket, FocusWindow, SemanticContext

from pointer_agent.capture.base import CaptureProvider
from pointer_agent.capture.macos.accessibility import capture_semantic_context
from pointer_agent.capture.macos.cursor import capture_cursor
from pointer_agent.capture.macos.screen import capture_hover_region
from pointer_agent.capture.macos.window import capture_focus_window

T = TypeVar("T")

logger = logging.getLogger(__name__)
_scale_lookup_debugged_once = False


class MacOSCaptureProvider(CaptureProvider):
    """Capture Aimer context from macOS cursor, window, and AX APIs."""

    def capture(self) -> ContextPacket:
        cursor = capture_cursor()
        display_scale = _display_scale_for_screen(cursor.screen_id)
        focus_window = _safe_capture(capture_focus_window, FocusWindow())
        semantic = _safe_capture(capture_semantic_context, SemanticContext())
        hover_region = _safe_capture(lambda: capture_hover_region(cursor), None)

        return ContextPacket(
            cursor=cursor,
            display_scale=display_scale,
            focus_window=focus_window,
            hover_region=hover_region,
            semantic=semantic,
        )


def _safe_capture(capture: Callable[[], T], default: T) -> T:
    try:
        return capture()
    except Exception:
        return default


def _display_scale_for_screen(screen_id: int) -> float:
    try:
        from AppKit import NSScreen

        screens = NSScreen.screens()
        for screen in screens:
            if int(screen.deviceDescription()["NSScreenNumber"]) == screen_id:
                return float(screen.backingScaleFactor())
    except Exception:
        _debug_scale_lookup_once("Unable to resolve display scale; defaulting to 1.0")
        return 1.0

    _debug_scale_lookup_once("No matching screen for cursor screen_id; defaulting to 1.0")
    return 1.0


def _debug_scale_lookup_once(message: str) -> None:
    global _scale_lookup_debugged_once

    if not _scale_lookup_debugged_once:
        logger.debug(message)
        _scale_lookup_debugged_once = True
