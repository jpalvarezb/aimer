"""macOS capture provider."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from aimer_core import ContextPacket, FocusWindow, FullFrame, SemanticContext

from pointer_agent.capture.base import CaptureProvider
from pointer_agent.capture.debounce import CursorSettleDetector
from pointer_agent.capture.macos.accessibility import capture_semantic_context
from pointer_agent.capture.macos.cursor import capture_cursor
from pointer_agent.capture.macos.screen import capture_full_frame, capture_hover_region
from pointer_agent.capture.macos.window import capture_focus_window

T = TypeVar("T")

logger = logging.getLogger(__name__)
_scale_lookup_debugged_once = False

_FULL_FRAME_MIN_INTERVAL_S: float = 2.0


class MacOSCaptureProvider(CaptureProvider):
    """Capture Aimer context from macOS cursor, window, and AX APIs."""

    def __init__(self, *, tiles_enabled: bool = True) -> None:
        self._tiles_enabled = tiles_enabled
        self._settle_detector = CursorSettleDetector()
        self._last_full_frame_t: float = 0.0
        self._cached_full_frame: FullFrame | None = None

    def capture(self) -> ContextPacket:
        cursor = capture_cursor()
        display_scale = _display_scale_for_screen(cursor.screen_id)
        focus_window = _safe_capture(capture_focus_window, FocusWindow())
        semantic = _safe_capture(capture_semantic_context, SemanticContext())

        new_full_frame: FullFrame | None = None
        if self._tiles_enabled and self._settle_detector.update(cursor):
            hover_region = _safe_capture(lambda: capture_hover_region(cursor, display_scale), None)
            now = time.monotonic()
            if now - self._last_full_frame_t >= _FULL_FRAME_MIN_INTERVAL_S:
                new_full_frame = _safe_capture(
                    lambda: capture_full_frame(cursor, display_scale), None
                )
                if new_full_frame is not None:
                    self._cached_full_frame = new_full_frame
                    self._last_full_frame_t = now
        else:
            hover_region = None

        return ContextPacket(
            cursor=cursor,
            display_scale=display_scale,
            focus_window=focus_window,
            hover_region=hover_region,
            full_frame=new_full_frame,
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
