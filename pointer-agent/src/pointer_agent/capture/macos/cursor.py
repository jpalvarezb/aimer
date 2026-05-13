"""macOS cursor capture."""

from __future__ import annotations

from aimer_core import CursorPosition


def capture_cursor() -> CursorPosition:
    """Return the current cursor position in global screen coordinates."""

    import Quartz

    event = Quartz.CGEventCreate(None)
    point = Quartz.CGEventGetLocation(event)
    screen_id = int(Quartz.CGMainDisplayID())
    return CursorPosition(x=float(point.x), y=float(point.y), screen_id=screen_id)
