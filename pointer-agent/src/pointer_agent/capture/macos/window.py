"""macOS focused-window capture."""

from __future__ import annotations

from typing import Any

from aimer_core import FocusWindow

_LAYER_NORMAL = 0


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "absoluteString"):
        return str(value.absoluteString())
    return str(value)


def _copy_ax_attribute(element: Any, attribute: str) -> Any | None:
    import ApplicationServices

    try:
        error, value = ApplicationServices.AXUIElementCopyAttributeValue(element, attribute, None)
    except Exception:
        return None
    if error != ApplicationServices.kAXErrorSuccess:
        return None
    return value


def capture_focus_window() -> FocusWindow:
    """Return best-effort metadata for the frontmost application window."""

    import ApplicationServices
    from AppKit import NSWorkspace

    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return FocusWindow()

    app_name = _stringify(app.localizedName())
    pid = app.processIdentifier()
    app_element = ApplicationServices.AXUIElementCreateApplication(pid)
    window = _copy_ax_attribute(app_element, ApplicationServices.kAXFocusedWindowAttribute)

    title = None
    url = None
    if window is not None:
        title = _stringify(_copy_ax_attribute(window, ApplicationServices.kAXTitleAttribute))
        url_attr = getattr(ApplicationServices, "kAXURLAttribute", "AXURL")
        url = _stringify(_copy_ax_attribute(window, url_attr))

    return FocusWindow(app=app_name, title=title, url=url)


def _owner_at_point(windows: list[dict[str, Any]], x: float, y: float) -> str | None:
    """Pure hit-test over a front-to-back window list (no Quartz needed — unit-testable).

    Skips non-layer-0 windows (menu bar, overlays) and returns the owner name of the first
    (frontmost) window whose bounds contain the point, or None if no window matches.
    """
    for window in windows:
        if window.get("kCGWindowLayer") != _LAYER_NORMAL:
            continue
        bounds = window.get("kCGWindowBounds")
        if not bounds:
            continue
        try:
            bx, by = float(bounds["X"]), float(bounds["Y"])
            bw, bh = float(bounds["Width"]), float(bounds["Height"])
        except (KeyError, TypeError, ValueError):
            continue
        if bx <= x <= bx + bw and by <= y <= by + bh:
            owner = window.get("kCGWindowOwnerName")
            return str(owner) if owner is not None else None
    return None


def capture_app_under_cursor(x: float, y: float) -> str | None:
    """Return the owner app name of the frontmost on-screen window under the cursor.

    None on any failure (never raises into the capture loop) or if no window matches.
    """
    try:
        import Quartz

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        return _owner_at_point(list(windows), x, y)
    except Exception:
        return None
