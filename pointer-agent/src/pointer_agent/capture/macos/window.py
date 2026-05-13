"""macOS focused-window capture."""

from __future__ import annotations

from typing import Any

from jointer_core import FocusWindow


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
