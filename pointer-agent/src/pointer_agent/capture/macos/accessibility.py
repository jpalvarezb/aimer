"""macOS Accessibility API capture."""

from __future__ import annotations

from typing import Any

from aimer_core import SemanticContext


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _copy_ax_attribute(element: Any, attribute: str) -> Any | None:
    import ApplicationServices

    try:
        error, value = ApplicationServices.AXUIElementCopyAttributeValue(element, attribute, None)
    except Exception:
        return None
    if error != ApplicationServices.kAXErrorSuccess:
        return None
    return value


def is_accessibility_trusted() -> bool:
    """Return whether the current process has macOS Accessibility permission."""

    import ApplicationServices

    try:
        return bool(ApplicationServices.AXIsProcessTrusted())
    except Exception:
        return False


def capture_semantic_context() -> SemanticContext:
    """Return selected text and labels from the focused accessibility element."""

    import ApplicationServices

    if not is_accessibility_trusted():
        return SemanticContext()

    system = ApplicationServices.AXUIElementCreateSystemWide()
    focused = _copy_ax_attribute(system, ApplicationServices.kAXFocusedUIElementAttribute)
    if focused is None:
        return SemanticContext()

    selected_text = _stringify(
        _copy_ax_attribute(focused, ApplicationServices.kAXSelectedTextAttribute)
    )
    accessibility_label = _first_present(
        _copy_ax_attribute(focused, ApplicationServices.kAXDescriptionAttribute),
        _copy_ax_attribute(focused, ApplicationServices.kAXTitleAttribute),
        _copy_ax_attribute(focused, ApplicationServices.kAXRoleDescriptionAttribute),
        _copy_ax_attribute(focused, ApplicationServices.kAXValueAttribute),
    )
    dom_identifier_attr = getattr(
        ApplicationServices,
        "kAXDOMIdentifierAttribute",
        "AXDOMIdentifier",
    )
    dom_path = _stringify(_copy_ax_attribute(focused, dom_identifier_attr))

    return SemanticContext(
        accessibility_label=accessibility_label,
        selected_text=selected_text,
        dom_path=dom_path,
        language=None,
    )


def _first_present(*values: Any) -> str | None:
    for value in values:
        text = _stringify(value)
        if text:
            return text
    return None
