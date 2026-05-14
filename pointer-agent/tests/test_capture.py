from __future__ import annotations

import pointer_agent.capture.macos as macos_capture
from aimer_core import CursorPosition, FocusWindow, SemanticContext
from pointer_agent.capture.macos import MacOSCaptureProvider


def test_macos_capture_provider_assembles_context_packet(monkeypatch) -> None:
    monkeypatch.setattr(
        macos_capture,
        "capture_cursor",
        lambda: CursorPosition(x=10, y=20, screen_id=1),
    )
    monkeypatch.setattr(
        macos_capture,
        "capture_focus_window",
        lambda: FocusWindow(app="Code", title="main.py", url=None),
    )
    monkeypatch.setattr(
        macos_capture,
        "capture_semantic_context",
        lambda: SemanticContext(accessibility_label="editor", selected_text="selected"),
    )
    monkeypatch.setattr(macos_capture, "capture_hover_region", lambda _cursor: None)
    monkeypatch.setattr(macos_capture, "_display_scale_for_screen", lambda _screen_id: 2.0)

    packet = MacOSCaptureProvider().capture()

    assert packet.cursor.x == 10
    assert packet.cursor.y == 20
    assert packet.display_scale == 2.0
    assert packet.focus_window.app == "Code"
    assert packet.semantic.selected_text == "selected"
    assert packet.hover_region is None
