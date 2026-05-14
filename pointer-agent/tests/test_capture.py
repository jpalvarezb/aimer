from __future__ import annotations

import pointer_agent.capture.macos as macos_capture
from aimer_core import BoundingBox, CursorPosition, FocusWindow, HoverRegion, SemanticContext
from pointer_agent.capture.macos import MacOSCaptureProvider


class FakeSettleDetector:
    def __init__(self, settled: bool) -> None:
        self._settled = settled

    def update(self, _cursor: CursorPosition) -> bool:
        return self._settled


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
    monkeypatch.setattr(macos_capture, "capture_hover_region", lambda _cursor, _scale: None)
    monkeypatch.setattr(macos_capture, "_display_scale_for_screen", lambda _screen_id: 2.0)

    packet = MacOSCaptureProvider().capture()

    assert packet.cursor.x == 10
    assert packet.cursor.y == 20
    assert packet.display_scale == 2.0
    assert packet.focus_window.app == "Code"
    assert packet.semantic.selected_text == "selected"
    assert packet.hover_region is None


def test_macos_capture_provider_includes_settled_hover_region(monkeypatch) -> None:
    monkeypatch.setattr(
        macos_capture,
        "capture_cursor",
        lambda: CursorPosition(x=10, y=20, screen_id=1),
    )
    monkeypatch.setattr(macos_capture, "capture_focus_window", lambda: FocusWindow(app="Code"))
    monkeypatch.setattr(macos_capture, "capture_semantic_context", lambda: SemanticContext())
    monkeypatch.setattr(macos_capture, "_display_scale_for_screen", lambda _screen_id: 2.0)
    monkeypatch.setattr(
        macos_capture,
        "capture_hover_region",
        lambda _cursor, _scale: HoverRegion(
            type="unknown",
            bbox=BoundingBox(x=0, y=0, width=256, height=256),
            tile_b64="abc123",
        ),
    )

    provider = MacOSCaptureProvider()
    provider._settle_detector = FakeSettleDetector(settled=True)
    packet = provider.capture()

    assert packet.hover_region is not None
    assert packet.hover_region.tile_b64 == "abc123"


def test_macos_capture_provider_skips_hover_region_when_unsettled(monkeypatch) -> None:
    monkeypatch.setattr(
        macos_capture,
        "capture_cursor",
        lambda: CursorPosition(x=10, y=20, screen_id=1),
    )
    monkeypatch.setattr(macos_capture, "capture_focus_window", lambda: FocusWindow())
    monkeypatch.setattr(macos_capture, "capture_semantic_context", lambda: SemanticContext())
    monkeypatch.setattr(macos_capture, "_display_scale_for_screen", lambda _screen_id: 2.0)

    def fail_capture(_cursor: CursorPosition, _scale: float) -> HoverRegion:
        raise AssertionError("capture_hover_region should not run for unsettled cursor")

    monkeypatch.setattr(macos_capture, "capture_hover_region", fail_capture)
    provider = MacOSCaptureProvider()
    provider._settle_detector = FakeSettleDetector(settled=False)

    packet = provider.capture()

    assert packet.hover_region is None


def test_macos_capture_provider_skips_hover_region_when_tiles_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        macos_capture,
        "capture_cursor",
        lambda: CursorPosition(x=10, y=20, screen_id=1),
    )
    monkeypatch.setattr(macos_capture, "capture_focus_window", lambda: FocusWindow())
    monkeypatch.setattr(macos_capture, "capture_semantic_context", lambda: SemanticContext())
    monkeypatch.setattr(macos_capture, "_display_scale_for_screen", lambda _screen_id: 2.0)

    def fail_capture(_cursor: CursorPosition, _scale: float) -> HoverRegion:
        raise AssertionError("capture_hover_region should not run when tiles are disabled")

    monkeypatch.setattr(macos_capture, "capture_hover_region", fail_capture)
    provider = MacOSCaptureProvider(tiles_enabled=False)
    provider._settle_detector = FakeSettleDetector(settled=True)

    packet = provider.capture()

    assert packet.hover_region is None
