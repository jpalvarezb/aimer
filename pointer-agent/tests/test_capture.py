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


# ---------------------------------------------------------------------------
# Week-9 live-smoke fix A3: window-under-cursor hit-test (pure logic, no Quartz needed)
# ---------------------------------------------------------------------------


def _win(owner: str, x: float, y: float, w: float, h: float, layer: int = 0) -> dict:
    return {
        "kCGWindowOwnerName": owner,
        "kCGWindowLayer": layer,
        "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h},
    }


def test_owner_at_point_returns_owner_name_when_cursor_inside_bounds() -> None:
    from pointer_agent.capture.macos.window import _owner_at_point

    windows = [_win("Notion", 0, 0, 200, 200)]
    assert _owner_at_point(windows, 50, 50) == "Notion"


def test_owner_at_point_returns_none_when_cursor_outside_all_bounds() -> None:
    from pointer_agent.capture.macos.window import _owner_at_point

    windows = [_win("Notion", 0, 0, 200, 200)]
    assert _owner_at_point(windows, 500, 500) is None


def test_owner_at_point_skips_non_layer_zero_windows() -> None:
    from pointer_agent.capture.macos.window import _owner_at_point

    # A menu-bar/overlay window (non-zero layer) covers the cursor but must be skipped.
    windows = [
        _win("Menu Bar", 0, 0, 1000, 1000, layer=25),
        _win("Notion", 0, 0, 200, 200, layer=0),
    ]
    assert _owner_at_point(windows, 50, 50) == "Notion"


def test_owner_at_point_front_to_back_first_match_wins() -> None:
    from pointer_agent.capture.macos.window import _owner_at_point

    # Two overlapping layer-0 windows: the list order IS front-to-back — the first wins.
    windows = [
        _win("Finder", 0, 0, 300, 300),
        _win("Notion", 0, 0, 200, 200),
    ]
    assert _owner_at_point(windows, 50, 50) == "Finder"


def test_owner_at_point_empty_list_returns_none() -> None:
    from pointer_agent.capture.macos.window import _owner_at_point

    assert _owner_at_point([], 50, 50) is None


# ---------------------------------------------------------------------------
# Week-9 live-smoke fix A3: MacOSCaptureProvider wires app_under_cursor into the packet
# ---------------------------------------------------------------------------


def test_macos_capture_provider_populates_app_under_cursor(monkeypatch) -> None:
    monkeypatch.setattr(
        macos_capture,
        "capture_cursor",
        lambda: CursorPosition(x=10, y=20, screen_id=1),
    )
    monkeypatch.setattr(macos_capture, "capture_focus_window", lambda: FocusWindow(app="Ghostty"))
    monkeypatch.setattr(macos_capture, "capture_semantic_context", lambda: SemanticContext())
    monkeypatch.setattr(macos_capture, "capture_hover_region", lambda _cursor, _scale: None)
    monkeypatch.setattr(macos_capture, "_display_scale_for_screen", lambda _screen_id: 2.0)
    monkeypatch.setattr(macos_capture, "capture_app_under_cursor", lambda _x, _y: "Notion")

    packet = MacOSCaptureProvider().capture()

    assert packet.app_under_cursor == "Notion"


def test_macos_capture_provider_app_under_cursor_none_when_capture_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        macos_capture,
        "capture_cursor",
        lambda: CursorPosition(x=10, y=20, screen_id=1),
    )
    monkeypatch.setattr(macos_capture, "capture_focus_window", lambda: FocusWindow())
    monkeypatch.setattr(macos_capture, "capture_semantic_context", lambda: SemanticContext())
    monkeypatch.setattr(macos_capture, "capture_hover_region", lambda _cursor, _scale: None)
    monkeypatch.setattr(macos_capture, "_display_scale_for_screen", lambda _screen_id: 2.0)

    def _raise(_x, _y):
        raise RuntimeError("boom")

    monkeypatch.setattr(macos_capture, "capture_app_under_cursor", _raise)

    packet = MacOSCaptureProvider().capture()

    assert packet.app_under_cursor is None


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
