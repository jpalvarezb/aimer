from __future__ import annotations

from aimer_core import CursorPosition
from pointer_agent.capture.debounce import CursorSettleDetector


def _cursor(x: float, y: float = 0.0) -> CursorPosition:
    return CursorPosition(x=x, y=y, screen_id=0)


def test_cursor_moving_returns_false() -> None:
    detector = CursorSettleDetector()

    assert detector.update(_cursor(0), now=0.00) is False
    assert detector.update(_cursor(20), now=0.05) is False
    assert detector.update(_cursor(40), now=0.10) is False
    assert detector.update(_cursor(60), now=0.15) is False


def test_cursor_settled_after_window() -> None:
    detector = CursorSettleDetector()

    assert detector.update(_cursor(100, 100), now=0.00) is False
    assert detector.update(_cursor(104, 103), now=0.10) is False
    assert detector.update(_cursor(104, 103), now=0.20) is True


def test_settle_is_one_shot_until_min_interval() -> None:
    detector = CursorSettleDetector()

    assert detector.update(_cursor(100), now=0.00) is False
    assert detector.update(_cursor(100), now=0.16) is True
    assert detector.update(_cursor(100), now=0.21) is False
    assert detector.update(_cursor(100), now=0.27) is True


def test_movement_resets_anchor() -> None:
    detector = CursorSettleDetector()

    assert detector.update(_cursor(100), now=0.00) is False
    assert detector.update(_cursor(100), now=0.16) is True
    assert detector.update(_cursor(150), now=0.17) is False
    assert detector.update(_cursor(150), now=0.30) is False
    assert detector.update(_cursor(150), now=0.33) is True


def test_first_update_returns_false() -> None:
    detector = CursorSettleDetector()

    assert detector.update(_cursor(100), now=100.0) is False
