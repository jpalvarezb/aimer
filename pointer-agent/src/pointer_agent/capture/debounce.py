"""Platform-agnostic cursor-settle detection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aimer_core import CursorPosition


@dataclass
class CursorSettleDetector:
    """Detect when the cursor has stopped moving.

    A cursor is "settled" once it has stayed within `settle_radius_px`
    of its last-moved position for at least `settle_time_s`. After a
    settle event fires, `min_interval_s` must elapse before another
    settle event can fire, even if the cursor moves and re-settles
    in the meantime.

    All distances are in the same units as CursorPosition (logical
    points). Default `settle_radius_px=8.0` is named for legacy
    reasons; treat it as 8 logical points.
    """

    settle_radius_px: float = 8.0
    settle_time_s: float = 0.15
    min_interval_s: float = 0.10

    _anchor: CursorPosition | None = field(default=None, init=False, repr=False)
    _anchor_set_at: float = field(default=0.0, init=False, repr=False)
    _last_emit_at: float = field(default=0.0, init=False, repr=False)

    def update(self, cursor: CursorPosition, now: float | None = None) -> bool:
        """Return True iff the cursor has just settled. Idempotent within a settle period."""

        t = now if now is not None else time.monotonic()

        if self._anchor is None or self._distance(cursor, self._anchor) > self.settle_radius_px:
            # Cursor is moving, or this is the first sample, so reset the anchor.
            self._anchor = cursor
            self._anchor_set_at = t
            return False

        if (t - self._anchor_set_at) < self.settle_time_s:
            return False
        if (t - self._last_emit_at) < self.min_interval_s:
            return False

        self._last_emit_at = t
        # Do not reset the anchor; staying still keeps emits suppressed by
        # _last_emit_at until the minimum interval elapses.
        return True

    @staticmethod
    def _distance(a: CursorPosition, b: CursorPosition) -> float:
        dx, dy = a.x - b.x, a.y - b.y
        return (dx * dx + dy * dy) ** 0.5
