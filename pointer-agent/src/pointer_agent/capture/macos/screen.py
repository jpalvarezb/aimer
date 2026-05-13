"""macOS cursor-adjacent screen capture.

Week 1 deliberately leaves pixel capture disabled. Week 2 will crop a 256x256 tile
around the cursor and return it as a base64-encoded hover region.
"""

from __future__ import annotations

from jointer_core import CursorPosition, HoverRegion


def capture_hover_region(_cursor: CursorPosition) -> HoverRegion | None:
    """Return the cursor-adjacent hover region, if pixel capture is enabled."""

    return None
