"""DESIGN STUB — not wired to anything, no pixels.

MiningRecord is the unit of the Week-8 non-pixel workflow-mining tier.
It is derived entirely from existing ContextPacket fields (all non-pixel)
and is meant to be appended to a local, low-cadence, append-only sink that
lives OUTSIDE the deixis hot path.

Nothing imports this module yet. It is provided as a typed design sketch so
that the Week-8 implementation has an agreed schema and derivation to build from.
Do NOT add this to aimer_core/__init__.py or any __all__ list until it is wired.
"""

# not yet wired

from __future__ import annotations

from aimer_core.schema import ContextPacket, StrictBaseModel


class MiningRecord(StrictBaseModel):
    """One row in the non-pixel workflow-mining append-only sink.

    All fields are derived from ContextPacket with no pixels involved.
    Suitable for: heatmap generation (x/y over time), recurring-workflow
    detection (app/window/ax_identity sequences), and coarse UX analytics.

    The sink cadence is deliberately lower than the 10 Hz deixis stream
    (e.g. 1 Hz or on AX-element-change events) and runs off the hot path.
    Pixels are intentionally excluded: heatmaps need only (x, y, t); workflow
    sequences need element identity. Pixels can be added later — under a
    separate opt-in flag — if canvas/video UIs prove to need them.
    """

    # Timestamp (Unix seconds) matching ContextPacket.t.
    t: float

    # Cursor coordinates in logical points (screen-absolute).
    # Source: ContextPacket.cursor.x / .cursor.y
    cursor_x: float
    cursor_y: float

    # Screen identifier (multi-display support).
    # Source: ContextPacket.cursor.screen_id
    screen_id: int

    # Focused application name (None when not available).
    # Source: ContextPacket.focus_window.app
    app: str | None

    # Focused window title (None when not available).
    # Source: ContextPacket.focus_window.title
    window_title: str | None

    # AX element identity: the coarsest stable identifier for the hovered element.
    # Prefer accessibility_label when set (descriptive + stable); fall back to
    # dom_path (CSS/XPath, browser-only, path-stable within a page).
    # Source: ContextPacket.semantic.accessibility_label or .semantic.dom_path
    ax_identity: str | None

    # Current user selection (first 200 chars to cap record size).
    # Source: ContextPacket.semantic.selected_text
    selected_text: str | None

    @classmethod
    def from_packet(cls, packet: ContextPacket) -> MiningRecord:
        """Derive a MiningRecord from a ContextPacket.

        Pure — no I/O, no pixels, no side effects. Safe to call on the hot path
        (cheap field extraction only) if needed; intended to be called on the
        low-cadence mining tap.
        """
        sem = packet.semantic
        ax_id: str | None = None
        if sem.accessibility_label:
            ax_id = sem.accessibility_label
        elif sem.dom_path:
            ax_id = sem.dom_path

        selected: str | None = None
        if sem.selected_text:
            selected = sem.selected_text[:200]

        fw = packet.focus_window
        return cls(
            t=packet.t,
            cursor_x=packet.cursor.x,
            cursor_y=packet.cursor.y,
            screen_id=packet.cursor.screen_id,
            app=fw.app if fw else None,
            window_title=fw.title if fw else None,
            ax_identity=ax_id,
            selected_text=selected,
        )
