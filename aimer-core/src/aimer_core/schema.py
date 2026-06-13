"""Shared context packet schema for Aimer services."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictBaseModel(BaseModel):
    """Base model that rejects unknown fields to prevent packet drift."""

    model_config = ConfigDict(extra="forbid")


class BoundingBox(StrictBaseModel):
    """Region in logical points, screen-absolute.

    Same coordinate system as CursorPosition. To convert to physical
    pixels, multiply by `ContextPacket.display_scale`.
    """

    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)


class CursorPosition(StrictBaseModel):
    """Cursor location in logical points, screen-absolute."""

    x: float
    y: float
    screen_id: int = 0


class FocusWindow(StrictBaseModel):
    """Best-effort metadata for the current foreground window."""

    app: str | None = None
    title: str | None = None
    url: HttpUrl | str | None = None


HoverRegionType = Literal["image", "text", "code", "table", "video_frame", "unknown"]


class HoverRegion(StrictBaseModel):
    """Pixel and semantic details for the cursor-adjacent visual region."""

    type: HoverRegionType = "unknown"
    bbox: BoundingBox | None = None
    tile_b64: str | None = None
    cursor_tile_x: float | None = Field(
        default=None,
        description="Cursor x offset from the tile origin, in logical points "
        "(cursor.x - bbox.x). Numeric deictic anchor, independent of any drawn marker.",
    )
    cursor_tile_y: float | None = Field(
        default=None,
        description="Cursor y offset from the tile origin, in logical points (cursor.y - bbox.y).",
    )


class FullFrame(StrictBaseModel):
    """Downscaled, cursor-marked full-display frame for relational-deixis escalation.

    Orthogonal to `HoverRegion` (a different scale): the tile is the cursor-adjacent
    context-floor crop; the full frame carries layout/multi-referent context for
    utterances like "compare these two". Captured off the hot path at a low cadence
    and force-sent at turn start only when escalation is enabled.
    """

    frame_b64: str
    width_px: int = Field(ge=1, description="Encoded width of the downscaled frame, in pixels.")
    height_px: int = Field(ge=1, description="Encoded height of the downscaled frame, in pixels.")
    display_scale: float = 1.0
    cursor_frame_x: float | None = Field(
        default=None,
        description="Cursor x offset from the display origin, in logical points.",
    )
    cursor_frame_y: float | None = Field(
        default=None,
        description="Cursor y offset from the display origin, in logical points.",
    )


class SemanticContext(StrictBaseModel):
    """Cheap semantic context from accessibility and app-specific surfaces."""

    accessibility_label: str | None = None
    selected_text: str | None = None
    dom_path: str | None = None
    language: str | None = None


EntityType = Literal["place", "date", "product", "code_span", "todo", "unknown"]


class Entity(StrictBaseModel):
    """Typed entity extracted from the hover region."""

    type: EntityType = "unknown"
    value: str
    bbox: BoundingBox | None = None


class ContextPacket(StrictBaseModel):
    """One 200 ms visual/deictic context packet for the duplex model."""

    t: float = Field(default_factory=time.time)
    cursor: CursorPosition
    display_scale: float = Field(
        default=1.0,
        description="backingScaleFactor of the screen containing the cursor at capture time.",
    )
    focus_window: FocusWindow = Field(default_factory=FocusWindow)
    hover_region: HoverRegion | None = None
    full_frame: FullFrame | None = Field(
        default=None,
        description="Optional downscaled full-display frame for relational-deixis escalation. "
        "Present only on the low-cadence ticks that refresh it; None on most packets.",
    )
    semantic: SemanticContext = Field(default_factory=SemanticContext)
    extracted_entities: list[Entity] = Field(default_factory=list)
