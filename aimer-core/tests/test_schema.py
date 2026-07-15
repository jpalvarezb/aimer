from __future__ import annotations

import pytest
from aimer_core import (
    ContextPacket,
    CursorPosition,
    Entity,
    FocusWindow,
    FullFrame,
    HoverRegion,
    SemanticContext,
)
from pydantic import ValidationError


def test_context_packet_serializes_spec_shape() -> None:
    packet = ContextPacket(
        cursor=CursorPosition(x=842, y=311, screen_id=0),
        focus_window=FocusWindow(app="Chrome", title="Aimer", url="https://example.com"),
        semantic=SemanticContext(
            accessibility_label="main article",
            selected_text="fix this",
            dom_path="main > article",
            language="ts",
        ),
        extracted_entities=[Entity(type="todo", value="wire pointer telemetry")],
    )

    payload = packet.model_dump()

    assert payload["cursor"]["x"] == 842
    assert payload["focus_window"]["app"] == "Chrome"
    assert payload["semantic"]["selected_text"] == "fix this"
    assert payload["extracted_entities"][0]["type"] == "todo"


def test_context_packet_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ContextPacket(cursor=CursorPosition(x=0, y=0), unknown=True)


def test_display_scale_defaults_to_one() -> None:
    packet = ContextPacket(cursor=CursorPosition(x=0, y=0))

    assert packet.display_scale == 1.0


def test_display_scale_can_be_set() -> None:
    packet = ContextPacket(cursor=CursorPosition(x=0, y=0), display_scale=2.0)

    assert packet.display_scale == 2.0


def test_full_frame_defaults_to_none() -> None:
    packet = ContextPacket(cursor=CursorPosition(x=0, y=0))

    assert packet.full_frame is None


def test_full_frame_round_trip() -> None:
    packet = ContextPacket(
        cursor=CursorPosition(x=100, y=200),
        full_frame=FullFrame(
            frame_b64="ZmFrZQ==",
            width_px=1024,
            height_px=640,
            display_scale=2.0,
            cursor_frame_x=100.0,
            cursor_frame_y=200.0,
        ),
    )

    restored = ContextPacket.model_validate_json(packet.model_dump_json())

    assert restored.full_frame is not None
    assert restored.full_frame.frame_b64 == "ZmFrZQ=="
    assert restored.full_frame.width_px == 1024
    assert restored.full_frame.cursor_frame_y == 200.0


def test_full_frame_rejects_nonpositive_dims() -> None:
    with pytest.raises(ValidationError):
        FullFrame(frame_b64="x", width_px=0, height_px=10)


def test_hover_region_carries_cursor_tile_offsets() -> None:
    region = HoverRegion(tile_b64="abc", cursor_tile_x=12.0, cursor_tile_y=34.0)

    assert region.cursor_tile_x == 12.0
    assert region.cursor_tile_y == 34.0


def test_old_packet_without_full_frame_deserializes() -> None:
    # A packet serialized before the full_frame field existed must still load.
    legacy = '{"t": 1.0, "cursor": {"x": 1.0, "y": 2.0}}'

    packet = ContextPacket.model_validate_json(legacy)

    assert packet.full_frame is None
    assert packet.hover_region is None


def test_app_under_cursor_defaults_to_none() -> None:
    # Week-9 live-smoke fix A3: pointer and focus can be different apps (e.g. cursor resting
    # over Notion while a terminal is focused); the field is additive and optional.
    packet = ContextPacket(cursor=CursorPosition(x=0, y=0))

    assert packet.app_under_cursor is None


def test_app_under_cursor_round_trips_through_dump_and_validate() -> None:
    packet = ContextPacket(
        cursor=CursorPosition(x=1, y=2),
        focus_window=FocusWindow(app="Ghostty"),
        app_under_cursor="Notion",
    )

    dumped = packet.model_dump()
    assert dumped["app_under_cursor"] == "Notion"

    restored = ContextPacket.model_validate(dumped)
    assert restored.app_under_cursor == "Notion"

    restored_json = ContextPacket.model_validate_json(packet.model_dump_json())
    assert restored_json.app_under_cursor == "Notion"
