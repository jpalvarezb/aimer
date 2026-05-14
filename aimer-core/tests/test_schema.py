from __future__ import annotations

import pytest
from aimer_core import ContextPacket, CursorPosition, Entity, FocusWindow, SemanticContext
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
        ContextPacket(cursor=CursorPosition(x=0, y=0), unknown=True)  # type: ignore[call-arg]


def test_display_scale_defaults_to_one() -> None:
    packet = ContextPacket(cursor=CursorPosition(x=0, y=0))

    assert packet.display_scale == 1.0


def test_display_scale_can_be_set() -> None:
    packet = ContextPacket(cursor=CursorPosition(x=0, y=0), display_scale=2.0)

    assert packet.display_scale == 2.0
