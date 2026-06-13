"""Shared extraction prompt + robust JSON->Entity parsing for the VLM backends.

Both the local MLX (Qwen3-VL) and the Gemini Flash-Lite extractors ask the model for
the same JSON shape and parse it the same way, so their outputs are directly
comparable and the router treats them identically.
"""

from __future__ import annotations

import json
import logging
from typing import get_args

from aimer_core.schema import Entity, EntityType

from .base import ExtractionContext

logger = logging.getLogger("duplex_bridge.entities")

_ALLOWED_TYPES: frozenset[str] = frozenset(get_args(EntityType))

_PROMPT = """\
You extract structured entities from a screenshot tile that is centered on the user's mouse \
cursor. Focus on what the cursor points at.

Return ONLY a JSON array. Each element is an object:
  "type": one of {types}
  "value": the literal text/value of the entity (a string)

Type guide: place = an address or location; date = a date/time; product = a named product; \
code_span = a code identifier or snippet; todo = an actionable task; unknown = anything else \
worth noting. Only include entities actually visible in the tile. If none, return [].

Example: [{{"type": "place", "value": "1600 Amphitheatre Pkwy, Mountain View"}}]
"""


def build_prompt(context: ExtractionContext | None = None) -> str:
    """Build the extraction prompt, appending non-pixel context hints when present."""
    prompt = _PROMPT.format(types=sorted(_ALLOWED_TYPES))
    if context is None:
        return prompt
    hints: list[str] = []
    if context.app:
        hints.append(f"app={context.app}")
    if context.window_title:
        hints.append(f"window={context.window_title}")
    if context.accessibility_label:
        hints.append(f"ax={context.accessibility_label}")
    if context.selected_text:
        hints.append(f"selected={context.selected_text}")
    if hints:
        prompt += "\nContext (non-pixel hints): " + " | ".join(hints)
    return prompt


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` fences and surrounding prose, keeping the JSON array."""
    text = raw.strip()
    if "```" in text:
        # take the content of the first fenced block
        parts = text.split("```")
        for part in parts:
            candidate = part
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:]
            if "[" in candidate and "]" in candidate:
                text = candidate
                break
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_entities(raw_text: str) -> list[Entity]:
    """Parse a model response into a list of validated ``Entity`` (tolerant of noise)."""
    payload = _strip_fences(raw_text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("entity parse: response was not valid JSON: %r", raw_text[:120])
        return []
    if not isinstance(data, list):
        return []

    entities: list[Entity] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        etype = item.get("type", "unknown")
        if etype not in _ALLOWED_TYPES:
            etype = "unknown"
        entities.append(Entity(type=etype, value=value.strip()))
    return entities
