"""MockExtractor — a deterministic EntityExtractor for tests and dry-run wiring.

Returns a fixed entity list (optionally derived from the context's selected_text), with
no model and no network, so the pipeline/router can be exercised without a download.
"""

from __future__ import annotations

from aimer_core.schema import Entity

from .base import EntityExtractor, ExtractionContext


class MockExtractor(EntityExtractor):
    """Returns a preset list of entities; records how many times it was called."""

    name = "mock"

    def __init__(self, entities: list[Entity] | None = None) -> None:
        self._entities = entities or []
        self.calls = 0

    async def extract(
        self, tile_jpeg: bytes, context: ExtractionContext | None = None
    ) -> list[Entity]:
        self.calls += 1
        return list(self._entities)
