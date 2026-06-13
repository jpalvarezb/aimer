"""EntityPipeline — extract typed entities from a packet's tile and route them.

This ties the :class:`EntityExtractor` seam to the :class:`EntityRouter` and runs the
whole thing *off the duplex audio hot path*: ``schedule`` fires extraction+routing as a
background task so the caller (the bridge) never awaits the VLM on the path that feeds
the duplex model. Week 5 needs it to run before the audio turn reaches the model; Week 6
hardens "never stalls the 200 ms tick under load".
"""

from __future__ import annotations

import asyncio
import base64
import logging

from aimer_core.schema import ContextPacket, Entity

from .base import EntityExtractor, ExtractionContext
from .router import EntityRouter, RoutedAction

logger = logging.getLogger("duplex_bridge.entities")


def context_from_packet(packet: ContextPacket) -> ExtractionContext:
    """Project the non-pixel hints already on a ContextPacket into an ExtractionContext."""
    return ExtractionContext(
        app=packet.focus_window.app,
        window_title=packet.focus_window.title,
        accessibility_label=packet.semantic.accessibility_label,
        selected_text=packet.semantic.selected_text,
    )


def tile_bytes_from_packet(packet: ContextPacket) -> bytes | None:
    """Decode the base64 cursor tile from a packet, or None if there is no tile."""
    if packet.hover_region is None or not packet.hover_region.tile_b64:
        return None
    try:
        return base64.b64decode(packet.hover_region.tile_b64)
    except (ValueError, TypeError):
        logger.warning("entity pipeline: undecodable tile_b64")
        return None


class EntityPipeline:
    """Runs extract -> populate packet.extracted_entities -> route+dispatch."""

    def __init__(self, extractor: EntityExtractor, router: EntityRouter | None = None) -> None:
        self.extractor = extractor
        self.router = router or EntityRouter()

    async def process_tile(
        self, tile_jpeg: bytes, context: ExtractionContext | None = None
    ) -> tuple[list[Entity], list[RoutedAction]]:
        """Extract entities from raw tile bytes and route them. Returns (entities, actions)."""
        entities = await self.extractor.extract(tile_jpeg, context)
        actions = self.router.dispatch(entities)
        return entities, actions

    async def process_packet(
        self, packet: ContextPacket
    ) -> tuple[list[Entity], list[RoutedAction]]:
        """Extract from a packet's tile, write entities back onto the packet, and route."""
        tile = tile_bytes_from_packet(packet)
        if tile is None:
            return [], []
        entities, actions = await self.process_tile(tile, context_from_packet(packet))
        packet.extracted_entities = entities
        return entities, actions

    def schedule(
        self, packet: ContextPacket
    ) -> asyncio.Task[tuple[list[Entity], list[RoutedAction]]]:
        """Fire-and-forget extraction+routing off the hot path. Returns the Task.

        The caller does NOT await this on the audio path — entity work happens in the
        background while the duplex audio loop keeps running.
        """
        return asyncio.create_task(self.process_packet(packet))

    async def aclose(self) -> None:
        await self.extractor.aclose()
