"""EntityExtractor seam — pluggable VLM that emits typed entities from a cursor tile.

Mirrors the other Aimer abstraction boundaries (``DuplexSession``,
``CaptureProvider``, ``AudioBackend``): the bridge depends only on this ABC, so the
concrete vision model — local MLX Qwen3-VL, Gemini Flash-Lite, or a future model —
stays swappable without touching the pipeline or the duplex session.

The extractor consumes the high-res cursor tile (base64-decoded JPEG bytes) plus
optional non-pixel hints (app / window / AX label / selected text, all already in
``ContextPacket``) and returns ``aimer_core.schema.Entity`` values. It does NOT route
or act on them — that is :mod:`duplex_bridge.entities.router`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from aimer_core.schema import Entity


@dataclass(frozen=True)
class ExtractionContext:
    """Non-pixel hints handed to the extractor alongside the tile.

    All fields are already present on ``ContextPacket`` (focus_window / semantic), so
    populating this from a packet is a pure projection — no extra capture.
    """

    app: str | None = None
    window_title: str | None = None
    accessibility_label: str | None = None
    selected_text: str | None = None


class EntityExtractor(ABC):
    """Abstract local-or-remote VLM entity extractor.

    Implementations must be safe to call from an asyncio event loop: a model that runs
    synchronously (e.g. MLX) should offload to a thread inside :meth:`extract` so the
    duplex audio loop is never blocked.
    """

    #: Short stable identifier for logs / metrics (e.g. "qwen3-vl-mlx").
    name: str = "base"

    @abstractmethod
    async def extract(
        self, tile_jpeg: bytes, context: ExtractionContext | None = None
    ) -> list[Entity]:
        """Return the typed entities the cursor tile contains (possibly empty)."""

    async def aclose(self) -> None:
        """Release any held resources (model handles, sessions). Default: no-op."""
        return None
