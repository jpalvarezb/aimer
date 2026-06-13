"""Week-5 entity extraction: a swappable VLM seam + typed router + off-hot-path pipeline.

Public surface:
  - EntityExtractor / ExtractionContext       — the pluggable VLM seam (base.py)
  - LocalQwenVLExtractor                       — local Qwen3-VL via MLX (default; vlm extra)
  - GeminiFlashLiteExtractor                   — remote fallback (no extra)
  - MockExtractor                              — deterministic, for tests/dry-run
  - EntityRouter / RoutedAction / RecordingHandler — place->Maps, date->Calendar, code->IDE
  - EntityPipeline                             — extract -> populate packet -> route, off hot path
"""

from __future__ import annotations

from .base import EntityExtractor, ExtractionContext
from .gemini_vlm import GeminiFlashLiteExtractor
from .mlx_vlm import LocalQwenVLExtractor
from .mock import MockExtractor
from .pipeline import EntityPipeline, context_from_packet, tile_bytes_from_packet
from .router import DEFAULT_ROUTES, EntityRouter, RecordingHandler, RoutedAction, RouteTarget

__all__ = [
    "DEFAULT_ROUTES",
    "EntityExtractor",
    "EntityPipeline",
    "EntityRouter",
    "ExtractionContext",
    "GeminiFlashLiteExtractor",
    "LocalQwenVLExtractor",
    "MockExtractor",
    "RecordingHandler",
    "RoutedAction",
    "RouteTarget",
    "context_from_packet",
    "tile_bytes_from_packet",
]
