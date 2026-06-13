"""Remote VLM entity extractor — Gemini Flash-Lite (fallback / non-Apple-Silicon).

Reuses ``google-genai`` (already a bridge dependency for Gemini Live), so it needs no
extra install. It is NOT local — kept as a swappable fallback behind the
``EntityExtractor`` seam for environments without MLX, and for A/B against the local
model. The criterion-satisfying default is :class:`LocalQwenVLExtractor`.
"""

from __future__ import annotations

import logging
import os

from aimer_core.schema import Entity

from ._extract_common import build_prompt, parse_entities
from .base import EntityExtractor, ExtractionContext

logger = logging.getLogger("duplex_bridge.entities")

DEFAULT_MODEL = "gemini-flash-lite-latest"


class GeminiFlashLiteExtractor(EntityExtractor):
    """Entity extractor backed by the Gemini Flash-Lite multimodal model (API)."""

    name = "gemini-flash-lite"

    def __init__(self, model: str = DEFAULT_MODEL, api_key_env: str = "GEMINI_API_KEY") -> None:
        self._model = model
        self._api_key_env = api_key_env
        self._client: object | None = None

    def _ensure_client(self) -> object:
        if self._client is None:
            from google import genai  # noqa: PLC0415

            api_key = os.environ.get(self._api_key_env, "")
            if not api_key:
                raise RuntimeError(f"{self._api_key_env} not set for GeminiFlashLiteExtractor")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def extract(
        self, tile_jpeg: bytes, context: ExtractionContext | None = None
    ) -> list[Entity]:
        from google.genai import types  # noqa: PLC0415

        client = self._ensure_client()
        prompt = build_prompt(context)
        try:
            resp = await client.aio.models.generate_content(  # type: ignore[attr-defined]
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=tile_jpeg, mime_type="image/jpeg"),
                    prompt,
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("gemini flash-lite extract failed: %s", exc)
            return []
        return parse_entities(resp.text or "")
