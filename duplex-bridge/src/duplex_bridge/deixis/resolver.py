"""Pointer-referent resolver — a small vision model that reads what the cursor points at.

Same shape as :class:`~duplex_bridge.entities.gemini_vlm.GeminiFlashLiteExtractor` (lazy
client, async ``generate_content``, never raises into the caller), but free-text output:
one short sentence naming the pointed-at element, injected as the ``pointer=`` field of
the live model's ``[context]`` annotation and handed to delegated tasks as grounding.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("duplex_bridge.deixis")

DEFAULT_MODEL = "gemini-flash-lite-latest"

# Referents are injected into a realtime text annotation; keep them one-line and short.
_MAX_REFERENT_CHARS = 200

_PROMPT = """\
The image is a screenshot tile centered on the user's mouse cursor. In ONE short sentence \
(under {max_chars} characters), state exactly what element or content the cursor is pointing \
at — specific enough that an assistant could answer "what is this?" about it. Name the thing \
itself (the button, link text, heading, code symbol, image subject...), not the screenshot. \
Reply with the sentence only: no preamble, no markdown.\
"""


@dataclass(frozen=True)
class PointerContext:
    """Non-pixel hints handed to the resolver alongside the tile (pure packet projection)."""

    app: str | None = None
    window_title: str | None = None
    accessibility_label: str | None = None
    selected_text: str | None = None
    cursor_tile_x: float | None = None
    cursor_tile_y: float | None = None


class PointerReferentResolver:
    """Resolve the pointed-at referent from a cursor tile via Gemini Flash-Lite."""

    name = "gemini-flash-lite-pointer"

    def __init__(self, model: str = DEFAULT_MODEL, api_key_env: str = "GEMINI_API_KEY") -> None:
        self._model = model
        self._api_key_env = api_key_env
        self._client: object | None = None

    def _ensure_client(self) -> object:
        if self._client is None:
            from google import genai  # noqa: PLC0415

            api_key = os.environ.get(self._api_key_env, "")
            if not api_key:
                raise RuntimeError(f"{self._api_key_env} not set for PointerReferentResolver")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def resolve(self, tile_jpeg: bytes, context: PointerContext | None = None) -> str:
        """Return a one-sentence referent description, or "" on any failure (never raises)."""
        from google.genai import types  # noqa: PLC0415

        try:
            client = self._ensure_client()
            resp = await client.aio.models.generate_content(  # type: ignore[attr-defined]
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=tile_jpeg, mime_type="image/jpeg"),
                    self._build_prompt(context),
                ],
            )
        except Exception as exc:  # noqa: BLE001 — resolver failures must never hit the hot path
            logger.warning("pointer resolve failed: %s", exc)
            return ""
        text = (getattr(resp, "text", None) or "").strip().replace("\n", " ")
        return text[:_MAX_REFERENT_CHARS]

    @staticmethod
    def _build_prompt(context: PointerContext | None) -> str:
        prompt = _PROMPT.format(max_chars=_MAX_REFERENT_CHARS)
        if context is None:
            return prompt
        if context.app:
            window_clause = f" (window: '{context.window_title}')" if context.window_title else ""
            prompt += (
                f"\nThis tile is from the app '{context.app}'{window_clause}. The app "
                "identity is known — never guess the app or app type from pixels; describe "
                f"the pointed-at element as belonging to the app '{context.app}'. Describe "
                "only the UI element or content itself (the button, heading, text, image "
                "subject...) — do not assert or name any app identity in your answer; the "
                "app is already known from this fact, not from your description."
            )
        hints: list[str] = []
        if context.cursor_tile_x is not None and context.cursor_tile_y is not None:
            hints.append(
                f"cursor_in_tile=({context.cursor_tile_x:.0f},{context.cursor_tile_y:.0f})"
            )
        if context.app:
            hints.append(f"app={context.app}")
        if context.window_title:
            hints.append(f"window={context.window_title}")
        if context.accessibility_label:
            hints.append(f"ax={context.accessibility_label[:80]}")
        if context.selected_text:
            hints.append(f"selected={context.selected_text[:80]}")
        if hints:
            prompt += "\nContext (non-pixel hints): " + " | ".join(hints)
        return prompt
