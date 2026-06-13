"""Local VLM entity extractor — Qwen3-VL via MLX (Apple Silicon).

This is the criterion-satisfying default for Week 5: the entity model runs *locally*,
offline, and off the duplex audio hot path (synchronous MLX inference is offloaded to a
worker thread inside :meth:`extract`). Requires the ``vlm`` extra
(``uv pip install -e "duplex-bridge[vlm]"``) and downloads the model on first use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aimer_core.schema import Entity

from ._extract_common import build_prompt, parse_entities
from .base import EntityExtractor, ExtractionContext

logger = logging.getLogger("duplex_bridge.entities")

# Default to the 2B model. NOTE on this dev Mac: the GPU faults *stochastically* under
# repeated MLX VLM inference — it is NOT a memory problem (per-inference peak ~2.6 GB on a
# ~13 GB GPU; capping memory did NOT help and is removed). A single occasional extraction
# (the production cadence — change-gated, one tile at a time) is safe; a back-to-back
# benchmark loop is not. For measured accuracy numbers, use the Gemini fallback backend.
# `model_id` is overridable (e.g. ...Qwen3-VL-4B-Instruct-4bit) for A/B.
DEFAULT_MODEL_ID = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

# Watchdog-mitigation levers (env-overridable). The heavier 4B can HANG this GPU
# (kIOGPUCommandBufferCallbackErrorHang = a command buffer ran too long and tripped the macOS
# GPU watchdog). Downscaling the tile shrinks the vision-encoder/prefill kernel; fewer max
# tokens shortens decode. These are the "cap resolution / reduce context" levers.
DEFAULT_MAX_TOKENS = int(os.environ.get("ENTITY_VLM_MAX_TOKENS", "128"))
DEFAULT_TILE_MAX_EDGE = int(os.environ.get("ENTITY_VLM_TILE_MAXPX", "0")) or None


def _clear_mlx_cache() -> None:
    """Free MLX's Metal buffer cache between generations (memory hygiene in long loops)."""
    try:
        import mlx.core as mx  # noqa: PLC0415

        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            clear()
    except Exception:  # noqa: BLE001
        pass


class LocalQwenVLExtractor(EntityExtractor):
    """Qwen3-VL entity extractor running locally through MLX.

    The model is loaded lazily on the first ``extract`` call (so constructing the
    extractor is cheap and import-safe on machines without the model downloaded).
    """

    name = "qwen3-vl-mlx"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_tokens: int | None = None,
        tile_max_edge: int | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        self._tile_max_edge = tile_max_edge if tile_max_edge is not None else DEFAULT_TILE_MAX_EDGE
        self._model: Any = None
        self._processor: Any = None
        self._config: Any = None
        # All MLX work runs on ONE dedicated thread. Metal binds its context to the thread
        # that first initialized it; running load() and generate() on different pool threads
        # (as asyncio.to_thread can) mismatches the context and hangs the GPU. A single-worker
        # executor pins every MLX op to the same thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-vlm")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load  # noqa: PLC0415 (heavy optional import)
        from mlx_vlm.utils import load_config  # noqa: PLC0415

        logger.info("loading local VLM %s ...", self._model_id)
        self._model, self._processor = load(self._model_id)
        self._config = load_config(self._model_id)
        logger.info("local VLM loaded: %s", type(self._model).__name__)

    def _maybe_downscale(self, tile_jpeg: bytes) -> bytes:
        """Shrink the tile's longest edge to ``tile_max_edge`` to bound the vision kernel.

        A smaller image means fewer vision tokens and a shorter vision-encoder/prefill command
        buffer — the lever against the macOS GPU watchdog hang seen on the heavier 4B model.
        """
        if not self._tile_max_edge:
            return tile_jpeg
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        img = Image.open(io.BytesIO(tile_jpeg)).convert("RGB")
        longest = max(img.size)
        if longest <= self._tile_max_edge:
            return tile_jpeg
        scale = self._tile_max_edge / longest
        new_size = (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _extract_sync(self, tile_jpeg: bytes, context: ExtractionContext | None) -> list[Entity]:
        import mlx.core as mx  # noqa: PLC0415
        from mlx_vlm import generate  # noqa: PLC0415
        from mlx_vlm.prompt_utils import apply_chat_template  # noqa: PLC0415

        self._ensure_loaded()
        tile_jpeg = self._maybe_downscale(tile_jpeg)
        prompt = build_prompt(context)
        formatted = apply_chat_template(self._processor, self._config, prompt, num_images=1)

        reset = getattr(mx, "reset_peak_memory", None)
        if reset is not None:
            reset()

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
                fh.write(tile_jpeg)
                tmp_path = fh.name
            result = generate(
                self._model,
                self._processor,
                formatted,
                image=tmp_path,
                max_tokens=self._max_tokens,
                temperature=0.0,
                verbose=False,
            )
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

        get_peak = getattr(mx, "get_peak_memory", None)
        peak_gb = (get_peak() / 1e9) if get_peak is not None else -1.0
        print(
            f"[VLM] peak_mem={peak_gb:.2f}GB  max_tokens={self._max_tokens}  "
            f"tile_max_edge={self._tile_max_edge}",
            file=sys.stderr,
            flush=True,
        )

        text = getattr(result, "text", str(result))
        _clear_mlx_cache()
        return parse_entities(text)

    async def extract(
        self, tile_jpeg: bytes, context: ExtractionContext | None = None
    ) -> list[Entity]:
        # MLX inference is synchronous and GPU-bound — offload to the dedicated single
        # thread so the duplex audio event loop is never blocked AND Metal's context stays
        # pinned to one thread (see __init__).
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._extract_sync, tile_jpeg, context)

    async def aclose(self) -> None:
        self._executor.shutdown(wait=True)
