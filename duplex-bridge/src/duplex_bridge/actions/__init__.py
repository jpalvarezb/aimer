"""Week-7 host-app actions — the actuators a tool call drives (Chrome, IDE).

These are the "host app actions" half of the loop: a deictic utterance resolves to a tool call
(the model decides *what*), and these handlers carry it out on the host (the *how*). They run on
the Week-6 :class:`~duplex_bridge.worker.BackgroundWorker`, off the audio hot path.

- :func:`rewrite_function_async` — IDE: rewrite a function to async in a real file on disk.
- :func:`compare_products` — Chrome: open a side-by-side comparison of the pointed-at products.

``TOOL_DECLARATIONS`` are provider-neutral function schemas; ``GeminiLiveSession`` converts them
to ``types.Tool`` so the model can emit these calls live.
"""

from __future__ import annotations

from typing import Any

from .chrome import ComparisonResult, compare_products
from .ide import RewriteResult, rewrite_function_async

# Provider-neutral tool/function declarations (JSON-schema-ish). The bridge converts these into
# the model provider's tool format so the duplex model can emit the corresponding tool calls.
TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "rewrite_function_async",
        "description": (
            "Rewrite the function the user is pointing at so it is asynchronous (async def, "
            "awaiting blocking calls). Use when the user says e.g. 'rewrite this function async'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to the source file."},
                "function": {"type": "string", "description": "Name of the function to rewrite."},
                "new_source": {
                    "type": "string",
                    "description": "Optional full async rewrite; omit to apply a transform.",
                },
            },
            "required": ["file", "function"],
        },
    },
    {
        "name": "compare_products",
        "description": (
            "Open a side-by-side comparison of two or more products the user is pointing at. "
            "Use when the user says e.g. 'compare these products'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Product names to compare.",
                }
            },
            "required": ["products"],
        },
    },
]

__all__ = [
    "TOOL_DECLARATIONS",
    "ComparisonResult",
    "RewriteResult",
    "compare_products",
    "rewrite_function_async",
]
