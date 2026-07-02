"""Headless probe: can one Interactions API call combine custom functions + built-in computer_use?

Phase-0 gate (goofy-sleeping-sutton plan). The SDK's ToolParam union statically allows
``{"type": "function", ...}`` and ``{"type": "computer_use", ...}`` in one ``tools=[...]``
list, which would let the DelegateAgent run a single combined loop instead of nesting the
computer-use executor behind a custom function. Server behavior may differ, so probe it:

    1. one interactions.create with BOTH tool types and a goal that needs the function first
    2. if it 400s → nested design stays (already the shipped default)
    3. if it answers, respond to the function call and see whether a computer_use action
       (predefined UI function like click/type) can follow in the same chain

Usage:
    uv run python scripts/diag/probe_interactions_combined_tools.py
"""

from __future__ import annotations

import base64
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "scripts" / "bench"))

from google import genai  # noqa: E402
from measure_ttfb import load_api_key  # noqa: E402

MODEL = "gemini-3.5-flash"
W, H = 1280, 800


def synthetic_screenshot() -> bytes:
    """A plain PNG desktop with one obvious dark 'icon' region — no PIL dependency."""
    rows = []
    for y in range(H):
        row = bytearray([0])
        for x in range(W):
            inside = 100 <= x <= 220 and 100 <= y <= 180
            row += bytes((40, 40, 60) if inside else (235, 238, 240))
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "run_shell",
        "description": "Run a shell command on the host and return its stdout.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {"type": "computer_use", "environment": "desktop"},
]


def steps_summary(interaction: Any) -> list[str]:
    out = []
    for step in interaction.steps or []:
        kind = getattr(step, "type", "?")
        name = getattr(step, "name", None)
        out.append(f"{kind}{f'({name})' if name else ''}")
    return out


def main() -> int:
    client = genai.Client(api_key=load_api_key())
    shot_b64 = base64.b64encode(synthetic_screenshot()).decode("ascii")
    try:
        first = client.interactions.create(
            model=MODEL,
            input=[
                {
                    "type": "text",
                    "text": (
                        "First use run_shell to check the current directory (pwd). After you "
                        "get the result, click the dark icon in the top-left of the screen."
                    ),
                },
                {"type": "image", "data": shot_b64, "mime_type": "image/png"},
            ],
            tools=TOOLS,
        )
    except Exception as exc:
        print(f"[probe] combined tools REJECTED at create: {type(exc).__name__}: {exc}")
        print("[probe] VERDICT: FAIL — keep the nested computer_use-as-function design")
        return 1
    print(f"[probe] round 1 steps: {steps_summary(first)}")

    fcs = [s for s in first.steps or [] if getattr(s, "type", None) == "function_call"]
    shell_calls = [s for s in fcs if s.name == "run_shell"]
    if not shell_calls:
        print(
            "[probe] model never called run_shell — combined list accepted but custom "
            "function unused; inspect steps above"
        )
        print("[probe] VERDICT: PARTIAL — accepted, custom-function usage unconfirmed")
        return 1
    call = shell_calls[0]
    second = client.interactions.create(
        model=MODEL,
        previous_interaction_id=first.id,
        input=[
            {
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [
                    {"type": "text", "text": '{"stdout": "/Users/probe"}'},
                    {"type": "image", "data": shot_b64, "mime_type": "image/png"},
                ],
            }
        ],
        tools=TOOLS,
    )
    print(f"[probe] round 2 steps: {steps_summary(second)}")
    ui_actions = [
        s
        for s in second.steps or []
        if getattr(s, "type", None) == "function_call" and s.name != "run_shell"
    ]
    if ui_actions:
        args = dict(ui_actions[0].arguments or {})
        args.pop("safety_decision", None)
        print(f"[probe] computer_use action followed: {ui_actions[0].name} {args}")
        print("[probe] VERDICT: PASS — one combined loop can serve the DelegateAgent")
        return 0
    print("[probe] no UI action followed the shell result; combined loop unproven")
    print("[probe] VERDICT: PARTIAL — accepted, but keep the nested design")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
