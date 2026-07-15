#!/usr/bin/env python3
"""Repro: does disabling enable_prompt_injection_detection clear the 'Input blocked' 400?

The block fires on the policy's FIRST interactions.create() call, before any Action is
applied. So this makes exactly ONE policy call per goal and inspects the outcome — it
takes a real screenshot but NEVER drives the mouse/keyboard. Safe to run mid-use.

  GEMINI_API_KEY=... uv run python scripts/diag/repro_injection_detection_disable.py
"""

import os
import traceback

from duplex_bridge.actions.computer import MacOSComputer
from duplex_bridge.actions.computer_policy import GeminiComputerUsePolicy


def probe(label: str, goal: str, shot: bytes) -> None:
    print(f"\n--- {label} ---")
    print(f"Goal: {goal}")
    # Fresh policy per goal so interaction-id chaining doesn't leak between probes.
    policy = GeminiComputerUsePolicy(model="gemini-3.5-flash", environment="desktop")
    try:
        action = policy(goal, shot, [])  # one perceive/decide step; no apply()
        print(f"NOT BLOCKED -> first Action: {action.type} note={action.note!r}")
    except Exception as e:  # noqa: BLE001 — we want the raw failure surface
        msg = str(e)
        blocked = "Input blocked" in msg or "blocked" in msg.lower()
        print(f"{'BLOCKED' if blocked else 'ERROR'}: {type(e).__name__}: {msg[:300]}")
        if not blocked:
            traceback.print_exc()


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set")
        return

    shot = MacOSComputer().screenshot()
    print(f"Captured screenshot: {len(shot)} bytes")

    # Previously blocked (2026-07-04 13:30): screen-viewing / screenshot framing.
    probe(
        "Test 1: screen-viewing goal (previously blocked)",
        "Take a screenshot and tell me what applications are currently visible on the screen",
        shot,
    )
    # Control: benign named-app action succeeded before the flag change.
    probe(
        "Test 2: named-app goal (control)",
        "Open the Notes application",
        shot,
    )


if __name__ == "__main__":
    main()
