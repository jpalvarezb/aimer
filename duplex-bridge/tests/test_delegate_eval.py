"""Delegate-safety-overhaul eval, gated in the suite (drives the eval_delegate_safety
scenarios) — mirrors test_fd.py's pattern for eval_fd.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))

import eval_delegate_safety  # noqa: E402
import pytest  # noqa: E402


@pytest.mark.parametrize("name", sorted(eval_delegate_safety.SCENARIOS))
async def test_delegate_safety_scenario(name: str) -> None:
    spec = eval_delegate_safety.SCENARIOS[name]
    ok, detail = await eval_delegate_safety.run_scenario(spec)
    assert ok, detail


def test_delegate_safety_eval_covers_15_to_20_canonical_tasks() -> None:
    assert 15 <= len(eval_delegate_safety.SCENARIOS) <= 20
