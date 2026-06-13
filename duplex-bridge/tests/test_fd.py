"""Week-8 full-duplex behaviors gated in the suite (drives the eval_fd scenarios)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))

import eval_fd  # noqa: E402


async def test_fd_backchannel_does_not_open_turn():
    ok, detail = await eval_fd.scenario_backchannel()
    assert ok, detail


async def test_fd_talkover_opens_turn():
    ok, detail = await eval_fd.scenario_talkover_opens_turn()
    assert ok, detail


async def test_fd_turn_taking_closes_on_silence():
    ok, detail = await eval_fd.scenario_turn_taking_closes()
    assert ok, detail


async def test_fd_interrupt_flushes_playback():
    ok, detail = await eval_fd.scenario_interrupt_flushes_playback()
    assert ok, detail


def test_fd_interrupt_detection():
    ok, detail = eval_fd.scenario_interrupt_detection()
    assert ok, detail
