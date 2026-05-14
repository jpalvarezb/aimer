"""Opt-in latency benchmark for tile capture.

Run with: pytest -m benchmark pointer-agent/tests/test_capture_tile_latency.py
Skipped by default in CI.
"""

from __future__ import annotations

import sys
import time

import pytest

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(sys.platform != "darwin", reason="macOS only"),
]


@pytest.mark.benchmark
def test_capture_tile_p95_under_30ms() -> None:
    """Capture 50 tiles, assert p95 latency < 30 ms.

    Week 2 will replace the stub here with real SCK capture.
    Currently asserts the stub returns quickly to validate the
    harness; the real perf bar activates when tile capture lands.
    """

    from aimer_core import CursorPosition
    from pointer_agent.capture.macos.screen import capture_hover_region

    cursor = CursorPosition(x=500.0, y=500.0, screen_id=0)
    samples = []
    for _ in range(50):
        t0 = time.perf_counter()
        capture_hover_region(cursor)
        samples.append((time.perf_counter() - t0) * 1000.0)

    # When Week 2 lands real capture, this test will start measuring actual latency.
    samples.sort()
    p95 = samples[int(len(samples) * 0.95) - 1]
    assert p95 < 30.0, f"p95 latency {p95:.2f}ms exceeds 30ms budget"
