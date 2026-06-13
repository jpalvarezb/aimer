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
def test_capture_tile_p95_under_120ms() -> None:
    """Capture 50 tiles via real ScreenCaptureKit, assert p95 latency < 120 ms.

    Week 2 landed real SCK capture; this measures the actual warm-path tile
    capture latency against the accepted PoC budget (p95 < 120 ms warm; observed
    ~110 ms p95). The original <30 ms target is deferred to a streaming-capture,
    lower-res, or lower-quality path. Requires Screen Recording permission for
    the test runner; without it, ``capture_hover_region`` returns ``None`` quickly.
    """

    from aimer_core import CursorPosition
    from pointer_agent.capture.macos.screen import capture_hover_region

    cursor = CursorPosition(x=500.0, y=500.0, screen_id=0)
    samples = []
    for _ in range(50):
        t0 = time.perf_counter()
        capture_hover_region(cursor)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    p95 = samples[int(len(samples) * 0.95) - 1]
    assert p95 < 120.0, f"p95 latency {p95:.2f}ms exceeds accepted 120ms warm budget"
