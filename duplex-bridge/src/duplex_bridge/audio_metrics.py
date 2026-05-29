"""Small audio diagnostics helpers."""

from __future__ import annotations

import struct
from math import sqrt
from typing import cast


def compute_rms_int16(pcm: bytes) -> float:
    """Return RMS for little-endian 16-bit PCM samples."""
    sample_count = len(pcm) // 2
    if sample_count == 0:
        return 0.0

    samples = cast(tuple[int, ...], struct.unpack(f"<{sample_count}h", pcm[: sample_count * 2]))
    mean_square = float(sum(sample * sample for sample in samples)) / sample_count
    return sqrt(mean_square)


def percentile(values: list[float], percentile_value: int) -> float | None:
    """Return a nearest-rank percentile for a small diagnostics window."""
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = round((percentile_value / 100.0) * (len(sorted_values) - 1))
    return sorted_values[index]
