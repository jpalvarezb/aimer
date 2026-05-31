"""Dependency-free (numpy-only) rational resampler with anti-aliasing.

Used by the software-AEC backend (24 kHz model playback <-> 16 kHz echo
reference) and the macOS VPIO backend (hardware float32 @ 44.1/48 kHz -> 16 kHz
int16, and 24 kHz -> engine rate).

The resampler is polyphase-style: upsample by L (zero-stuffing), low-pass with a
windowed-sinc FIR, then downsample by M, where L/M = out_rate/in_rate in lowest
terms. It is block-friendly in the sense that each call is self-contained and
deterministic; callers can feed arbitrary chunk sizes (output length tracks the
ideal ratio within +/-1 sample).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import numpy.typing as npt

_INT16_MAX = 32767
# Floor at -32767 (not the true int16 min -32768) so |sample| < 32768 always
# holds and the output is symmetric / wrap-free.
_INT16_MIN = -32767

# Number of zero crossings on each side of the sinc kernel. Larger = sharper
# transition / better stopband at the cost of more compute.
_FILTER_HALF_ZEROS = 16


@lru_cache(maxsize=32)
def _design_lowpass(up: int, down: int) -> npt.NDArray[np.float64]:
    """Design a windowed-sinc (Kaiser) low-pass FIR for an L/M resampler.

    The filter runs in the upsampled domain (rate = in_rate * up). Its cutoff is
    the lower of the input/output Nyquist limits so it both reconstructs after
    upsampling and anti-aliases before downsampling. Gain is ``up`` to undo the
    energy lost to zero-stuffing.
    """
    max_rate = max(up, down)
    # Normalized cutoff (cycles/sample in the upsampled domain). 1.0 == Nyquist
    # of the upsampled signal. We back off slightly to leave a transition band.
    cutoff = 1.0 / max_rate
    # Half-length in upsampled samples: enough zero crossings for a clean stopband.
    half_len = _FILTER_HALF_ZEROS * max_rate
    n = np.arange(-half_len, half_len + 1, dtype=np.float64)
    # Ideal low-pass impulse response: cutoff * sinc(cutoff * n).
    sinc = cutoff * np.sinc(cutoff * n)
    # Kaiser window for a strong stopband (~ -74 dB at beta=8).
    window = np.kaiser(n.size, 8.0)
    taps = sinc * window
    taps *= up / taps.sum()
    return taps.astype(np.float64)


def _resample_float(
    samples: npt.NDArray[np.float64], in_rate: int, out_rate: int
) -> npt.NDArray[np.float64]:
    """Rational resample a float64 mono signal. Returns float64 (unclipped)."""
    if in_rate <= 0 or out_rate <= 0:
        raise ValueError("rates must be positive")
    if samples.size == 0:
        return samples.astype(np.float64, copy=False)
    if in_rate == out_rate:
        return samples.astype(np.float64, copy=True)

    g = math.gcd(in_rate, out_rate)
    up = out_rate // g
    down = in_rate // g

    taps = _design_lowpass(up, down)

    # Upsample: insert (up - 1) zeros between samples.
    upsampled = np.zeros(samples.size * up, dtype=np.float64)
    upsampled[::up] = samples

    # Low-pass filter. 'same' keeps the output centered on the input so group
    # delay is compensated and length bookkeeping stays simple.
    filtered = np.convolve(upsampled, taps, mode="same")

    # Downsample by M.
    out = filtered[::down]

    # Trim/pad to the ideal length so callers get round(n * out/in) +/- 0.
    ideal = int(round(samples.size * out_rate / in_rate))
    if out.size > ideal:
        out = out[:ideal]
    elif out.size < ideal:
        out = np.concatenate([out, np.zeros(ideal - out.size, dtype=np.float64)])
    return out


def _float_to_int16_bytes(samples: npt.NDArray[np.float64]) -> bytes:
    """Clip float samples (already in int16 amplitude domain) to int16 bytes."""
    clipped = np.clip(np.rint(samples), _INT16_MIN, _INT16_MAX)
    return clipped.astype("<i2").tobytes()


def resample_int16(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Resample little-endian mono int16 PCM from ``in_rate`` to ``out_rate``.

    Anti-aliased when downsampling. Output is clipped (never wrapped) to the
    int16 range. ``in_rate == out_rate`` returns equivalent samples.
    """
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    if in_rate == out_rate:
        return _float_to_int16_bytes(samples)
    resampled = _resample_float(samples, in_rate, out_rate)
    return _float_to_int16_bytes(resampled)


def resample_f32_to_int16(samples: npt.NDArray[np.float32], in_rate: int, out_rate: int) -> bytes:
    """Resample float32 mono samples in [-1, 1] to int16 PCM bytes.

    For the VPIO tap path: hardware float32 frames -> resampled int16. Values are
    scaled by 32767, anti-aliased, and clipped (never wrapped) to int16.
    """
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return b""
    scaled = arr * _INT16_MAX
    if in_rate == out_rate:
        return _float_to_int16_bytes(scaled)
    resampled = _resample_float(scaled, in_rate, out_rate)
    return _float_to_int16_bytes(resampled)
