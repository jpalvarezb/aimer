"""Synthetic AEC scenario builders and efficacy metrics.

Everything here is deterministic: signals are built from seeded RNGs
(``np.random.default_rng``) or analytic sines, never from global random state.

Used as the headless test oracle for echo-cancellation efficacy (ERLE on a
far-end-only segment) and near-end preservation during double-talk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000


def make_far_end(n: int, seed: int = 1) -> NDArray[np.float64]:
    """Far-end reference: a band-limited noisy multi-tone (speech-like)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    sig = (
        0.6 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 540.0 * t)
        + 0.15 * np.sin(2 * np.pi * 1300.0 * t)
    )
    sig += 0.05 * rng.standard_normal(n)
    return np.asarray(sig, dtype=np.float64)


def make_near_end(n: int, seed: int = 2) -> NDArray[np.float64]:
    """Near-end speech: a distinct multi-tone + noise, decorrelated from far-end."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    sig = (
        0.5 * np.sin(2 * np.pi * 330.0 * t + 0.7)
        + 0.25 * np.sin(2 * np.pi * 770.0 * t + 1.3)
        + 0.12 * np.sin(2 * np.pi * 1600.0 * t)
    )
    sig += 0.05 * rng.standard_normal(n)
    return np.asarray(sig, dtype=np.float64)


def make_rir(decay: float = 0.5, taps: int = 6, lead_delay: int = 0) -> NDArray[np.float64]:
    """Short linear room-impulse-response: a few exponentially-decaying taps."""
    h = np.zeros(lead_delay + taps, dtype=np.float64)
    for k in range(taps):
        h[lead_delay + k] = (decay**k) * (1.0 if k % 2 == 0 else -0.6)
    return h


def apply_echo_path(
    far: NDArray[np.float64],
    delay_samples: int = 80,
    attenuation: float = 0.4,
    rir: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply an echo path to ``far``: pure delay + attenuation, plus optional RIR.

    Returns an array of the same length as ``far`` (echo as heard at the mic).
    """
    far = np.asarray(far, dtype=np.float64).reshape(-1)
    n = far.shape[0]

    if rir is not None:
        rir = np.asarray(rir, dtype=np.float64).reshape(-1)
        convolved = np.convolve(far, rir)[:n]
    else:
        convolved = far

    echo = np.zeros(n, dtype=np.float64)
    if delay_samples < n:
        echo[delay_samples:] = attenuation * convolved[: n - delay_samples]
    return echo


def compute_erle(
    mic_echo: NDArray[np.float64],
    residual: NDArray[np.float64],
) -> float:
    """Echo Return Loss Enhancement in dB over a far-end-only segment.

    ``10 * log10(mean(mic_echo**2) / mean(residual**2))``. Both arrays should be
    sliced to the same far-end-only sample range by the caller.
    """
    mic_echo = np.asarray(mic_echo, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)

    mic_power = float(np.mean(mic_echo**2))
    res_power = float(np.mean(residual**2))

    # Guard against zero residual (perfect cancellation) and zero input.
    eps = 1e-12
    if res_power < eps:
        res_power = eps
    if mic_power < eps:
        return 0.0
    return 10.0 * float(np.log10(mic_power / res_power))


def compute_double_talk_corr(
    residual: NDArray[np.float64],
    near: NDArray[np.float64],
) -> float:
    """Pearson correlation between residual and near-end over a double-talk segment.

    A high value means near-end speech was preserved (not suppressed) by the AEC.
    """
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    near = np.asarray(near, dtype=np.float64).reshape(-1)

    rs = residual - residual.mean()
    ns = near - near.mean()
    denom = float(np.sqrt(np.dot(rs, rs) * np.dot(ns, ns)))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(rs, ns) / denom)


@dataclass(frozen=True)
class AecScenario:
    """A deterministic AEC scenario with labeled segment ranges.

    All signals share one timeline of length ``total``. ``mic`` is what the AEC
    sees on its near input; ``far`` is the reference. ``echo`` is the echo-only
    component of the mic (useful for measuring ERLE), ``near`` is the near-end
    speech component.

    Segments are half-open ``(start, stop)`` sample ranges:
      - ``far_only``: far-end plays, no near-end speech (echo-only) -> measure ERLE.
      - ``double_talk``: far-end and near-end both active -> measure preservation.
      - ``near_only``: only near-end speech (optional).
    """

    far: NDArray[np.float64]
    near: NDArray[np.float64]
    echo: NDArray[np.float64]
    mic: NDArray[np.float64]
    far_only: tuple[int, int]
    double_talk: tuple[int, int]
    near_only: tuple[int, int]


def build_scenario(
    delay_samples: int = 80,
    attenuation: float = 0.4,
    rir: NDArray[np.float64] | None = None,
    far_only_s: float = 1.0,
    double_talk_s: float = 0.5,
    near_only_s: float = 0.25,
    far_seed: int = 1,
    near_seed: int = 2,
) -> AecScenario:
    """Build a 3-segment scenario: far-only, then double-talk, then near-only.

    The mic is ``echo + near``, where near is zero outside the double-talk and
    near-only segments, and far is zero during the near-only segment.
    """
    n_far_only = int(round(far_only_s * SAMPLE_RATE))
    n_double = int(round(double_talk_s * SAMPLE_RATE))
    n_near_only = int(round(near_only_s * SAMPLE_RATE))
    total = n_far_only + n_double + n_near_only

    far_full = make_far_end(total, seed=far_seed)
    near_full = make_near_end(total, seed=near_seed)

    far_only = (0, n_far_only)
    double_talk = (n_far_only, n_far_only + n_double)
    near_only = (n_far_only + n_double, total)

    # Far-end is silent during the near-only tail.
    far = far_full.copy()
    far[near_only[0] : near_only[1]] = 0.0

    # Near-end is silent during the far-only lead-in.
    near = np.zeros(total, dtype=np.float64)
    near[double_talk[0] : near_only[1]] = near_full[double_talk[0] : near_only[1]]

    echo = apply_echo_path(far, delay_samples=delay_samples, attenuation=attenuation, rir=rir)
    mic = echo + near

    return AecScenario(
        far=far,
        near=near,
        echo=echo,
        mic=mic,
        far_only=far_only,
        double_talk=double_talk,
        near_only=near_only,
    )
