"""Headless deterministic tests for the NLMS echo canceller and ERLE oracle."""

from __future__ import annotations

import numpy as np
from duplex_bridge.dsp.nlms import NlmsCanceller
from testkit.erle import (
    SAMPLE_RATE,
    build_scenario,
    compute_double_talk_corr,
    compute_erle,
    make_rir,
)

# Discard the first ~0.5 s while the adaptive filter converges.
WARMUP_SAMPLES = SAMPLE_RATE // 2
ERLE_TARGET_DB = 15.0
RIR_ERLE_TARGET_DB = 15.0
DOUBLE_TALK_CORR_TARGET = 0.8


def _run(scenario_kwargs: dict[str, object]) -> tuple[np.ndarray, object]:
    scenario = build_scenario(**scenario_kwargs)  # type: ignore[arg-type]
    aec = NlmsCanceller(filter_len=256, mu=0.5, eps=1e-3)
    residual = aec.process(scenario.mic, scenario.far)
    return residual, scenario


def test_erle_delay_attenuation_far_only() -> None:
    residual, scenario = _run({"delay_samples": 80, "attenuation": 0.4})
    start, stop = scenario.far_only
    seg = slice(start + WARMUP_SAMPLES, stop)
    erle = compute_erle(scenario.echo[seg], residual[seg])
    assert erle >= ERLE_TARGET_DB, f"ERLE {erle:.1f} dB < {ERLE_TARGET_DB} dB"


def test_erle_with_rir_far_only() -> None:
    rir = make_rir(decay=0.5, taps=6, lead_delay=40)
    residual, scenario = _run({"delay_samples": 30, "attenuation": 0.5, "rir": rir})
    start, stop = scenario.far_only
    seg = slice(start + WARMUP_SAMPLES, stop)
    erle = compute_erle(scenario.echo[seg], residual[seg])
    assert erle >= RIR_ERLE_TARGET_DB, f"RIR ERLE {erle:.1f} dB < {RIR_ERLE_TARGET_DB} dB"


def test_double_talk_preserves_near_end() -> None:
    residual, scenario = _run({"delay_samples": 80, "attenuation": 0.4})
    start, stop = scenario.double_talk
    corr = compute_double_talk_corr(residual[start:stop], scenario.near[start:stop])
    assert corr >= DOUBLE_TALK_CORR_TARGET, (
        f"double-talk corr {corr:.3f} < {DOUBLE_TALK_CORR_TARGET}"
    )


def test_zeros_in_zeros_out() -> None:
    aec = NlmsCanceller()
    n = 1000
    residual = aec.process(np.zeros(n), np.zeros(n))
    assert np.allclose(residual, 0.0)


def test_filter_does_not_diverge() -> None:
    aec = NlmsCanceller(filter_len=256, mu=0.5, eps=1e-3)
    rng = np.random.default_rng(7)
    # Long run of loud, partially-correlated input.
    for _ in range(50):
        far = rng.standard_normal(2000)
        near = 0.4 * np.roll(far, 50) + 0.3 * rng.standard_normal(2000)
        aec.process(near, far)
    tap_norm = float(np.linalg.norm(aec.taps))
    assert np.isfinite(tap_norm)
    assert tap_norm < 1.0e3, f"taps diverged: L2 norm {tap_norm}"


def test_streaming_matches_block() -> None:
    """Feeding 10 ms frames must equal one big call (state persists across calls)."""
    scenario = build_scenario(delay_samples=80, attenuation=0.4)
    block_aec = NlmsCanceller(filter_len=256, mu=0.5, eps=1e-3)
    block_res = block_aec.process(scenario.mic, scenario.far)

    frame = SAMPLE_RATE // 100  # 10 ms
    stream_aec = NlmsCanceller(filter_len=256, mu=0.5, eps=1e-3)
    chunks = []
    for i in range(0, scenario.mic.shape[0], frame):
        chunks.append(stream_aec.process(scenario.mic[i : i + frame], scenario.far[i : i + frame]))
    stream_res = np.concatenate(chunks)
    assert np.allclose(block_res, stream_res, atol=1e-9)


def test_passthrough_fails_erle_bar() -> None:
    """Sanity: a trivial passthrough (residual == near) must NOT pass the ERLE bar."""
    scenario = build_scenario(delay_samples=80, attenuation=0.4)
    start, stop = scenario.far_only
    seg = slice(start + WARMUP_SAMPLES, stop)
    # Passthrough residual == mic (== echo on far-only segment).
    passthrough_erle = compute_erle(scenario.echo[seg], scenario.mic[seg])
    assert passthrough_erle < ERLE_TARGET_DB, (
        f"passthrough ERLE {passthrough_erle:.3f} should be well below {ERLE_TARGET_DB} dB"
    )
    # On the far-only segment mic == echo, so ERLE of passthrough is ~0 dB.
    assert abs(passthrough_erle) < 1.0


def test_erle_segment_is_far_end_only() -> None:
    """The far-only segment must contain no near-end energy (never measured in double-talk)."""
    scenario = build_scenario(delay_samples=80, attenuation=0.4)
    start, stop = scenario.far_only
    assert np.all(scenario.near[start:stop] == 0.0)
    # And the double-talk segment must actually have near-end energy.
    ds, de = scenario.double_talk
    assert float(np.mean(scenario.near[ds:de] ** 2)) > 0.0
