from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from duplex_bridge.audio_backends.software_aec_backend import SoftwareAecBackend
from duplex_bridge.dsp.resample import resample_int16
from testkit.erle import build_scenario, compute_double_talk_corr, compute_erle

_SCALE = 8000.0  # lift [-1, 1] scenario signals into int16 range


def _pcm(arr: np.ndarray) -> bytes:
    return np.clip(np.rint(arr * _SCALE), -32767, 32767).astype("<i2").tobytes()


def _collect_residual(scenario, *, model_rate: int = 16_000) -> tuple[np.ndarray, list[int]]:
    """Run the scenario's mic through the backend's cancellation path, no device."""
    backend = SoftwareAecBackend(model_rate=model_rate)
    frames: list[bytes] = []
    backend._on_frame = frames.append  # type: ignore[assignment]
    # Feed the reference (as model audio) first so far-end aligns with the mic frames.
    backend._on_model_audio(_pcm(scenario.far))
    backend._ingest_mic(_pcm(scenario.mic))
    residual = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float64)
    return residual, [len(f) for f in frames]


def test_software_aec_cancels_synthetic_echo() -> None:
    scenario = build_scenario(delay_samples=80, attenuation=0.4)
    residual, _ = _collect_residual(scenario)

    f0, f1 = scenario.far_only
    warmup = 8_000  # 0.5 s
    echo_seg = scenario.echo[f0 + warmup : f1] * _SCALE
    res_seg = residual[f0 + warmup : f1]

    erle = compute_erle(echo_seg, res_seg)
    # Passthrough (residual == mic == echo on the far-only segment) scores 0 dB;
    # require the canceller to beat that contrast by a wide margin.
    passthrough = compute_erle(echo_seg, echo_seg)
    assert passthrough == pytest.approx(0.0, abs=0.01)
    assert erle >= passthrough + 15.0, f"ERLE {erle:.1f} dB below 15 dB target"


def test_software_aec_cancels_echo_at_24khz_model_rate() -> None:
    """Exercise the production 24 kHz model-audio resample path end-to-end."""
    # Build the echo from the backend's OWN 24 kHz->16 kHz reference so the reference
    # and echo source match exactly, isolating the resample+cancellation behaviour.
    n24 = 24_000  # 1 s of model audio at 24 kHz
    t = np.arange(n24, dtype=np.float64) / 24_000.0
    far24 = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 540 * t)
    far24_bytes = _pcm(far24)
    far16 = np.frombuffer(resample_int16(far24_bytes, 24_000, 16_000), dtype="<i2").astype(
        np.float64
    )
    echo16 = np.zeros_like(far16)
    echo16[80:] = 0.4 * far16[:-80]  # delay 80, attenuation 0.4

    backend = SoftwareAecBackend(model_rate=24_000)
    frames: list[bytes] = []
    backend._on_frame = frames.append  # type: ignore[assignment]
    backend._on_model_audio(far24_bytes)
    backend._ingest_mic(np.clip(np.rint(echo16), -32767, 32767).astype("<i2").tobytes())
    residual = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float64)

    warmup = 8_000
    n = min(residual.shape[0], echo16.shape[0])
    erle = compute_erle(echo16[warmup:n], residual[warmup:n])
    assert erle >= 15.0, f"24 kHz ERLE {erle:.1f} dB below 15 dB target"


def test_software_aec_preserves_near_end_during_double_talk() -> None:
    """Barge-in survives: the residual still carries near-end speech."""
    scenario = build_scenario(delay_samples=80, attenuation=0.4)
    residual, _ = _collect_residual(scenario)

    d0, d1 = scenario.double_talk
    near_seg = scenario.near[d0:d1] * _SCALE
    res_seg = residual[d0:d1]

    # Shape preserved...
    corr = compute_double_talk_corr(res_seg, near_seg)
    assert corr >= 0.8, f"near-end correlation {corr:.2f} below 0.8 (barge-in suppressed)"
    # ...and not linearly attenuated (correlation alone is scale-invariant): the
    # residual power must stay close to the near-end power, not be ducked away.
    power_ratio = float(np.mean(res_seg**2) / np.mean(near_seg**2))
    assert power_ratio >= 0.7, f"near-end power ratio {power_ratio:.2f} too low (suppressed)"
    assert power_ratio <= 2.0, f"near-end power ratio {power_ratio:.2f} too high (echo leak)"


def test_software_aec_emits_100ms_frames() -> None:
    scenario = build_scenario(far_only_s=1.0, double_talk_s=0.0, near_only_s=0.0)
    _, frame_lengths = _collect_residual(scenario)
    assert frame_lengths, "no frames emitted"
    assert all(length == 1600 * 2 for length in frame_lengths)


def test_software_aec_resamples_model_audio_to_reference() -> None:
    """24 kHz model audio is resampled to the 16 kHz reference buffer."""
    backend = SoftwareAecBackend(model_rate=24_000)
    backend._on_model_audio(b"\x10\x00" * 240)  # 240 samples @ 24 kHz = 10 ms
    # 10 ms at 16 kHz ≈ 160 samples in the far buffer.
    assert backend.stats["far_buffer_samples"] == pytest.approx(160, abs=2)


def test_software_aec_zero_pads_on_far_underrun() -> None:
    """With no model audio yet, the reference underruns and the mic passes through."""
    backend = SoftwareAecBackend()
    frames: list[bytes] = []
    backend._on_frame = frames.append  # type: ignore[assignment]

    near = (4000).to_bytes(2, "little", signed=True) * 3200  # 2 frames of 1600, no far-end
    backend._ingest_mic(near)

    residual = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float64)
    assert len(frames) == 2
    # No echo reference → residual is essentially the near-end (passes through).
    near_arr = np.frombuffer(near, dtype="<i2").astype(np.float64)
    assert np.allclose(residual, near_arr, atol=1.0)


@pytest.mark.asyncio
async def test_software_aec_owns_playback_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStream:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self.audio_callback = None

        def on_audio_out(self, callback) -> None:  # noqa: ANN001
            self.audio_callback = callback

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawInputStream=FakeStream, RawOutputStream=FakeStream),
    )

    backend = SoftwareAecBackend()
    session = FakeSession()
    ok = await backend.start(lambda _f: None, session, asyncio.get_running_loop())
    assert ok
    assert session.audio_callback is not None  # backend owns the playback subscription
    assert backend.capture_config.frame_samples == 1600
    await backend.stop()
