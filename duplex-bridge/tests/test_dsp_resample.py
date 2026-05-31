from __future__ import annotations

import numpy as np
from duplex_bridge.dsp.resample import resample_f32_to_int16, resample_int16


def _sine_int16(freq: float, rate: int, n: int, amp: float = 0.8) -> bytes:
    t = np.arange(n) / rate
    sig = np.sin(2 * np.pi * freq * t) * amp * 32767.0
    return np.rint(sig).astype("<i2").tobytes()


def _to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


def _dominant_freq(samples: np.ndarray, rate: int) -> tuple[float, float]:
    """Return (dominant frequency Hz, bin width Hz)."""
    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / rate)
    idx = int(np.argmax(spectrum))
    return freqs[idx], freqs[1] - freqs[0]


def test_downsample_48k_to_16k_preserves_tone() -> None:
    pcm = _sine_int16(1000.0, 48000, 48000)
    out = resample_int16(pcm, 48000, 16000)
    samples = _to_float(out)
    dom, binw = _dominant_freq(samples, 16000)
    assert abs(dom - 1000.0) <= binw


def test_upsample_24k_to_48k_preserves_tone() -> None:
    pcm = _sine_int16(1000.0, 24000, 24000)
    out = resample_int16(pcm, 24000, 48000)
    samples = _to_float(out)
    dom, binw = _dominant_freq(samples, 48000)
    assert abs(dom - 1000.0) <= binw


def test_44100_to_16000_preserves_tone() -> None:
    pcm = _sine_int16(1000.0, 44100, 44100)
    out = resample_int16(pcm, 44100, 16000)
    samples = _to_float(out)
    dom, binw = _dominant_freq(samples, 16000)
    assert abs(dom - 1000.0) <= binw


def test_amplitude_preserved_within_half_db() -> None:
    amp = 0.8
    pcm = _sine_int16(1000.0, 48000, 48000, amp=amp)
    out = resample_int16(pcm, 48000, 16000)
    samples = _to_float(out)
    # Ignore FIR transient at the edges.
    core = samples[800:-800]
    in_rms = (amp * 32767.0) / np.sqrt(2)
    out_rms = np.sqrt(np.mean(core**2))
    db = 20 * np.log10(out_rms / in_rms)
    assert abs(db) <= 0.5


def test_no_clipping_full_scale() -> None:
    # Near full-scale input must not wrap to large-negative values.
    pcm = _sine_int16(1000.0, 48000, 48000, amp=0.999)
    out = resample_int16(pcm, 48000, 16000)
    samples = np.frombuffer(out, dtype="<i2")
    assert samples.size > 0
    assert np.all(np.abs(samples.astype(np.int32)) < 32768)


def test_anti_aliasing_above_output_nyquist() -> None:
    # 12 kHz tone @ 48 kHz -> 16 kHz (Nyquist 8 kHz). Would alias to 4 kHz
    # without anti-aliasing; the low-pass must strongly attenuate it instead.
    pcm = _sine_int16(12000.0, 48000, 48000, amp=0.8)
    out = resample_int16(pcm, 48000, 16000)
    samples = _to_float(out)
    core = samples[800:-800]
    out_rms = np.sqrt(np.mean(core**2))
    in_rms = (0.8 * 32767.0) / np.sqrt(2)
    atten_db = 20 * np.log10((out_rms + 1e-9) / in_rms)
    # Strongly attenuated (well below input level).
    assert atten_db < -30.0


def test_output_length_matches_ratio() -> None:
    n_in = 4800
    pcm = _sine_int16(1000.0, 48000, n_in)
    out = resample_int16(pcm, 48000, 16000)
    n_out = len(out) // 2
    expected = round(n_in * 16000 / 48000)
    assert abs(n_out - expected) <= 1


def test_zeros_in_zeros_out() -> None:
    pcm = np.zeros(4800, dtype="<i2").tobytes()
    out = resample_int16(pcm, 48000, 16000)
    samples = np.frombuffer(out, dtype="<i2")
    assert np.all(samples == 0)


def test_identity_rate_returns_equivalent_samples() -> None:
    pcm = _sine_int16(1000.0, 16000, 1600)
    out = resample_int16(pcm, 16000, 16000)
    assert _to_float(out).tolist() == _to_float(pcm).tolist()


def test_empty_input() -> None:
    assert resample_int16(b"", 48000, 16000) == b""
    assert resample_f32_to_int16(np.array([], dtype=np.float32), 48000, 16000) == b""


def test_f32_to_int16_preserves_tone_and_scales() -> None:
    t = np.arange(48000) / 48000.0
    f32 = (np.sin(2 * np.pi * 1000.0 * t) * 0.8).astype(np.float32)
    out = resample_f32_to_int16(f32, 48000, 16000)
    samples = _to_float(out)
    dom, binw = _dominant_freq(samples, 16000)
    assert abs(dom - 1000.0) <= binw
    core = samples[800:-800]
    out_rms = np.sqrt(np.mean(core**2))
    expected_rms = (0.8 * 32767.0) / np.sqrt(2)
    db = 20 * np.log10(out_rms / expected_rms)
    assert abs(db) <= 0.5
    assert np.all(np.abs(samples.astype(np.int32)) < 32768)


def test_f32_no_clipping_full_scale() -> None:
    t = np.arange(48000) / 48000.0
    f32 = (np.sin(2 * np.pi * 1000.0 * t) * 1.0).astype(np.float32)
    out = resample_f32_to_int16(f32, 48000, 16000)
    samples = np.frombuffer(out, dtype="<i2")
    assert np.all(np.abs(samples.astype(np.int32)) < 32768)
