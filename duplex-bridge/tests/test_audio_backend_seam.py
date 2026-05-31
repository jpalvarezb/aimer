from __future__ import annotations

import pytest
from duplex_bridge.audio_backends import BackendName, make_backend
from duplex_bridge.audio_backends.sounddevice_backend import SoundDeviceBackend
from duplex_bridge.audio_input import MicCaptureConfig


def test_factory_selects_sounddevice() -> None:
    backend = make_backend("sounddevice", MicCaptureConfig())
    assert isinstance(backend, SoundDeviceBackend)
    assert backend.capture_config.frame_samples == 1_600
    assert backend.capture_config.sample_rate == 16_000


def test_factory_passes_capture_format_from_config() -> None:
    backend = make_backend("sounddevice", MicCaptureConfig(blocksize=320, sample_rate=16_000))
    assert backend.capture_config.frame_samples == 320


def test_factory_selects_software_aec() -> None:
    from duplex_bridge.audio_backends.software_aec_backend import SoftwareAecBackend

    backend = make_backend("software-aec", MicCaptureConfig())
    assert isinstance(backend, SoftwareAecBackend)
    assert backend.capture_config.frame_samples == 1600


def test_factory_selects_vpio() -> None:
    pytest.importorskip("AVFoundation")
    from duplex_bridge.audio_backends.vpio_backend import VpioBackend

    backend = make_backend("vpio", MicCaptureConfig())
    assert isinstance(backend, VpioBackend)
    assert backend.capture_config.frame_samples == 1600


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        make_backend("bogus", MicCaptureConfig())


def test_backend_name_values() -> None:
    assert {b.value for b in BackendName} == {
        "sounddevice",
        "software-aec",
        "vpio",
        "native-vpio",
    }
