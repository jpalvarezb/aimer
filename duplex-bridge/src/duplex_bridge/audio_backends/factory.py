"""Backend selection factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from duplex_bridge.audio_backends.base import AudioBackend, BackendName

if TYPE_CHECKING:
    from duplex_bridge.audio_input import MicCaptureConfig


def make_backend(name: str, config: MicCaptureConfig) -> AudioBackend:
    """Build the audio backend selected by ``name``.

    The microphone capture format is taken from ``config``. Backends that are not
    yet implemented raise ``NotImplementedError`` with a pointer to the milestone.
    """
    resolved = BackendName(name)
    if resolved is BackendName.SOUNDDEVICE:
        from duplex_bridge.audio_backends.sounddevice_backend import SoundDeviceBackend

        return SoundDeviceBackend(
            sample_rate=config.sample_rate,
            channels=config.channels,
            blocksize=config.blocksize,
            dtype=config.dtype,
            device=config.device,
        )
    if resolved is BackendName.SOFTWARE_AEC:
        try:
            from duplex_bridge.audio_backends.software_aec_backend import SoftwareAecBackend
        except ImportError as exc:  # numpy missing
            raise NotImplementedError(
                "the 'software-aec' backend requires numpy; install duplex-bridge[aec]."
            ) from exc

        return SoftwareAecBackend(
            sample_rate=config.sample_rate,
            channels=config.channels,
            blocksize=config.blocksize,
            dtype=config.dtype,
            device=config.device,
        )
    if resolved is BackendName.VPIO:
        try:
            from duplex_bridge.audio_backends.vpio_backend import VpioBackend
        except ImportError as exc:  # pyobjc/numpy missing
            raise NotImplementedError(
                "the 'vpio' backend requires pyobjc; install duplex-bridge[vpio]."
            ) from exc

        return VpioBackend()
    if resolved is BackendName.NATIVE_VPIO:
        from duplex_bridge.audio_backends.native_vpio_backend import NativeVpioBackend

        return NativeVpioBackend()
    raise NotImplementedError(f"unknown audio backend: {name}")
