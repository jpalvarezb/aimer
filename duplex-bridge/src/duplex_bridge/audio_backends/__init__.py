"""Audio backends behind the capture pipeline.

Importing this package is side-effect free: heavy/optional dependencies
(sounddevice, pyobjc, numpy) are imported lazily inside each backend's ``start``.
"""

from __future__ import annotations

from duplex_bridge.audio_backends.base import (
    AudioBackend,
    BackendName,
    CaptureFormat,
    OnFrame,
)
from duplex_bridge.audio_backends.factory import make_backend

__all__ = [
    "AudioBackend",
    "BackendName",
    "CaptureFormat",
    "OnFrame",
    "make_backend",
]
