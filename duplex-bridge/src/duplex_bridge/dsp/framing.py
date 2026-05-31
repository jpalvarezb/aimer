"""Reframe arbitrary-length int16 PCM into fixed-size frames.

The AEC/VPIO pipelines consume fixed frame sizes (e.g. 160 samples == 10 ms @
16 kHz). Upstream producers deliver irregular chunk sizes, so :class:`Reframer`
buffers the remainder between pushes.
"""

from __future__ import annotations

_BYTES_PER_SAMPLE = 2


class Reframer:
    """Accumulate int16 PCM bytes and emit fixed-size frames.

    Each frame is ``frame_samples`` int16 samples (``frame_samples * 2`` bytes).
    Bytes that do not complete a frame are buffered for the next :meth:`push`.
    """

    def __init__(self, frame_samples: int) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        self._frame_samples = frame_samples
        self._frame_bytes = frame_samples * _BYTES_PER_SAMPLE
        self._buf = bytearray()

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def push(self, pcm: bytes) -> list[bytes]:
        """Append ``pcm`` and return zero or more complete frames."""
        if pcm:
            self._buf.extend(pcm)
        frames: list[bytes] = []
        while len(self._buf) >= self._frame_bytes:
            frames.append(bytes(self._buf[: self._frame_bytes]))
            del self._buf[: self._frame_bytes]
        return frames

    def flush(self) -> bytes:
        """Return and clear any buffered remainder (may be partial)."""
        remainder = bytes(self._buf)
        self._buf.clear()
        return remainder
