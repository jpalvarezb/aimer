from __future__ import annotations

import numpy as np
import pytest
from duplex_bridge.dsp.framing import Reframer


def _pcm(n: int, start: int = 0) -> bytes:
    return np.arange(start, start + n, dtype="<i2").tobytes()


def test_exact_frame() -> None:
    r = Reframer(160)
    frames = r.push(_pcm(160))
    assert len(frames) == 1
    assert len(frames[0]) == 320
    assert r.flush() == b""


def test_partial_buffers_remainder() -> None:
    r = Reframer(160)
    # 161 samples -> one full frame + 1 buffered.
    frames = r.push(_pcm(161))
    assert len(frames) == 1
    assert len(frames[0]) == 320
    # The 159 missing samples complete the second frame.
    frames2 = r.push(_pcm(159, start=161))
    assert len(frames2) == 1
    assert len(frames2[0]) == 320
    assert r.flush() == b""


def test_frame_contents_contiguous() -> None:
    r = Reframer(160)
    frames = r.push(_pcm(161))
    frames += r.push(_pcm(159, start=161))
    combined = b"".join(frames)
    assert np.frombuffer(combined, dtype="<i2").tolist() == list(range(320))


def test_multiple_frames_in_one_push() -> None:
    r = Reframer(160)
    frames = r.push(_pcm(320))
    assert len(frames) == 2
    assert all(len(f) == 320 for f in frames)


def test_flush_returns_and_empties() -> None:
    r = Reframer(160)
    r.push(_pcm(50))
    rem = r.flush()
    assert len(rem) == 100
    assert r.flush() == b""


def test_empty_push() -> None:
    r = Reframer(160)
    assert r.push(b"") == []


def test_invalid_frame_size() -> None:
    with pytest.raises(ValueError):
        Reframer(0)
