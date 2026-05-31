"""Probe whether macOS Voice Processing I/O can capture the mic under bare `uv run`.

M4 step 4a (blocking spike): determines whether a plain `uv run` process gets real
microphone access, or whether a minimal .app bundle (with NSMicrophoneUsageDescription)
is required. Reports the TCC authorization status, the VPIO-negotiated input format,
and whether live (non-silent) audio frames arrive.

Run:  uv run python scripts/probe_vpio_permission.py

Exit code: 0 = live audio captured; 2 = silent/denied; 1 = error / unavailable.
"""

from __future__ import annotations

import time

try:  # pyobjc 12 split AVFAudio into its own top-level module
    from AVFAudio import AVAudioEngine
except ImportError:  # pragma: no cover - pyobjc < 12
    try:
        from AVFoundation import AVAudioEngine
    except ImportError:
        AVAudioEngine = None  # type: ignore[assignment, misc]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


_AUTH_STATUS = {0: "notDetermined", 1: "restricted", 2: "denied", 3: "authorized"}


def _check_authorization() -> str:
    """Report and best-effort request microphone TCC authorization."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    except ImportError:
        return "unknown (AVCaptureDevice unavailable)"

    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
    label = _AUTH_STATUS.get(int(status), str(status))
    if int(status) == 0:  # notDetermined -> trigger the consent dialog
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio, lambda _granted: None
        )
        time.sleep(1.0)  # give the dialog a moment
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        label = _AUTH_STATUS.get(int(status), str(status)) + " (after request)"
    return label


def _peak_amplitude(buf: object, peaks: list[float]) -> None:
    """Append the peak |sample| of an AVAudioPCMBuffer to ``peaks`` (best-effort)."""
    try:
        n = int(buf.frameLength())  # type: ignore[attr-defined]
        if n <= 0:
            return
        fcd = buf.floatChannelData()  # type: ignore[attr-defined]
        if fcd is None or np is None:
            peaks.append(-1.0)  # buffer delivered but amplitude unreadable
            return
        mv = fcd[0].as_buffer(n * 4)
        samples = np.frombuffer(mv, dtype=np.float32, count=n)
        peaks.append(float(np.abs(samples).max()))
    except Exception:
        peaks.append(-1.0)


def main() -> int:
    if AVAudioEngine is None:
        print("ERROR: AVFoundation/AVFAudio unavailable; install duplex-bridge[vpio]")
        return 1

    print(f"microphone authorization: {_check_authorization()}")

    engine = AVAudioEngine.alloc().init()
    input_node = engine.inputNode()

    ok, err = input_node.setVoiceProcessingEnabled_error_(True, None)
    if not ok:
        print(f"ERROR: setVoiceProcessingEnabled failed: {err}")
        return 1
    print("voice processing (AEC): enabled")

    fmt = input_node.outputFormatForBus_(0)
    print(
        f"VPIO input format: {fmt.sampleRate():.0f} Hz, {fmt.channelCount()} ch, "
        f"commonFormat={fmt.commonFormat()}"
    )

    peaks: list[float] = []
    input_node.installTapOnBus_bufferSize_format_block_(
        0, 1024, fmt, lambda buf, when: _peak_amplitude(buf, peaks)
    )

    started, err = engine.startAndReturnError_(None)
    if not started:
        print(f"ERROR: engine start failed: {err}")
        return 1

    print("capturing for 1.5 s ...")
    time.sleep(1.5)
    engine.stop()
    input_node.removeTapOnBus_(0)

    buffers = len(peaks)
    readable = [p for p in peaks if p >= 0.0]
    max_peak = max(readable) if readable else -1.0
    print(f"buffers received: {buffers}; max peak amplitude: {max_peak:.5f}")

    if buffers == 0:
        print("VERDICT: no audio delivered — VPIO unavailable under bare `uv run`.")
        return 2
    if max_peak < 0.0:
        print("VERDICT: audio frames delivered but amplitude unreadable; treat as INCONCLUSIVE.")
        return 2
    if max_peak < 1e-5:
        print("VERDICT: SILENT — frames are all-zero (mic permission denied to this process).")
        print("  -> a minimal .app bundle with NSMicrophoneUsageDescription is required.")
        return 2
    print("VERDICT: LIVE audio captured — bare `uv run` has mic access; no bundle needed for dev.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
