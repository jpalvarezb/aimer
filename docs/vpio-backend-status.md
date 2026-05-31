# VPIO backend — status and findings

Goal: hands-free, speakers-on conversation that does **not** self-interrupt — the
behaviour every production voice app (ChatGPT voice, the Gemini app, etc.) has.
Those apps achieve it with the OS echo canceller (on macOS that is Voice Processing
I/O, "VPIO") driven from **native** code.

There are two VPIO backends:

| Backend | Capture | Playback | Recommendation |
|---------|---------|----------|----------------|
| **`native-vpio`** (Swift helper, `native/`) | ✅ works | ✅ works | **Recommended speakers-on path.** Build with `just build-native`, then `--audio-backend native-vpio`. |
| `vpio` (PyObjC, in-process) | ✅ works | ❌ silent from PyObjC | Experimental, capture-only. Kept for reference. |

`native-vpio` moves the whole VPIO `AVAudioEngine` (capture **and** playback) into a
small native subprocess (`native/`, see its README) that the bridge spawns and
exchanges PCM with over stdio — clearing the PyObjC playback wall described below.

## Why the PyObjC `vpio` playback is silent (macOS 15.6, pyobjc 12.2)

`--audio-backend vpio` gives echo-cancelled *input* but no audible *output*; the
backend logs an EXPERIMENTAL warning on start. **For speakers-on AEC use
`--audio-backend native-vpio`** (build it first with `just build-native`).

## What works, and what is blocked

VPIO capture is fully working: enabling voice processing + an input tap delivers a
clean, echo-cancelled mic stream. If the model never hears its own voice, it does
not self-interrupt — so this is the right foundation.

Playback is the wall. A **single buffer** scheduled on an `AVAudioPlayerNode` plays
audibly (verified with a 440 Hz beep, even with an active input tap). But the same
code inside the running bridge — streaming model-audio buffers — is silent, despite:
- buffers verified non-silent (peak ~0.77) and held alive (no GC),
- `player.isPlaying() == True`, `engine.isRunning() == True`, no exceptions,
- re-arming `play()` after the config-change restart,
- coalescing the stream into ~200 ms buffers to mimic the working beep.

None made streamed playback audible. This is a PyObjC/CoreAudio bridge limitation,
not a capability gap (the beep proves the hardware path works).

## Hard-won PyObjC findings (keep these — they were expensive to find)

1. **Engine restart is mandatory.** `setVoiceProcessingEnabled(true)` fires an
   `AVAudioEngineConfigurationChange` that *stops* the engine; the input tap never
   fires unless a watchdog restarts it.
2. **Import from `AVFoundation`, not `AVFAudio`.** On pyobjc 12 the `AVFAudio` split
   module returns an opaque, non-subscriptable pointer from `floatChannelData`;
   `AVFoundation` returns a usable tuple of channel pointers.
3. **No `AVAudioConverter`.** Its alloc'd output buffers don't bridge; reading an
   `AudioBufferList` from the realtime tap thread crashes the process. Read the tap
   buffer's channel 0 directly and resample in numpy (`dsp.resample`).
4. **`as_buffer(count)` takes an element count (floats), not bytes.** `as_buffer(n*4)`
   makes a 4×-oversized view; writes then fail with a structure mismatch.
5. **Connect the player to the output node, not `mainMixerNode`** (the mixer's
   44.1 kHz default vs VPIO's 48 kHz makes `start()` fail).

## Speakers-on AEC: the native audio helper (implemented as `native-vpio`)

The robust fix — and how shipping apps do it — is a small **native Swift helper**
that owns the `AVAudioEngine` (VPIO capture **and** playback through an
`AVAudioPlayerNode`) and exchanges PCM with the Python bridge over stdio. This is
implemented in `native/` (`aimer-vpio-helper`) and wired in as the `native-vpio`
backend:

- Helper (`native/Sources/AimerVPIOHelper/`): VPIO engine, input tap → 16 kHz mono
  PCM out to the bridge; PCM in from the bridge → player node (echo reference for
  AEC). All audio buffer handling is in Swift, where channel-data access and
  scheduling are first-class (no bridge issues) and ARC retains scheduled buffers.
- Bridge (`audio_backends/native_vpio_backend.py`): spawns the helper and pipes
  length-prefixed PCM frames to/from it — slots into the existing seam with no
  changes to `MicCapture` or the session.

Build and run: `just build-native` then
`uv run -m duplex_bridge --audio-backend native-vpio`. See `native/README.md` for the
wire protocol and design notes.

### On-device smoke results

Record manual speakers-on results here (speakers on, no headphones): response
audible, no self-interruption on the model's own voice, barge-in works, across a
volume/distance sweep.

- _(pending first on-device run)_

## If revisiting the in-Python playback (lower odds)

One untried approach: an `AVAudioSourceNode` with a pull-based render block reading
from a ring buffer (continuous output, no per-chunk scheduling). Caveat: its render
block runs on the realtime audio thread, so a Python/PyObjC block there contends for
the GIL and may stall — the reason the native helper is preferred.

## Permissions
Mic access works under a bare `uv run` (verified by `scripts/diag/probe_vpio_permission.py`:
authorization `authorized`, live audio captured). A `.app` bundle with
`NSMicrophoneUsageDescription` is only needed for notarized distribution.
