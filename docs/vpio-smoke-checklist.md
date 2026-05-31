# VPIO backend — status, findings, and the path to speakers-on AEC

Goal: hands-free, speakers-on conversation that does **not** self-interrupt — the
behaviour every production voice app (ChatGPT voice, the Gemini app, etc.) has.
Those apps achieve it with the OS echo canceller (on macOS that is Voice Processing
I/O, "VPIO") driven from **native** code. This backend attempts the same from
Python via PyObjC.

## Current status (macOS 15.6, pyobjc 12.2)

| Half | Status |
|------|--------|
| **Capture** (echo-cancelled mic → Gemini) | ✅ **works** — clean 16 kHz frames, verified live |
| **Playback** (model audio out through the engine) | ❌ **silent from PyObjC** — unresolved |

So `--audio-backend vpio` currently gives echo-cancelled *input* but no audible
*output*. The backend logs an EXPERIMENTAL warning on start to make this explicit.
**For a working assistant today, use headphones with any backend, or
`--audio-backend software-aec` (audible; partial AEC).**

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

## The path to speakers-on AEC: a native audio helper

The robust fix — and how shipping apps do it — is a small **native Swift/Obj-C
helper** that owns the `AVAudioEngine` (VPIO capture **and** playback through an
`AVAudioPlayerNode`/`AVAudioSourceNode`) and exchanges PCM with the Python bridge
over a local socket or stdio:

- Helper: VPIO engine, input tap → 16 kHz mono PCM out to the bridge; PCM in from
  the bridge → player node (echo reference for AEC). All audio buffer handling in
  Swift, where channel-data access and scheduling are first-class (no bridge issues).
- Bridge: a new `AudioBackend` (`native-vpio`) that spawns the helper and pipes
  frames to/from it — slots into the existing seam with no changes to `MicCapture`
  or the session.

This should be built where the Swift can be compiled and the audio heard while
iterating (not patched blind through Python). It is a focused, well-scoped follow-up.

## If revisiting the in-Python playback (lower odds)

One untried approach: an `AVAudioSourceNode` with a pull-based render block reading
from a ring buffer (continuous output, no per-chunk scheduling). Caveat: its render
block runs on the realtime audio thread, so a Python/PyObjC block there contends for
the GIL and may stall — the reason the native helper is preferred.

## Permissions
Mic access works under a bare `uv run` (verified by `scripts/probe_vpio_permission.py`:
authorization `authorized`, live audio captured). A `.app` bundle with
`NSMicrophoneUsageDescription` is only needed for notarized distribution.
