# VPIO backend — status and findings

Goal: hands-free, speakers-on conversation that does **not** self-interrupt — the
behaviour every production voice app (ChatGPT voice, the Gemini app, etc.) has.
Those apps achieve it with the OS echo canceller (on macOS that is Voice Processing
I/O, "VPIO") driven from **native** code.

There are two VPIO backends:

| Backend | Capture | Playback | Recommendation |
|---------|---------|----------|----------------|
| **`vpio`** (PyObjC, in-process) | ✅ works | ✅ works (ear-verified on-device 2026-07-03) | **The working speakers-on path.** `--audio-backend vpio`, zero build steps. |
| `native-vpio` (Swift helper, `native/`) | ✅ by design | ✅ by design (no on-device run yet) | Robust fallback if in-process playback is silent on your setup. Build with `just build-native`. |

`native-vpio` moves the whole VPIO `AVAudioEngine` (capture **and** playback) into a
small native subprocess (`native/`, see its README) that the bridge spawns and
exchanges PCM with over stdio. It was written when in-process playback was believed
unfixable (history below); the shipped `vpio` backend later proved audible on-device,
so the helper is now the fallback, pending its own first on-device run.

## History: the PyObjC playback silence, and how it got fixed

**Resolved.** During development (macOS 15.6, pyobjc 12.2), `--audio-backend vpio` gave
echo-cancelled *input* but no audible *output*, and the backend shipped with an
EXPERIMENTAL "you will not hear responses" warning. The final combination of fixes in
`vpio_backend.py` — buffer-retention deque (PyObjC doesn't retain scheduled buffers),
~200 ms coalescing (the player drops out on rapid tiny buffers), and the watchdog
re-arming `player.play()` after every engine restart — actually cured it, but all the
fixes and the stale warning landed in one squashed commit (3138693) and nobody
re-ear-tested until the 2026-07-03 live sessions: playback is audible and the
echo-cancelled mic stops the model from self-interrupting on its own voice.

## What works

VPIO capture is fully working: enabling voice processing + an input tap delivers a
clean, echo-cancelled mic stream. If the model never hears its own voice, it does
not self-interrupt — and with in-process playback confirmed audible, `vpio` covers
both directions with no build step.

Mid-development observations kept for reference (all pre-fix): a **single buffer**
scheduled on an `AVAudioPlayerNode` played audibly (440 Hz beep, active input tap),
while streamed model audio stayed silent despite non-silent buffers held alive,
`player.isPlaying() == True`, and no exceptions — until retention + coalescing +
watchdog re-arm were combined.

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

## The native audio helper (implemented as `native-vpio`)

The belt-and-braces alternative — and how shipping apps do it — is a small **native
Swift helper** that owns the `AVAudioEngine` (VPIO capture **and** playback through an
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

- **`vpio` (in-process)** 2026-07-03, macOS 15.6 / pyobjc 12.2: responses audible on
  speakers, no self-interruption on the model's own voice across two live sessions
  (the sounddevice baseline self-interrupted constantly in the same setup). Barge-in
  flush observed working in the bridge log.
- **`native-vpio`**: _(pending first on-device run)_

## If revisiting the in-Python playback (lower odds)

One untried approach: an `AVAudioSourceNode` with a pull-based render block reading
from a ring buffer (continuous output, no per-chunk scheduling). Caveat: its render
block runs on the realtime audio thread, so a Python/PyObjC block there contends for
the GIL and may stall — the reason the native helper is preferred.

## Permissions
Mic access works under a bare `uv run` (verified by `scripts/diag/probe_vpio_permission.py`:
authorization `authorized`, live audio captured). A `.app` bundle with
`NSMicrophoneUsageDescription` is only needed for notarized distribution.
