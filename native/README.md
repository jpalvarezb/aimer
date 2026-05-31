# aimer-vpio-helper

A small native macOS helper that owns the Voice Processing I/O (VPIO)
`AVAudioEngine` — microphone capture **and** speaker playback — and exchanges raw
PCM with the Python `duplex-bridge` over its stdio. It backs the bridge's
`--audio-backend native-vpio`, the recommended speakers-on path: hardware acoustic
echo cancellation so the assistant does not self-interrupt on its own voice.

Why native: the PyObjC `vpio` backend echo-cancels capture but its playback through
the player node is silent (driving CoreAudio's render path through PyObjC is the
wall — see `../docs/vpio-backend-status.md`). In Swift, audio buffer scheduling and
channel-data access are first-class and ARC retains scheduled buffers, so playback
works.

## Build

Requires Swift 6 / Xcode (macOS 14+).

```sh
cd native
swift build -c release
# binary at .build/release/aimer-vpio-helper
```

Or from the repo root: `just build-native`.

## Run

The bridge spawns the helper automatically once it is built — it resolves the
binary via `$AIMER_VPIO_HELPER`, then `native/.build/release/aimer-vpio-helper`,
then `PATH`:

```sh
just build-native
uv run -m duplex_bridge --audio-backend native-vpio --thinking-level minimal
```

Sanity-check capture on-device (runs the engine ~2 s, prints frame count + peak to
stderr, exits non-zero if no frames were captured):

```sh
.build/release/aimer-vpio-helper --selftest
```

## Wire protocol

Length-prefixed binary PCM over stdio; **stderr is logs only**.

| Direction | Framing | Audio |
|-----------|---------|-------|
| bridge → helper **stdin** (model audio) | `[4-byte LE uint32 N][N bytes]` | 24 kHz mono int16 |
| helper → bridge **stdout** (echo-cancelled mic) | `[4-byte LE uint32 N][N bytes]` | 16 kHz mono int16, fixed **N = 3200** (100 ms) |

The fixed 3200-byte capture frame matches the bridge's `CaptureFormat(16000, 1, 1600)`
and VAD cadence. The Python side of this contract lives in
`../duplex-bridge/src/duplex_bridge/audio_backends/native_vpio_backend.py`; the
constants are mirrored in `Sources/AimerVPIOHelper/Protocol.swift`.

## Design notes

- **Echo reference:** the `AVAudioPlayerNode` is connected to the **output node**
  (not the main mixer — VPIO forces the output to the input's rate). Playing model
  audio through it gives VPIO its reference signal to cancel from the mic.
- **Configuration-change restart:** `setVoiceProcessingEnabled(true)` fires an
  `AVAudioEngineConfigurationChange` that stops the engine; an observer restarts it
  and re-arms the player. Without this the input tap never fires.
- **Realtime safety:** the input tap converts to 16 kHz int16 and hands bytes to a
  serial I/O queue; all stdout writes happen there, never on the audio thread. The
  stdin reader runs on its own thread and only schedules buffers.
- **Lifecycle:** stdin EOF (bridge exited) → the helper stops the engine, disables
  voice processing, and exits, so no orphan holds the microphone.
- **Permissions:** the embedded `Info.plist` carries `CFBundleIdentifier` +
  `NSMicrophoneUsageDescription` (a bundle id is required for VPIO). Mic access is
  inherited from the parent process under a bare `uv run`; a notarized `.app` bundle
  is only needed for distribution.
