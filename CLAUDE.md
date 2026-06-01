# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Aimer

Aimer is a pointer-grounded, full-duplex assistant. The user points at something on screen and speaks; the system captures cursor-aware visual context at 10 Hz and streams it to a real-time duplex model (currently Gemini Live). Currently macOS-only, targeting Python 3.12+, managed via `uv`.

## Commands

**Install dependencies:**
```bash
uv sync
# Optional audio (requires PortAudio: brew install portaudio)
uv pip install -e "duplex-bridge[audio]"
```

**Run tests:**
```bash
uv run pytest                          # all tests
uv run pytest aimer-core/tests/        # single package
uv run pytest -k test_name             # single test
uv run pytest -m benchmark             # benchmark-only (opt-in, skipped by default)
```

**Lint and type-check:**
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy aimer-core/src pointer-agent/src duplex-bridge/src
```

**Run the system (two terminals):**
```bash
# Terminal 1 — duplex bridge
export GEMINI_API_KEY=...
uv run -m duplex_bridge --host 127.0.0.1 --port 8765

# Terminal 2 — pointer agent
uv run -m pointer_agent --hz 10 --ws-url ws://127.0.0.1:8765/context --log-latency
```

**Smoke / diagnostics (no API key needed):**
```bash
uv run pointer-agent --limit 5                     # 5 packets to stdout
uv run pointer-agent --no-tiles --limit 5          # Week 1 context only
uv run pointer-agent --tiles --output .data/t.jsonl --limit 50  # write JSONL
uv run -m duplex_bridge --no-audio                 # visual-only CI smoke
```

**Low-latency mode (recommended — client-side end-of-turn detection):**
```bash
uv run -m duplex_bridge --manual-vad --thinking-level minimal --end-of-turn-silence-ms 400
```
`--manual-vad` is the lever that actually reduces end-of-speech→audio latency (~1343 ms → ~711 ms floor); see Latency budgets.

**Push-to-talk (snappiest turns; hold a key to speak):**
```bash
uv pip install -e "duplex-bridge[ptt]"   # installs pynput
uv run -m duplex_bridge --push-to-talk --ptt-key cmd_r --thinking-level minimal
```
`--push-to-talk` implies `--manual-vad` and removes the end-of-turn silence wait entirely (true ~711 ms floor, no false mid-pause cutoffs). macOS needs Input Monitoring permission for the global key listener. `--ptt-key` takes a pynput key name (`cmd_r`, `shift_r`, `space`) or a single character.

**Echo cancellation (so the assistant doesn't self-interrupt on its own playback):** `--audio-backend {sounddevice,software-aec,vpio}`. `sounddevice` (default, no AEC — use headphones or PTT). `software-aec` (numpy NLMS, cross-platform, ERLE ~20 dB; needs `duplex-bridge[aec]`). `vpio` (macOS hardware AEC via Voice Processing I/O; needs `duplex-bridge[vpio]`). Pipeline lives behind the `AudioBackend` seam in `duplex_bridge/audio_backends/`; turn-detection/VAD/PTT in `MicCapture` is backend-agnostic. VPIO efficacy is validated manually — see `docs/vpio-smoke-checklist.md` (and the PyObjC gotchas documented there: engine-restart watchdog, import from `AVFoundation` not `AVFAudio`).
```bash
uv pip install -e "duplex-bridge[vpio]"
uv run -m duplex_bridge --audio-backend vpio --thinking-level minimal
```

The automatic-VAD knobs below have **no measurable effect** on the native-audio model and are kept only for A/B record-keeping:
```bash
uv run -m duplex_bridge \
  --turn-coverage activity_only \
  --vad-start-sensitivity high \
  --vad-silence-ms 500 \
  --audio-activity-rms-threshold 300
```
Baseline smoke protocol: start the bridge, wait 3–5 s in silence, say "Hello, respond briefly.", stop after the response, repeat ≥5 times before comparing.

## macOS permissions

Three system permissions are required; without them, capture silently degrades or fails:

- **Accessibility** — selected text and UI labels (`System Settings → Privacy & Security → Accessibility`)
- **Screen Recording** — cursor-settled tiles at native display scale (256×256 pt → 512×512 px on Retina) (`System Settings → Privacy & Security → Screen Recording`); tile capture requires macOS 14 (Sonoma)+, silently no-ops on older versions
- **Microphone** — mic input when running duplex-bridge with audio enabled

## Architecture

```
aimer-core/          Shared Pydantic schema (ContextPacket and sub-models)
pointer-agent/       macOS telemetry harness → emits ContextPacket at 10 Hz
duplex-bridge/       WebSocket server + Gemini Live session boundary
pointer-extension/   Non-functional Chrome MV3 placeholder (ignore for now)
docs/architecture.md Full design spec
```

### Data flow

1. `pointer-agent` captures cursor position (Quartz), focused window (Cocoa/AX), selected text (AX), and a 256-pt JPEG screen tile at native display scale (512×512 px on Retina) (ScreenCaptureKit, ~150 ms settle debounce).
2. Each `ContextPacket` is serialized to JSON and either written to stdout/JSONL or sent over WebSocket.
3. `duplex-bridge` receives packets at `/context`, validates them against `ContextPacket`, and forwards them to `GeminiLiveSession.send_visual_context()`.
4. `GeminiLiveSession` also receives PCM mic frames (`send_audio`) and streams PCM speaker output back via callbacks.

### Visual context send model

Visual context is **cached and injected per turn, never streamed per packet.** Forwarding it on every 10 Hz packet was the original barge-in bug: `send_realtime_input(text=...)` cancels in-progress generation on the native-audio model, so every packet aborted the turn (silence / chopped replies). `GeminiLiveSession.send_visual_context()` now caches the latest `ContextPacket` and:

- **manual VAD / PTT:** during a turn (model idle, awaiting `activity_end`) streams the tile (realtime-video) *and* the text annotation, throttled to ≤1 FPS, so the model tracks what the user points at / selects mid-sentence; force-sends the freshest context at `activity_start`. Between turns it only caches.
- **automatic VAD:** streams the tile only — no idle window, so realtime text could interrupt a live response.

Realtime channels by interrupt behavior: **video never interrupts; text interrupts only while the model is generating** (safe during a turn, never per-packet/mid-response); `send_client_content` is rejected mid-conversation on `gemini-3.1-flash-live-preview`. Validate headlessly with `scripts/diag/repro_interrupt.py` (`--validate-text`, `--validate-text-stream`, `--smoke`).

### Key abstractions

- **`ContextPacket`** (`aimer-core/src/aimer_core/schema.py`) — the single wire format shared by all packages. Uses `extra="forbid"` to prevent packet drift. Screen tiles are `hover_region.tile_b64` (base64 JPEG). Coordinates are logical points; `display_scale` carries the backing scale factor.
- **`DuplexSession`** (`duplex-bridge/src/duplex_bridge/session.py`) — ABC that decouples the bridge from any specific model provider. Implement `open`, `send_audio`, `send_visual_context`, `on_audio_out`, `on_interrupt` (barge-in flush hook; default no-op), `on_tool_call`, `close`.
- **`CaptureProvider`** (`pointer-agent/src/pointer_agent/capture/base.py`) — ABC for platform-specific capture. macOS implementation lives in `capture/macos/`. `PlatformCaptureProvider` in `capture/__init__.py` selects the right implementation at runtime.
- **`WebSocketContextServer`** (`duplex-bridge/src/duplex_bridge/server.py`) — single-client-at-a-time server; session lifecycle is owned by the caller, not the server.

### Optional audio modules

`duplex-bridge/src/duplex_bridge/audio_input.py` (`MicCapture`) and `audio_output.py` (`SpeakerOutput`) are only imported when `--no-audio` is not set and the `sounddevice` extra is installed. `audio_metrics.py` tracks RMS health logs.

## Tooling notes

- `uv` workspace: root `pyproject.toml` declares `members = [aimer-core, pointer-agent, duplex-bridge]`. Each package has its own `pyproject.toml`.
- `ruff` line length is 100; target Python 3.10; rules `E F I UP B SIM`.
- `mypy` is strict; PyObjC modules (`AppKit`, `Quartz`, etc.) and `google.*` have `ignore_missing_imports = true`.
- `pytest-asyncio` is in `asyncio_mode = auto`; mark performance tests with `@pytest.mark.benchmark` — they are skipped by default.
- `--output` and `--ws-url` are mutually exclusive in `pointer-agent`.
- `pointer-agent` reads `AIMER_TELEMETRY_HZ` and `AIMER_TELEMETRY_OUTPUT` env vars as defaults for `--hz` and `--output`.

## Latency budgets

- Tile-to-wire warm path: p95 < 120 ms (observed ~110 ms p95). Original <30 ms target deferred.
- End-of-speech → first audio out: ≤ 700 ms target (Week 3). **Accepted** at p50≈711 ms (within measurement jitter of target) with `--manual-vad --thinking-level minimal`. Authoritative run (10 runs, `models/gemini-3.1-flash-live-preview`): `last_activity→response` p50=711 ms p95=815 ms (range 685–815; 4/10 ≤702 ms). This is the native-audio model + network floor.
  - **Manual VAD is the only lever that moves this.** With Gemini's automatic VAD, `last_activity→response` is stuck at ~1343 ms regardless of `--vad-silence-ms`, `--vad-end-sensitivity`, or `--turn-coverage` — the native-audio model ignores the silence-duration knob. `--manual-vad` disables server VAD and signals end-of-turn explicitly (`activity_end`), removing the server's ~630 ms silence wait. The half-cascade models that honored the VAD knobs (`gemini-2.0-flash-live-001`) are shut down.
  - The 711 ms floor assumes end-of-turn at the true end of speech (explicit / push-to-talk, or an oracle VAD). Continuous-mic capture adds its own client-side end-of-turn silence window (`--end-of-turn-silence-ms`, default 400 ms) on top, since it must observe silence to know the user stopped. Even so, ~400+711 ms beats automatic VAD's ~1343 ms.
  - Re-measure: `uv run python scripts/measure_ttfb.py --manual-vad --thinking-level minimal`. The `scripts/measure_ttfb.py` harness drives `GeminiLiveSession` directly with `say`+ffmpeg speech; `--warmup-turns N` measures warm (steady-state) turns (warm vs cold is negligible for this model).
- Use `--log-latency` on the pointer agent to get rolling p50/p95 per stage (capture, jpeg, base64, packet_build, ws_send, total).
- Bridge logs four split first-audio metrics; `first_audio_out_after_last_audio_activity_ms` (end-of-speech → first audio) is the primary Week 3 acceptance metric. `first_audio_out_after_first_audio_activity_send_ms` is inflated by phrase duration.

## Roadmap

- **Week 1** — macOS pointer telemetry (cursor, window, AX, selected text) at 10 Hz
- **Week 2** — 256×256 cursor-settle tile pipeline; `--log-latency` percentiles
- **Week 3** (accepted) — Gemini Live bridge: audio in + audio out + tile in one session; split first-audio metrics; RMS activity; client-side end-of-turn detection (`--manual-vad`) hitting p50≈711 ms end-of-speech→audio
- **Week 4** — Deictic resolver: "Fix this" / "summarize that" correct on ≥80% of a 50-task deictic eval
- **Week 5** — Entity extraction (DeepMind Principle 4): local VLM (Qwen2.5-VL-7B or Gemini Flash-Lite) emits typed entities from cursor tiles; routes to Maps / Calendar / IDE. Schema stub (`extracted_entities: list[Entity]`) already in `ContextPacket`.
- **Week 6** — Async background worker: tool calls off the hot path; duplex audio never stalls
- **Week 7** — Host app actions (Chrome + IDE): live demos "compare these products" + "rewrite this function async"
- **Week 8** — FD-bench-style eval: interrupt / backchannel / talk-over + custom pointer-deixis suite
- Post-Week-8 — Windows (UI Automation) + Linux (AT-SPI) portability; `DuplexSession` adapter for TML swap
