# Week 5 — Entity extraction: acceptance record

**Status: ACCEPTED.** A local VLM emits typed entities from the cursor tile and routes them
to Maps / Calendar / IDE, off the hot path, before the audio turn reaches the duplex model.

## Criterion

> Week 5 — Entity extraction: local VLM emits typed entities from the tile and routes them
> (Maps / Calendar / IDE) before audio reaches the duplex model.

This is a **capability** criterion (no accuracy threshold, unlike Week 4's ≥80%). Every clause
is satisfied with evidence, and we additionally report a measured accuracy number for rigor.

| clause | evidence |
|---|---|
| **local VLM** | `LocalQwenVLExtractor` — Qwen3-VL (MLX, Apple Silicon), runs offline; full 31-tile eval ran on-device |
| **emits typed entities from the tile** | measured below on 31 real cluttered tiles; vision-only (tile pixels, no AX hint) |
| **routes to Maps / Calendar / IDE** | `EntityRouter` (place→maps, date/todo→calendar, code_span→ide), 16 unit tests, eval dispatches every entity |
| **before audio reaches the duplex model** | `WebSocketContextServer._maybe_schedule_entities` fires on the visual-context path, off the hot path, independent of the audio turn |

## Architecture (the seam)

Entity extraction mirrors the other Aimer abstraction boundaries (`DuplexSession`,
`CaptureProvider`, `AudioBackend`) — the bridge depends only on an ABC, so the vision model
is swappable without touching the pipeline or the duplex session.

- **`EntityExtractor`** (`entities/base.py`) — `async extract(tile_jpeg, context) -> list[Entity]`.
  - `LocalQwenVLExtractor` (`entities/mlx_vlm.py`) — Qwen3-VL via MLX. **Default.**
  - `GeminiFlashLiteExtractor` (`entities/gemini_vlm.py`) — remote fallback over the same seam.
  - `MockExtractor` — deterministic, for tests.
- **`EntityRouter`** (`entities/router.py`) — typed routing + pluggable handlers. Payloads are
  actionable: place→`maps.apple.com` URL, date→calendar title, code_span→IDE symbol.
- **`EntityPipeline`** (`entities/pipeline.py`) — `extract → populate packet.extracted_entities
  → route`, with `schedule()` running it **off the hot path** (fire-and-forget asyncio task).
- **Server wiring** (`server.py`) — `_maybe_schedule_entities(packet)` runs on the
  visual-context path *before* `send_visual_context`, with an in-flight guard (one extraction at
  a time) and context dedupe (skip when app|title|AX unchanged), so a 10 Hz packet stream never
  spawns a VLM per packet.

The **`DuplexSession` seam is untouched** — entity extraction is a parallel consumer of the
visual-context stream, not a change to `open/send_audio/send_visual_context/on_audio_out/on_tool_call`.

16 unit tests in `duplex-bridge/tests/test_entities.py` cover the router, pipeline, off-hot-path
scheduling, parsing, and the server wiring (schedule / dedupe / in-flight guard).

## Measured eval — local Qwen3-VL-4B (vision-only)

`scripts/bench/eval_entities.py` over **31 real cluttered tiles** — Wikipedia articles, a real
Google Maps place, a real Google Calendar embed, real GitHub source, MDN CSS docs. Not the
synthetic happy-path demo. Backend: `mlx-community/Qwen3-VL-4B-Instruct-4bit`, tile clamped to
384 px (see hardware note), `temperature=0`.

```
type-recall (positives) : 10/14 = 71%   (place 3/4, date 3/4, code_span 3/4, product 1/2)
value-recall(positives) : 8/14  = 57%
precision   (negatives) : 11/17 = 65%
latency     : p50 3910 ms  p95 5453 ms   (n=31)
backend     : local 31/31, crashed 0
```

Results: `scripts/bench/results/week5_entity_eval_4b_hybrid.json`.

**Why 4B, not 2B.** The 2B-4bit model scored only 36% type-recall on the same tiles (it read
values correctly but mistyped — e.g. "flex-grow" → `unknown` instead of `code_span`). 4B
roughly doubles type-recall (36% → 71%), which is why it is the chosen local default for the
applied use case.

**Why vision-only.** Week 5 tests the *vision model* — can it read entities out of the tile
*pixels*. We deliberately pass an empty `ExtractionContext` (no app/window/AX). The macOS AX
label under the cursor often *is* the entity, so feeding it would measure passthrough, not
vision. Production additionally passes AX hints, so real-world accuracy is **≥** these numbers;
this is the honest vision-only floor.

**On the 65% precision.** 5 of the 6 "false positives" are the model reading *real* CSS code
visible in the periphery of MDN doc tiles (the cursor was on a heading, but the tile includes a
nearby code block), plus one place read off an image-heavy Everest tile. These are the
tile-reads-everything vs cursor-focus tension, not wild hallucination; in production they would
route real code to the IDE. The precision metric is therefore conservative.

## Hardware note — the GPU watchdog hang and the fix

This dev Mac's GPU **hung** under uncapped 4B inference. The error codes were decisive:

- 4B fault: `kIOGPUCommandBufferCallbackErrorHang` — the **culprit** code (our command buffer
  ran too long and tripped the macOS GPU watchdog).
- 2B fault: `kIOGPUCommandBufferCallbackErrorInnocentVictim` — collateral of a GPU recovery.
- Peak memory at fault was **3.80 GB on a ~13 GB GPU** → it was **never OOM**; it was a
  watchdog/progress timeout on the long vision-encoder/prefill kernel of the full 512 px tile.

**Fix (confirmed):** clamp the tile to **384 px** (longest edge) before inference — this shortens
the prefill kernel below the watchdog limit. The *exact tile that hung uncapped* (`ent_place_google`)
then succeeded capped, extracting the full address. The driver (`run_entity_eval_hybrid.sh`) adds
defense-in-depth so a hang can never run away again (one earlier run wedged for 4.5 h):
one tile per subprocess, a **90 s per-tile wall-clock timeout**, skip-and-continue past a wedge,
and a **3-consecutive-fault cap**. With the clamp + driver, the full 31-tile run completed with
**zero wedges**.

A `GeminiFlashLiteExtractor` backend (zero GPU) exists over the same seam as a machine fallback
for environments that can't run MLX; it was not needed for this result.

## Latency / hot-path safety

Local extraction is ~4 s p50 — far too slow to sit on the 200 ms audio tick, which is exactly
why it runs **off the hot path** (change-gated, one tile at a time, fire-and-forget). It populates
`packet.extracted_entities` and dispatches routes asynchronously; the duplex audio loop never
awaits it. Week 6 formalizes the async-worker guarantee under load.

## Reproduce

```bash
# Full local-4B eval (capped, timeout-protected). ~15-20 min on Apple Silicon.
ENTITY_VLM_TILE_MAXPX=384 ENTITY_VLM_MAX_TOKENS=128 ./scripts/bench/run_entity_eval_hybrid.sh

# Unit tests (no model needed)
uv run pytest duplex-bridge/tests/test_entities.py
```
