# Week 8 — FD-bench-style eval: acceptance record

**Status: ACCEPTED.** A local rerun of the core full-duplex behaviors (interrupt / backchannel /
talk-over / turn-taking) scores **5/5**, alongside the custom pointer-deixis suite at **87.1%**.

## Criterion

> Week 8 — FD-bench-style eval: local rerun of interrupt / backchannel / talk-over plus the
> custom pointer-deixis suite; report scores.

## Method

`scripts/bench/eval_fd.py` scores the full-duplex behaviors **deterministically** by driving the
real bridge components with synthetic audio frames and model events — no live audio or API needed,
so it is repeatable and gated in the suite (`duplex-bridge/tests/test_fd.py`). 20 ms PCM frames at
RMS 1200 (voiced) / 0 (silence) feed the actual `MicCapture` turn-detection machine, and a real
`SpeakerOutput` + `GeminiLiveSession` exercise barge-in.

## Scores

**Full-duplex behaviors — 5/5** (`scripts/bench/results/week8_fd_eval.json`):

| behavior | scenario | result |
|---|---|---|
| **backchannel** | 100 ms "mhm" (< 250 ms onset window) | no turn opened (`turn_starts=0`) ✓ |
| **talk-over** | 280 ms sustained speech | exactly one turn opened (`turn_starts=1`) ✓ |
| **turn-taking** | speech + 420 ms trailing silence | turn closed (`activity_end` fired) ✓ |
| **interrupt** | barge-in flag during playback | buffered audio flushed (`5 → 0`) ✓ |
| **interrupt detection** | model message flag | read correctly (`yes=True, no=False`) ✓ |

The backchannel-vs-talk-over distinction is the onset debounce (`onset_speech_ms`, default 250 ms):
a short vocalization is held in a look-back buffer and discarded on the next sub-threshold frame
(no turn), while sustained speech clears the window and opens a turn. Interrupt handling is the
barge-in path: `GeminiLiveSession._recv_loop` reads the model's `interrupted` flag →
`_dispatch_interrupt` → the registered `on_interrupt` callback → `SpeakerOutput.flush()` drops
buffered playback so the assistant stops promptly.

**Pointer-deixis suite — 87.1%** — the custom 58-task deictic eval from Week 4 (production-faithful
config; two runs 86.2% / 87.9%, 3-vote sonnet-4-6 judge, full hand-verification). See
`docs/week4-deictic-acceptance.md`. (Re-running it live requires `GEMINI_API_KEY`; the accepted,
committed score is cited here for the combined report.)

## What this does / doesn't cover

Covers the *bridge's* full-duplex turn-taking + barge-in logic — the parts that are deterministic
and ours. It does not score the native-audio model's own conversational quality (that's the model,
and end-of-speech→audio latency is the Week-3 metric at p50 ≈ 711 ms). A fully live FD-bench run
(real speech overlapping real model audio) needs `GEMINI_API_KEY` + a mic; the behaviors it would
exercise are the same ones gated deterministically here.

## Reproduce

```bash
uv run python scripts/bench/eval_fd.py            # report + JSON
uv run pytest duplex-bridge/tests/test_fd.py      # gated in the suite
```
