# Week 6 — Async background worker: acceptance record

**Status: ACCEPTED.** Long tool calls run off the duplex audio hot path; the ~200 ms audio tick
stays on-cadence under load (measured max lateness **2.3 ms** vs **6142 ms** when the same work
runs inline).

## Criterion

> Week 6 — Async background worker: long tool calls (web, code edits, file I/O, reasoning) run
> off the hot path; the 200 ms duplex audio tick never stalls. Prove it under load.

## The hot path, and how tool calls stalled it

The bridge runs audio on one asyncio event loop. `GeminiLiveSession._recv_loop` pulls model audio
and enqueues it for the speaker (a PortAudio thread drains the queue), and mic frames go out via
`send_audio`. The recv loop **awaits the registered `on_tool_call` callback inline**
(`gemini_live.py`). So a tool callback that takes real time stalls audio two ways:

1. an `async` handler awaited inline blocks the recv loop while it runs; or
2. a **blocking** call on the loop thread (sync HTTP, file I/O, a subprocess for a code edit,
   CPU-bound reasoning) freezes the whole loop.

Either way the recv loop stops pulling model audio → the speaker queue underruns → the assistant's
voice cuts out.

## The fix — `BackgroundWorker` (`duplex_bridge/worker.py`)

`submit()` schedules work and **returns immediately** — it never awaits the job:

- coroutines / async functions → scheduled as loop tasks;
- blocking callables → offloaded to a `ThreadPoolExecutor`.

So the recv loop's tool callback hands off and returns at once; the loop keeps servicing audio.
Results are delivered out-of-band via an `on_result` callback. `ToolDispatcher` registers as the
session's `on_tool_call`, parses the provider tool call (Gemini `function_calls` or a plain
mapping), and submits each function's handler to the worker — returning `None` to honour the
callback contract. This generalises the Week-5 `EntityPipeline.schedule` pattern and **preserves
the `DuplexSession` seam**: the session still just invokes `on_tool_call`; only the consumer side
changed. Wired in `__main__` (`session.on_tool_call(tool_dispatcher.dispatch)`); Week 7 registers
the actual host-action handlers.

> Note on CPU-bound work: the thread pool fully frees the loop for I/O-bound tool calls (web,
> file, subprocess, model API calls — all release the GIL). A genuinely CPU-bound, GIL-holding
> handler should be submitted as a process-backed callable; the `submit` seam supports that
> without changing callers.

## Proof under load

`scripts/bench/bench_worker_hotpath.py` models the hot path as a 200 ms audio-servicing tick on
the event loop, lets it settle, then fires **20 tool calls × 0.3 s** of blocking work (6 s of
work total). Two arms — through the worker (`offload`) vs run inline on the loop (`inline`, the
stall we are preventing). Per-tick lateness = actual interval − 200 ms.

| arm | tick lateness p50 | p95 | **max** | load wall-time |
|---|---|---|---|---|
| **offload (worker)** | 2.2 ms | 2.3 ms | **2.3 ms** | 1.55 s (4-way concurrent) |
| inline (control) | 2.2 ms | 2.4 ms | **6142 ms** | 6.14 s (serial on loop) |

With the worker the 200 ms tick never deviates more than **2.3 ms** despite 6 s of tool work, and
4-way concurrency drains the 20 jobs in 1.55 s. Run the identical work inline and the loop stalls
for **6.1 s** — the control proves the measurement has teeth. Result:
`scripts/bench/results/week6_worker_hotpath.json`.

The same assertion runs in the unit suite
(`test_worker.py::test_tick_survives_load_when_offloaded_and_stalls_when_inline`): offload max
lateness < 60 ms, inline max > 400 ms. 10 worker tests total (submit non-blocking, off-loop
execution, async jobs, error capture, drain, dispatcher routing/parsing).

## Reproduce

```bash
uv run python scripts/bench/bench_worker_hotpath.py     # measured proof + JSON
uv run pytest duplex-bridge/tests/test_worker.py        # unit + in-suite proof
```
