# Week 9 — Decoupled deixis + delegated action stack

**Status: built + verified headlessly; live voice smoke is user-run; decoupled eval number
below.** This week turns Aimer from "a voice assistant with a few bespoke tools" into the
full architecture the `experiment/decoupled-deixis` branch was created for: the realtime
model stays a conversational front-end, and everything heavy — reading the screen,
multi-step action, dangerous commands — runs on decoupled components off the audio hot path.

```
                   ┌──────────── voice ────────────┐
 user ⇄ mic/speaker ⇄  Gemini Live (realtime)      │   pointer-agent (10 Hz packets)
                   │   • tiles keep streaming       │        │
                   │   • pointer= referent annotation◀── PointerReferentResolver
                   │   • delegate_task / check_tasks│        (Flash-Lite, resolve-on-settle)
                   │     confirm_task (NON_BLOCKING)│
                   └──────┬─────────▲───────────────┘
                          │ ack     │ FunctionResponse (WHEN_IDLE — never barges in)
                          ▼         │
                    ToolDispatcher → BackgroundWorker (Week 6)
                          │
                          ▼
                    TaskManager ── N concurrent DelegateTasks
                          │
                    DelegateAgent (gemini-3.5-flash, async Interactions)
                     ├─ run_shell        (allowlist; else voice confirmation)
                     ├─ run_applescript  (allowlist; else voice confirmation)
                     ├─ browser_*        (persistent Chromium, page per task)
                     └─ computer_use     (Week-7b executor + GeminiComputerUsePolicy,
                                          desktop mutex — one task drives the mouse)
```

## What shipped (each independently landed + tested)

1. **Tool results flow back** (`worker.py`, `gemini_live.py`, `session.py`) — the
   foundational gap: dispatch used to drop `fc.id` and never answer the model. Now every
   finished job with a call id becomes a `FunctionResponse` (WHEN_IDLE, so completions wait
   for a speech gap); long tools are declared `NON_BLOCKING` and get an immediate SILENT
   `will_continue=True` "started" ack — validated end-to-end against the Live API
   (`scripts/diag/probe_tool_response_scheduling.py`: ack silent, completion spoken,
   session healthy afterward). Server `tool_call_cancellation` ids cancel in-flight jobs.
2. **Decoupled deixis, resolve-on-settle** (`duplex_bridge/deixis/`) — the Week-4 finding
   productionized: `PointerReferentResolver` (Flash-Lite) reads tile+AX off the hot path.
   Staleness answer: the resolver re-fires each time the cursor settles on a NEW target
   (64-pt bucket + AX label; pending settle task cancelled on target change), so referents
   update continuously through the utterance — latest-wins in the `pointer=` annotation,
   and `activity_end` snapshots the full history for delegated tasks (multi-referent
   utterances like "compare this and this" fall out of the history). Tiles keep streaming.
3. **DelegateAgent** (`actions/delegate.py`) — the robust model seam: one
   `delegate_task(goal)` from the live model; a gemini-3.5-flash function-calling loop
   (async Interactions client, `previous_interaction_id` chaining) orchestrates
   `run_shell` / `run_applescript` / `browser_*` / nested `computer_use`
   (`asyncio.to_thread` + desktop mutex; Week-7b executor and policy reused unmodified).
4. **Persistent browser** (`actions/browser.py`) — one Chromium process, an isolated
   context+page per task; sync `playwright_navigator` in chrome.py untouched (thread-pool
   call site). Real-model validation: `demo_delegate_browser.py` ran headless — the
   delegate navigated to the Wikipedia Coffee article and reported its title.
5. **TaskManager multitasking** (`actions/tasks.py`) — N concurrent named tasks
   (`check_tasks` reports status; wall-clock test proves gather-level overlap), desktop
   mutex proof (two computer-use tasks never interleave on a shared FakeComputer),
   barge-in cancellation marks tasks cancelled.
6. **Safety: allowlist + voice confirmation** (`actions/safety.py`) — allowlisted
   read-only/reversible commands run autonomously; everything else raises
   `ConfirmationRequired`, pausing the task; the live model asks out loud and
   `confirm_task(task_id, approved)` resumes the SAME interaction chain. Destructive
   patterns (rm/sudo/force-push/`curl|sh`/Mail-send/…) always confirm; compound shell
   strings are never allowlisted. Independent of Gemini's built-in computer-use
   `safety_decision` handling (also honored).

## Probes (Phase 0, both PASS against the real API)

- `probe_tool_response_scheduling.py` — NON_BLOCKING + two-stage FunctionResponse works
  and the session stays usable (the initial "abort" was idle-connection reaping).
- `probe_interactions_combined_tools.py` — custom function tools + built-in computer_use
  coexist in ONE Interactions loop (`run_shell` call → grounded `click` in one chain).
  Shipped design keeps the nested executor (mutex semantics are cleaner); the combined
  loop is a known optimization for later.

## Validation — decoupled deixis e2e (Week-4 methodology)

The Week-4 experiment's resolver was never committed (its `pointer_referent` fixtures were
generated out-of-band, and the judged run scored 0/58 on a dead Anthropic key). Week 9
regenerates everything reproducibly with the RUNTIME resolver:

```bash
uv run python scripts/bench/generate_pointer_referents.py         # 58 referents, ~24 s
uv run python scripts/bench/eval_deictic.py \
  --tasks scripts/bench/fixtures/deictic_tasks_web_ax_resolved.jsonl \
  --escalate-full-frame --out scripts/bench/results/week9_decoupled.jsonl
uv run python scripts/bench/rejudge_subagent.py \
  --results scripts/bench/results/week9_decoupled.jsonl \
  --fixtures scripts/bench/fixtures/deictic_tasks_web_ax_resolved.jsonl
```

`rejudge_subagent.py` = rejudge.py's 3-vote majority with the judge running through the
authenticated `claude` CLI (Sonnet) instead of the dead SDK key.

**Result (58 tasks, production-faithful config + `pointer=` referent injection):**
- inline single judge: _see `results/week9_decoupled.jsonl`_
- 3-vote Sonnet rejudge: **TBD — filled in below when the run completes**
- baselines: 87.1% Week-4 acceptance (3-vote), ~80% observed on the drifted preview model.

## Not auto-verified (user-run)

The live voice smoke: mic + Accessibility + Screen Recording, `uv run -m duplex_bridge
--manual-vad --thinking-level minimal`, then e.g. "list the files on my desktop"
(allowlisted shell), "delete that folder" (voice-confirmation path), "open Wikipedia and
find the coffee article" (browser), "click the submit button" while pointing
(deixis → computer-use, desktop mutex).

## Reproduce

```bash
uv run pytest duplex-bridge/tests/test_worker.py duplex-bridge/tests/test_gemini_live.py \
  duplex-bridge/tests/test_deictic_context.py duplex-bridge/tests/test_delegate.py \
  duplex-bridge/tests/test_delegate_browser.py duplex-bridge/tests/test_tasks.py \
  duplex-bridge/tests/test_safety.py duplex-bridge/tests/test_computer_policy.py
uv run python scripts/demo/demo_delegate_browser.py --headless   # real-model browser e2e
```
