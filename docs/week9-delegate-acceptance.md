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
   context+page per task, headless by default (research browsing never flashes a visible
   window); `_ensure_started` prefers the user's installed Google Chrome
   (`channel="chrome"`) and falls back to Playwright's bundled Chromium only if that
   channel is unavailable. When the delegate wants the user to SEE a page it opens the URL
   in their real default browser via `open <url>` (`run_shell`), not this research browser.
   Sync `playwright_navigator` in chrome.py untouched (thread-pool call site; still headed
   for `compare_products`, which the user watches by design). Real-model validation:
   `demo_delegate_browser.py` ran headless — the delegate navigated to the Wikipedia Coffee
   article and reported its title.
5. **TaskManager multitasking** (`actions/tasks.py`) — N concurrent named tasks
   (`check_tasks` reports status; wall-clock test proves gather-level overlap), desktop
   mutex proof (two computer-use tasks never interleave on a shared FakeComputer),
   barge-in cancellation marks tasks cancelled.
6. **Safety: allowlist + voice confirmation** (`actions/safety.py`) — allowlisted
   read-only/reversible commands run autonomously; everything else raises
   `ConfirmationRequired`, pausing the task; the live model asks out loud and
   `confirm_task(task_id, approved, approve_all=False)` resumes the SAME interaction chain.
   `approve_all=True` pre-approves the rest of that task's confirmations from one spoken
   yes (still bounded by `MAX_CONFIRMATIONS_PER_TASK`; destructive patterns always confirm
   individually regardless). Destructive patterns (rm/sudo/force-push/`curl|sh`/Mail-send/…)
   always confirm; compound shell strings are decomposed on `; && || |` / newlines and
   auto-allowed only when every segment is allowlisted and non-destructive (backticks/`$()`
   always confirm — they hide nested commands). Independent of Gemini's built-in
   computer-use `safety_decision` handling (also honored).

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

`rejudge_subagent.py` = rejudge.py's 3-vote majority with the judge routed through the
`claude` CLI (Sonnet). Note: nested `claude` invocations are blocked from inside a Claude
Code session — there, dispatch 3 independent Sonnet subagents with the same judge prompt
instead (how the number below was produced; the script serves standalone runs).

**Result (58 tasks, production-faithful config + `pointer=` referent injection,
3-vote majority of independent Sonnet judges):**

| config | pass | wrong-element (incorrect) | ambiguous |
|---|---|---|---|
| baseline, drifted live model (`week4_ax_on_recheck_rejudged`) | 45/58 = **77.6%** | 11 | 2 |
| **decoupled (`week9_decoupled_rejudged`)** | 47/58 = **81.0%** | **1** | 10 |

81.0% clears the Week-4 acceptance bar (80%) on today's drifted model, where the coupled
baseline fails it. The failure-mode shift is the real story: **wrong-element attention
errors drop 11 → 1** with the referent injected — the decoupled reader nearly eliminates
grounding mistakes. The remaining misses are "ambiguous" verdicts dominated by transcripts
truncated by the eval harness's response-capture window (`RESPONSE_TIMEOUT_S`/
`TEXT_SETTLE_S`), a measurement artifact worth a follow-up, not a grounding failure.
(Week-4's original 87.1% predates the model drift; inline single-judge on this run reads
71% — single-judge underscoring is a known ~8-10 pt effect, hence the 3-vote protocol.)

## Live-smoke fixes

The first user-run voice smoke against this stack surfaced five behavior bugs, all now
fixed behind existing seams with deterministic tests (no API keys / live audio needed):

- **Delegate browser UX** — `DelegateBrowser._ensure_started` now tries the user's
  installed Google Chrome (`chromium.launch(channel="chrome")`) before falling back to
  Playwright's bundled Chromium; the research browser defaults `headless=True` (it used to
  flash a visible "Chrome for Testing" window per task); when the delegate wants the user
  to SEE something it surfaces the URL via `open <url>` (macOS) to their real default
  browser instead. The browser stays persistent across tasks — no open/close flash.
- **Safety de-nagging** — compound shell commands are decomposed on `; && || |` and
  newlines and auto-allowed when every segment is allowlisted and non-destructive
  (`curl ... | head` no longer needs a voice confirmation); backticks/`$()` still always
  confirm since they hide nested commands, and `curl | sh`-style pipe-to-interpreter stays
  destructive. Read-only System Events AppleScript (`get`/`count`/`exists` of
  processes/windows/UI elements, plain `activate`) and `screencapture` are allowlisted.
  `confirm_task` gained an `approve_all` flag: one voice yes can pre-approve the rest of a
  task's confirmations, bounded by `MAX_CONFIRMATIONS_PER_TASK`; destructive patterns still
  confirm individually even under `approve_all`. `curl` is allowlisted only for read-only
  GETs — state-changing methods (`-X POST/PUT/DELETE/PATCH`), request bodies (`-d`/`--data`,
  data exfiltration), form/file uploads (`-F`/`-T`), and arbitrary file writes
  (`-o`/`--output`/`-O`) are pulled back to always-confirm (post-review hardening, checked
  before the allowlist so they win).
- **VPIO mic static** — `_handle_tap_buffer` no longer blindly reads channel 0 of the tap
  buffer; a pure, unit-testable `_extract_mono` function inspects the buffer's actual
  format (interleaved vs. deinterleaved, channel count, stride) and extracts the correct
  voice channel, covering a 9-channel interleaved aggregate input device, a 3-channel
  deinterleaved device, and plain mono, all bit-exact against synthetic buffers. The chosen
  extraction path is logged once.
- **computer_use robustness** — the live tool call is wrapped in
  `run_computer_use_with_timeout` (120 s wall clock via `asyncio.wait_for`), so a dangling
  policy call (observed live: two Teams-click calls that never returned) always resolves to
  a final `FunctionResponse` instead of silence. It shares
  `wrap_with_vision_loop_fallback` (`actions/computer_policy.py`) with the delegate's own
  `computer_use` call site: the hosted `GeminiComputerUsePolicy` runs first, and on a
  server-side "Input blocked" classifier rejection it retries once via the plain-
  `generateContent` `GeminiVisionLoopPolicy`. A new `click_pointer` tool gives a
  deterministic fast path — when a fresh pointer referent + settled cursor coordinates are
  available (`GeminiLiveSession.pointer_click_target()`), it issues one `Computer.click`
  directly, no screenshot round-trip, no vision-model call; the referent + coordinates are
  also folded into the `computer_use` goal string as a grounding hint even when the fast
  path doesn't fire.
- **Delegate polish** — the `_SYSTEM` prompt now steers the agent toward `browser_*` for
  web reading (headless), `open <url>` via `run_shell` when the user should see a page,
  read-only AppleScript queries before `computer_use`, and asking once before a long run of
  similar shell steps instead of per-command. The `UserWarning: Async interactions client
  cannot use aiohttp, falling back to httpx` emitted at client construction is suppressed
  narrowly (`warnings.catch_warnings`, matched by message) around just that construction
  call — nothing else is silenced.

Re-test live: `uv run -m duplex_bridge --audio-backend vpio --manual-vad --thinking-level
minimal`.

## Not auto-verified (user-run)

The live voice smoke: mic + Accessibility + Screen Recording, `uv run -m duplex_bridge
--audio-backend vpio --manual-vad --thinking-level minimal` (in-process hardware AEC,
capture + playback ear-verified on-device 2026-07-03; the default backend self-interrupts
on speakers — use it only with headphones or `--push-to-talk`), then e.g. "list the files
on my desktop" (allowlisted shell), "delete that
folder" (voice-confirmation path), "open Wikipedia and find the coffee article" (browser),
"click the submit button" while pointing (deixis → computer-use, desktop mutex).

## Reproduce

```bash
uv run pytest duplex-bridge/tests/test_worker.py duplex-bridge/tests/test_gemini_live.py \
  duplex-bridge/tests/test_deictic_context.py duplex-bridge/tests/test_delegate.py \
  duplex-bridge/tests/test_delegate_browser.py duplex-bridge/tests/test_tasks.py \
  duplex-bridge/tests/test_safety.py duplex-bridge/tests/test_computer_policy.py
uv run python scripts/demo/demo_delegate_browser.py --headless   # real-model browser e2e
```
