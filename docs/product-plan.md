# Aimer — Product Plan

Status: **active plan** (2026-08-06)
Decision of record: Aimer is a **product built for users** — a futuristic automation agent
for macOS — not a portfolio artifact or a research vehicle. Everything below follows from
that choice.

Companion docs: `docs/glass-loop-design.md` (embodiment layer), `docs/architecture.md`
(system), `docs/week*-acceptance.md` (what is already proven).

---

## 1. Positioning

**Aimer is the Mac assistant that knows exactly what you're pointing at — and learns what
you do over and over.**

Two sentences, two bets:

- **Precision** — deictic reference resolved from continuous pointer telemetry, not guessed
  from a screenshot. "Fix this," "summarize that," "what is this" land on the right element.
- **Discovery** — because Aimer observes continuously, it can notice repeated workflows and
  offer to run them. The user never has to know an automation existed to get one.

Not the pitch: "voice-control your whole Mac." That space is occupied, commoditizing, and
is the source of every reliability problem this project has had.

---

## 2. Competitive baseline

### Cursor Voice (`github.com/cursorvoice/cursor-voice`, source read 2026-08-06)

Native Swift/SwiftUI menu-bar app, OpenAI Realtime API, free + BYO key, open source.

- **Cursor attachment**: two panels follow the cursor at 60 Hz — a 300×280 orb panel offset
  down-right, and `CursorHalo`, a 120×120 click-through panel centered on the cursor
  rendering layered radial blooms ("no ring, no dot"), breathing on a sine and brightening
  while the AI synthesizes input. It does **not** hide or replace the system cursor
  (verified: no `NSCursor.hide` / `CGDisplayHideCursor` / CGS private API in source).
- **State model**: idle/connecting/listening/thinking/speaking/error + `aiControlling`,
  `activeTool`, audio-reactive input/output levels, live `sessionCost`.
- **Perception**: on-demand ScreenCaptureKit screenshot + OCR + AX tree + **Set-of-Marks**
  (`MarkOverlay`: numbered badges over AX/OCR candidate regions so the model picks a target
  by index instead of pixel coordinates).
- **Actions**: AX-tree clicks, CGEvent input synthesis, AppleScript, shell, browser bridge,
  web search, file ops, window management, clipboard, Google auth, plugin system.
- **Automation**: `MacroStore` — *explicitly recorded* sequences of tool calls, replayed by
  name ("record a macro called deploy" → steps → "run my deploy macro").
- **Weaknesses**: ad-hoc signing (Gatekeeper blocks first launch), sandbox disabled,
  Apple-Silicon only, Screen Recording permission honored only on fresh process launch.

**Read**: they own breadth and have shipped. They have no referent resolution and no
continuous observation. Their automations are recorded by the user; ours would be
discovered by the system.

### Apollo (`heyapollo.dev`)

Different genre — voice agent for *physical devices* (mic + speaker + face), wake-word,
brain on Cloudflare, open firmware, privacy-first ("keys never leave your account"). Not a
competitor; a useful reference for the four-face state personality and for how bluntly a
privacy story can be foregrounded.

### The threat that actually matters

Not Cursor Voice — a platform vendor (Apple, OpenAI) shipping this natively. Strategic
consequence: build what a platform vendor structurally won't — deep personal workflow
learning from **local** telemetry, with a privacy posture a cloud vendor can't match.

---

## 3. Wedge and moat

**Wedge (why someone switches):** point at a cell, a function, a paragraph — say "fix
this" — and it's right. Measured today at **87.1%** on the 58-task deictic eval
(`docs/week4-deictic-acceptance.md`). Nobody else in this category measures referent
accuracy at all.

**Moat (why it isn't copied in a weekend):** the 10 Hz telemetry harness — cursor, window,
app, AX labels, selection, tiles — running continuously. Built for deixis, it is also the
only viable substrate for discovered automations. Matching it means rebuilding perception
*and* earning the trust to observe continuously.

**Moat's cost:** continuous capture is a harder trust sell than on-demand screenshots. The
moat and the biggest adoption obstacle are the same feature, so local-first processing is
a requirement, not a nicety (§6).

---

## 4. Scope: three jobs, done flawlessly

1. **Point-and-ask** — "what is this / summarize this / explain this / translate this."
   Answered from context already in hand, sub-second, never delegated.
2. **Point-and-do-one-thing** — open, search, copy, create, rename, send. Bounded, verified,
   honest on failure.
3. **Glass widgets** — timer and calculator as the delightful, zero-risk surface; task
   tracker as the visibility fix for anything longer-running.

Explicitly **not yet**: arbitrary desktop automation. `computer_use` and the open-ended
delegate remain as bounded fallbacks (§5, phase 4), not headline features.

---

## 5. Build sequence

Each phase ships behind a flag where possible, and each ends with a ≤10-minute scripted
live smoke in **PTT mode** before the next begins (per
`memory: live-smoke-bugs-live-in-the-prompt-layer`). No multi-week batches.

### Phase 0 — Clean base *(1 day)*
Merge PR #4 after a PTT live smoke. Close the Week-9 book.
**Done when:** `main` carries the Week-9 stack; no uncommitted work.

### Phase 1 — It's an app *(1–2 weeks)*
One signed, launchable macOS app: menu-bar item, supervises bridge + pointer agent, no
terminals. Onboarding for the three permissions (Accessibility, Screen Recording,
Microphone) including the **restart-after-Screen-Recording** trap Cursor Voice documents.
Developer ID signing + notarization (their ad-hoc weakness is our easy win).
**Done when:** a non-developer can install and hold a conversation without a terminal.

### Phase 2 — The orb *(1–2 weeks)*
`/ui` WebSocket + overlay per `docs/glass-loop-design.md`, revised to the **aura** form:
a click-through panel centered on the cursor with layered radial blooms, breathing, and
state-driven intensity/tint — *no cursor replacement* except optionally during `acting`,
where we own every injected CGEvent. Steal: 60 Hz `NSEvent.mouseLocation` polling (avoids
an Accessibility prompt) and synthetic-event tagging via `eventSourceUserData` so our own
global monitors ignore agent-generated clicks.
**Done when:** mic/turn state is glanceable; agent-driven input is unmistakable; zero
added latency on the audio path.

### Phase 3 — Point-and-ask, flawless *(1–2 weeks)*
Deterministic routing: describe/identify never becomes a tool call. Latency target
sub-second end-of-speech → first audio in PTT mode. Expand `scripts/bench/eval_behavior.py`
to cover the full job (referent phrasing variants, multilingual, no-referent fallback).
Push the deictic eval past 90% on the production config.
**Done when:** behavior eval green across ≥15 scenarios; deictic eval ≥90%; a live smoke
of 20 consecutive point-and-ask turns has zero wrong-element answers.

### Phase 4 — Bounded actions *(1–2 weeks)*
Wall-clock budget (~45 s) and fail-fast on the delegate: any approach failing twice ends
the task with a spoken honest failure. **Visual verification** after actions (borrowed from
Cursor Voice's auto-screenshot-after-each-action) generalizing today's file-only
verify-before-done. Adopt **Set-of-Marks** grounding for `computer_use`. Session cost meter
surfaced in the orb.
**Done when:** no silent grind exceeds the budget; a failed task always says why, out loud.

### Phase 5 — Discovered automations v0 *(3–4 weeks)*
Mine the telemetry stream already being collected for repeated sequences (app + AX-label +
action n-grams over a rolling window). Surface **one** high-confidence suggestion at a
calm moment: "You've done this three times this week — want me to do it next time?"
Accept by voice; store as a named automation; run on request. Recorded macros are the
manual fallback; discovery is the product.
**Done when:** on a week of real usage, ≥1 correct suggestion and zero false suggestions
that survive user review; all mining runs locally.

---

## 6. Decisions that must be made early

| Decision | Options | Recommendation |
|---|---|---|
| **Cost model** | BYO API key (free, dev audience, no margin) vs hosted with a meter | **BYO key at launch.** Matches the competitor's floor, removes billing infrastructure from the critical path, and the cost meter (phase 4) makes spend honest. Revisit if consumer adoption outpaces developer adoption. |
| **Privacy posture** | Cloud-processed telemetry vs local-first | **Local-first, stated loudly.** Only what the user points at (tile + annotation) leaves the machine, during a turn. Pattern mining runs entirely on-device. This is a requirement of the moat, not a marketing choice. |
| **Model provider** | Gemini Live (today) vs OpenAI Realtime vs both | **Stay on Gemini Live**; the `DuplexSession` seam already isolates it. Provider choice is not a differentiator and switching costs attention the wedge needs. |
| **Open source?** | Closed vs open | **Undecided — decide by Phase 1.** The competitor is open and free; being closed demands the signed-and-polished advantage actually materialize. |

---

## 7. Risks

- **Trust in continuous capture.** Mitigation: local-first, an obvious kill switch, a
  visible indicator whenever telemetry is active, and no cloud upload outside a turn.
- **Platform vendor ships it.** Mitigation: personalization depth and local learning — the
  part a vendor with everyone's data still can't do per-user without the same trust cost.
- **Solo bandwidth vs a shipped open-source competitor.** Mitigation: refuse breadth. Three
  jobs (§4). Every "wouldn't it be cool if" goes to a parking list.
- **Reliability regressions in model behavior.** Mitigation: the behavior eval is the merge
  gate for every prompt/routing change; live smokes stay small and frequent.
- **The orb becomes a time sink.** Mitigation: Phase 2 is timeboxed to the aura + states;
  artifacts land only after Phase 3.

---

## 8. What "working, not clunky" means concretely

The acceptance bar for calling this a product, all measured in PTT mode on a real Mac:

1. Launch from Spotlight, no terminal, no ceremony.
2. Hold key → aura responds within a frame or two; release → first audio ≤ 1 s.
3. Twenty consecutive point-and-ask turns, zero wrong-element answers.
4. Every action either completes with verification or fails out loud inside the budget.
5. Nothing the agent does is invisible: state in the aura, work in the tracker, spend in
   the meter.
