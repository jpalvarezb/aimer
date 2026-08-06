# The Glass Loop — Aimer's embodiment layer

Status: **design** (2026-08-06, pre-implementation)
Depends on: PTT-first product direction; Week-9 delegate/task stack; the behavior-eval
process rule (`scripts/bench/eval_behavior.py`).

## Vision

Aimer's thesis is pointer-grounded interaction: the user points, speaks, and the agent
acts. The glass loop closes that loop in the other direction — it gives the agent a
visible body at the point of attention. The cursor itself becomes a glassmorphic blob — a soft glass orb — while the
user talks to it, and can shed persistent glass artifacts (timer, calculator, task
tracker) that live on the desktop while work happens.

It solves three observed UX failures from live smokes, not hypothetical ones:

1. **"Am I being heard?"** — PTT/turn state today is a log line. The orb makes mic and
   turn state glanceable exactly where the user is already looking (the cursor).
2. **Invisible tasks** — on 2026-08-05 a delegated task ground for 90 s with zero visible
   evidence; the user had to ask. The task-tracker artifact shows current work + reasoning
   as it happens.
3. **Barge-in confidence** — you can't preempt what you can't see; listening vs speaking
   states make interruption predictable.

## Two surfaces, one visual language

### 1. The orb (the cursor *becomes* it, transient, non-interactive)

The form is a filled glass blob/circle — think liquid-glass orb, not a ring outline —
whose surface animates per state and which *morphs into* artifacts when they are born.

While active, the orb **replaces** the cursor's visual: the system cursor is hidden and
the glass form renders at the pointer position. This is a hard requirement, not polish —
in `acting` state the agent is moving the pointer itself, and a normal-looking cursor
moving on its own is indistinguishable from a haunted mouse. The orb being the cursor
is how the user knows who has the wheel.

States (a closed enum; the state machine is deterministic and unit-testable):

| state       | trigger (existing bridge event)                 | look                    |
|-------------|-------------------------------------------------|-------------------------|
| `hidden`    | idle — normal system cursor, orb absent        | nothing                 |
| `listening` | PTT key down (`MicCapture` PTT active)          | orb, steady glow        |
| `thinking`  | `activity_end` sent, no audio out yet           | orb, slow shimmer       |
| `speaking`  | first audio out → playback drained              | orb, gentle pulse       |
| `acting`    | agent holds the desktop mutex (`computer_use` / | orb, distinct tint +    |
|             | delegate driving mouse/keyboard)                | trailing motion         |
| `confirm`   | a task paused awaiting voice confirmation       | orb, amber pulse        |

Strictly click-through (`ignoresMouseEvents`); it must never steal a click.

**Cursor-replacement mechanics** (two regimes with different difficulty):

- **`acting` — fully ours, zero fragility.** The agent generates every pointer move
  (Quartz `CGEvent` posts in the `Computer` seam), so the overlay renders the orb at the
  exact injected coordinates. Hiding the real cursor during a drive is safe and scoped to
  the drive (`CGDisplayHideCursor` on mutex-acquire / show on release). This is the
  flagship moment: the user watches the glass orb do the clicking.
- **`listening`/`thinking`/`speaking`/`confirm` — user still owns the mouse.** The orb
  tracks the real pointer via a CGEventTap / global monitor (the pointer agent already
  samples position at 10 Hz; the overlay taps at display rate for smoothness). Hiding the
  system cursor from a background app requires the private-but-battle-tested CGS
  connection property (`SetsCursorInBackground`, the Cursorcerer technique) or the
  cursor-scale trick. Ship replacement as the design intent; if an OS update breaks the
  private API, the automatic fallback is an orb hugging the cursor — a degraded mode, not the
  design.

### 2. Artifacts (detached, persistent, interactive)

Born from the orb (it visually morphs or splits one off at the cursor), then independent:
draggable, dismissable, surviving PTT release. The cursor goes back to being a cursor; the artifact
stays until its job ends.

**Widget vocabulary v1** — a fixed registry, deliberately limited by us. The model cannot
invent UI; it can only instantiate registered kinds via a constrained tool call.

- **`timer`** — countdown / stopwatch inside the glass form. Model-created ("time three
  minutes"), voice-controlled (pause, cancel), X-to-dismiss.
- **`calculator`** — running calculation strip: the model appends steps/results; user can
  ask follow-ups by voice ("times twelve"). v1 is display-first (voice operates it);
  clickable keys are a later enhancement.
- **`task_tracker`** — one per delegated task. Shows: goal line, status
  (running/paused/done/failed), the current tool invocation, a scrolling reasoning/progress
  feed, and — when the task drives a browser or the desktop — a **live viewport pane**:
  throttled screenshot frames from the task's Playwright page or the `computer_use`
  `Computer.screenshot()` seam. This is how MCP-style tool UIs (e.g. Playwright) surface:
  the tracker embeds a *view* of the tool's world, not the tool's own chrome. Paused-for-
  confirmation renders as the amber state with the pending question; the user answers by
  voice (existing `confirm_task` path).

Registry growth is a product decision per addition — never a model capability.

## Architecture

The overlay is an **event consumer off the hot path** — the same seam discipline as
`DuplexSession` and the audio backends. Nothing in the audio path blocks on rendering.

```
bridge (existing events)                    overlay client
  MicCapture PTT / turn machine   ─┐
  GeminiLiveSession turn state    ─┤   /ui WebSocket    ┌─ orb window (click-through)
  TaskManager start/progress/done ─┼──────────────────▶ │
  delegate reasoning + viewport   ─┤   JSON events      └─ artifact windows (draggable)
  show_widget tool calls          ─┘
```

- **Transport**: a `/ui` endpoint beside `/context` on the existing WebSocket server.
  The bridge broadcasts; the overlay renders. Overlay crash/absence must be a no-op for
  the voice loop.
- **Overlay client**: prototype in PyObjC (borderless `NSPanel` at status-window level,
  `NSVisualEffectView` materials for the glass, zero build steps). Promote to a small
  Swift helper (`native/`, like the VPIO helper) only if rendering polish demands it.
- **No new permissions** for drawing; dragging uses normal window events on artifact
  panels only.

### Event schema (v1)

```jsonc
{"type": "orb",           "state": "listening"}                      // enum above; "acting"
                                                                     // fires on desktop-
                                                                     // mutex acquire/release
{"type": "task_started",  "task_id": "task-1", "goal": "..."}
{"type": "task_progress", "task_id": "task-1", "tool": "run_shell",
 "detail": "ls ~/Desktop", "reasoning": "checking the file exists"}
{"type": "task_viewport", "task_id": "task-1", "jpeg_b64": "...",
 "source": "playwright"}                                             // ≤2 FPS, only while
                                                                     // a tracker is open
{"type": "task_state",    "task_id": "task-1", "state": "paused",
 "question": "Run `rm -rf ...`?"}                                    // amber confirm
{"type": "task_done",     "task_id": "task-1", "summary": "..."}
{"type": "widget_show",   "widget_id": "w1", "kind": "timer",
 "params": {"seconds": 180, "label": "pasta"}}
{"type": "widget_update", "widget_id": "w1", "params": {...}}
{"type": "widget_close",  "widget_id": "w1"}
```

### `show_widget` tool (live model)

One new declaration in `TOOL_DECLARATIONS`:

```jsonc
{"name": "show_widget",
 "description": "Display a small glass widget near the user's cursor.",
 "parameters": {"kind": {"enum": ["timer", "calculator"]},   // registry-gated
               "params": {"type": "object"}}}
```

`task_tracker` is **not** model-invocable — it is spawned automatically by TaskManager
events, so every delegated task is visible by construction.

## Interaction rules

- Voice-first: widgets display; voice operates. Clicks are limited to drag, dismiss, and
  (later) calculator keys.
- One orb, N artifacts. Artifacts never follow the cursor after birth.
- Motion budget: subtle. The orb is glanceable truth, not animation showcase.
- When PTT ends mid-task: orb hides, trackers stay — matching "the cursor can exist
  somewhere else, but if a task is being conducted, it's still the loop."

## Testing discipline (per the 2026-08-06 process rule)

- **Deterministic**: the orb state machine (events → state) and the event broadcaster
  are plumbing — full unit coverage, no model in the loop.
- **Behavioral**: `show_widget` routing gets scenarios in `eval_behavior.py`
  ("time three minutes" → `show_widget(timer)`, not `delegate_task`; "what is this" must
  still never spawn a widget).
- **Feel**: each phase ships behind a `--overlay` flag and gets a 10-minute scripted live
  smoke before the next phase starts. No multi-week batches.

## Phasing

1. **Orb** — `/ui` endpoint + PyObjC overlay with the six states, including cursor
   replacement in `acting` (the easy, fully-owned regime) and the background-hide
   technique for the user-owned states. Acceptance: live smoke confirms PTT/turn state is
   glanceable, an agent-driven mouse is unmistakably the orb, and zero added latency on
   the audio path.
2. **Task tracker** — auto-spawned chip per task: goal, status, progress feed, amber
   confirm state. Acceptance: a delegated task is followable end-to-end without asking
   "how's it going", and a safety pause is visually obvious.
3. **Viewport pane** — Playwright / computer_use frames inside the tracker (≤2 FPS,
   only while open).
4. **Widget vocabulary** — `show_widget` + `timer`, then `calculator`. Acceptance:
   behavior-eval scenarios green; live smoke of each widget.

## Open questions (decide before Phase 2)

- Multi-display: orb follows the cursor's display; where do artifacts live when their
  display disconnects?
- Artifact persistence across bridge restarts (probably: none in v1 — tasks die with the
  bridge today anyway).
- Reasoning feed verbosity: raw delegate step lines v1; summarized later if noisy.
