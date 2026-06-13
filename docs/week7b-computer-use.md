# Week 7b — General computer-use (cross-application host control)

**Status: mechanism built + verified; live drive is user-run.** Week 7 shipped *bespoke* per-app
actions (a specific IDE edit, a specific Chrome compare). Production reality is **cross-application**
— point at anything in any app and act — so this adds a general computer-use layer: OS-level
*screenshot + mouse + keyboard* behind one seam, driven by a vision **policy**, exposed as a single
`computer_use(goal)` tool. The bespoke tools become fast paths; `computer_use` is the general fallback.

## Why (the critique this answers)

> "we built specific IDE things whereas the reality of production is cross-application"

A handler per task (`rewrite_function_async`, `compare_products`) doesn't generalize. A pointer-
grounded assistant must act on *whatever is on screen*. That is computer use: perceive the screen,
decide an action, execute it with mouse/keyboard, repeat — independent of which app is in front.

## Architecture (`duplex-bridge/src/duplex_bridge/actions/computer.py`)

```
goal ─▶ ComputerUseExecutor.run ──loop──▶ screenshot ─▶ policy(goal, shot, history) ─▶ Action ─▶ Computer.apply
                                            ▲                                                        │
                                            └──────────────── until policy returns `done` / max_steps ┘
```

- **`Computer`** (ABC) — OS primitives: `screenshot / click / move / type_text / key / scroll`, plus
  `apply(Action)`. One seam; macOS / Windows / Linux / a fake all implement it.
- **`MacOSComputer`** — real macOS impl: input via Quartz `CGEvent` (mouse/keyboard/scroll), screen
  via `screencapture` (lazy PyObjC import). Needs **Accessibility** (post input) + **Screen
  Recording** (capture). Posting synthetic input drives the real desktop, so it is run by the user.
- **`ComputerUseExecutor`** — the cross-application loop. It never knows which app it's driving — the
  **policy** decides from the screenshot, the primitives are OS-level. Runs off the audio hot path
  (Week-6 worker).
- **`Policy`** = `(goal, screenshot, history) -> Action` — the pluggable decision model. A real one
  wraps a computer-use vision model (Claude computer-use, Gemini vision); a scripted one drives tests.
- **`computer_use(goal)`** tool (`TOOL_DECLARATIONS`) — the duplex model emits it for any action not
  covered by a specific tool; `ToolDispatcher` runs it off the hot path.

## Verification

The executor loop — the cross-application core — is verified **deterministically** with
`FakeComputer` + a scripted policy (`duplex-bridge/tests/test_computer.py`, 7 tests): runs an
action sequence then terminates on `done`, respects `max_steps`, supports async policies, dispatches
every primitive, rejects unknown actions, and `MacOSComputer` constructs without importing Quartz
(lazy). What is *not* auto-verified is (a) driving the real desktop (synthetic input on your machine
— run by you, with Accessibility granted) and (b) a real vision policy's decisions (model behaviour).

## Plugging in a real policy

```python
from duplex_bridge.actions.computer import Action, MacOSComputer, run_computer_use

def my_policy(goal: str, screenshot: bytes, history: list[Action]) -> Action:
    # send screenshot+goal to a computer-use vision model (Claude/Gemini); map its reply to an Action
    ...

# in __main__, replace _unconfigured_computer_policy:
tool_dispatcher.register(
    "computer_use",
    lambda a: run_computer_use(a["goal"], MacOSComputer(), my_policy),
)
```

The default wiring uses `_unconfigured_computer_policy` (returns `done` with a "configure a policy"
note) so the tool is declared and dispatched end-to-end; dropping in a real policy is the only step
to a live cross-application agent.

## Reproduce

```bash
uv run pytest duplex-bridge/tests/test_computer.py   # 7 tests — the executor loop + dispatch
```
