"""Turn-level behavior eval for the live model + system prompt.

Every scenario is a canned [context] annotation plus one spoken utterance, driven
through a real ``GeminiLiveSession`` (manual VAD, audio in, transcribed audio out,
the production tool declarations) exactly like ``measure_ttfb.py`` drives latency
runs. Assertions are about *behavior*: what the transcript says and which tools the
model did or did not call.

This is the regression suite for ``_SYSTEM_INSTRUCTION``: every clause change needs
a scenario here reproducing the live bug it fixes, and the existing scenarios must
stay green. Seeded from the 2026-08-05 live smoke:

  - pointer_app_identity   "what app am I pointing at" answered from pointer_app=,
                           not the focused app= (the Ghostty-instead-of-Notion bug)
  - focused_app_identity   "what app am I in" still answered from app= (guards
                           against overcorrecting the other way)
  - deixis_direct_answer   "what is this" answered directly from pointer=, without
                           spawning delegate_task (the Swift-OCR spiral bug)
  - same_app_pointer       pointer_app absent -> focused app is the pointer answer
  - status_check_tasks     task-status question calls check_tasks first (7/23 fix)

Usage:
    uv run python scripts/bench/eval_behavior.py                 # all scenarios
    uv run python scripts/bench/eval_behavior.py --only deixis_direct_answer
    uv run python scripts/bench/eval_behavior.py --runs 3        # vote per scenario

Needs GEMINI_API_KEY in the repo .env (same convention as measure_ttfb.py).
Each scenario is one short Live session; a full run costs a few cents.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent.parent  # scripts/bench/ -> repo root
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "aimer-core" / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "bench"))

from aimer_core.schema import ContextPacket, CursorPosition, FocusWindow  # noqa: E402
from duplex_bridge.actions import TOOL_DECLARATIONS  # noqa: E402
from duplex_bridge.providers.gemini_live import GeminiLiveSession  # noqa: E402
from measure_ttfb import chunk_pcm, generate_speech_pcm, load_api_key  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("eval_behavior")

MODEL = "models/gemini-3.1-flash-live-preview"
TURN_TIMEOUT_S = 25.0
# A tool call usually precedes turn_complete; once one arrives, wait this much longer
# for the transcript tail, then score.
POST_TOOL_GRACE_S = 4.0


@dataclasses.dataclass
class Scenario:
    name: str
    description: str
    utterance: str
    focus_app: str
    focus_title: str | None = None
    app_under_cursor: str | None = None
    pointer_referent: str | None = None
    # Transcript must contain at least one of these (case-insensitive). Empty = no check.
    expect_any: tuple[str, ...] = ()
    # Transcript must contain none of these.
    expect_none: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()
    require_tools: tuple[str, ...] = ()


SCENARIOS: list[Scenario] = [
    Scenario(
        name="pointer_app_identity",
        description="Hovering Notion while a terminal is focused: pointer_app= wins.",
        utterance="What app am I pointing at right now?",
        focus_app="Ghostty",
        focus_title="zsh",
        app_under_cursor="Notion",
        pointer_referent="The cursor is pointing at a block drag handle in a document page.",
        expect_any=("notion",),
        expect_none=("ghostty",),
        forbid_tools=("delegate_task", "computer_use"),
    ),
    Scenario(
        name="focused_app_identity",
        description="Same split context: 'which app am I in' still means the focused app.",
        utterance="Which app am I in right now, the focused one?",
        focus_app="Ghostty",
        focus_title="zsh",
        app_under_cursor="Notion",
        pointer_referent="The cursor is pointing at a block drag handle in a document page.",
        expect_any=("ghostty",),
        forbid_tools=("delegate_task", "computer_use"),
    ),
    Scenario(
        name="deixis_direct_answer",
        description="'What is this?' is answered from pointer=, never delegated.",
        utterance="What is this that I'm pointing at?",
        focus_app="Ghostty",
        focus_title="zsh",
        app_under_cursor="Notion",
        pointer_referent=(
            "The cursor is pointing at the heading 'Q3 Budget Review' at the top of a "
            "document page."
        ),
        expect_any=("budget", "q3", "heading"),
        forbid_tools=("delegate_task", "computer_use", "click_pointer"),
    ),
    Scenario(
        name="same_app_pointer",
        description="No pointer_app (cursor over the focused app): app= answers both forms.",
        utterance="What app am I pointing at?",
        focus_app="Notion",
        focus_title="Q3 Budget Review",
        app_under_cursor=None,
        pointer_referent="The cursor is pointing at a paragraph of meeting notes.",
        expect_any=("notion",),
        forbid_tools=("delegate_task", "computer_use"),
    ),
    Scenario(
        name="this_app_identity",
        description=(
            "Ambiguous phrasing ('what app is this?') while hovering a different app: the "
            "user means the pointed-at app, not the focused one (2026-08-05 Ghostty bug)."
        ),
        utterance="What app is this?",
        focus_app="Ghostty",
        focus_title="zsh",
        app_under_cursor="Notion",
        pointer_referent="The cursor is pointing at a block drag handle in a document page.",
        expect_any=("notion",),
        expect_none=("ghostty",),
        forbid_tools=("delegate_task", "computer_use"),
    ),
    Scenario(
        name="status_check_tasks",
        description="Task-status questions must call check_tasks before answering.",
        utterance="How is that background task you started going?",
        focus_app="Ghostty",
        focus_title="zsh",
        require_tools=("check_tasks",),
        forbid_tools=("delegate_task",),
    ),
]


@dataclasses.dataclass
class ScenarioResult:
    name: str
    passed: bool
    transcript: str
    tool_calls: list[str]
    failures: list[str]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_packet(s: Scenario) -> ContextPacket:
    return ContextPacket(
        cursor=CursorPosition(x=640.0, y=400.0),
        focus_window=FocusWindow(app=s.focus_app, title=s.focus_title),
        app_under_cursor=s.app_under_cursor,
    )


async def run_scenario(s: Scenario, frames: list[bytes]) -> ScenarioResult:
    transcript_parts: list[str] = []
    tool_calls: list[str] = []
    turn_done = asyncio.Event()
    saw_tool = asyncio.Event()

    session = GeminiLiveSession(
        model=MODEL,
        api_key_env="GEMINI_API_KEY",
        manual_vad=True,
        thinking_level="minimal",
        output_audio_transcription=True,
        tools=TOOL_DECLARATIONS,
    )
    session.on_text_out(lambda text: transcript_parts.append(text))
    session.on_turn_complete(lambda: turn_done.set())

    def _on_tool(tc: Any) -> None:
        # on_tool_call delivers the LiveServerToolCall container; the individual
        # FunctionCalls (with .name/.id) live in its .function_calls list.
        for fc in getattr(tc, "function_calls", None) or []:
            name = getattr(fc, "name", None)
            if name:
                tool_calls.append(name)
        saw_tool.set()

    session.on_tool_call(_on_tool)

    try:
        await session.open()
        await asyncio.sleep(0.3)
        await session.send_visual_context(_build_packet(s))
        if s.pointer_referent:
            # Inject a pre-resolved referent exactly where the settle resolver would put
            # it, so the eval isolates live-model behavior from resolver behavior.
            session._latest_pointer_referent = s.pointer_referent
            session._latest_pointer_referent_app = s.app_under_cursor or s.focus_app

        await session.send_activity_start()
        for frame in frames[:-1]:
            await session.send_audio(frame)
            await asyncio.sleep(0.1)
        await session.send_audio(frames[-1])
        await session.send_activity_end()

        try:
            await asyncio.wait_for(
                asyncio.wait(
                    [
                        asyncio.ensure_future(turn_done.wait()),
                        asyncio.ensure_future(saw_tool.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                ),
                timeout=TURN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return ScenarioResult(s.name, False, "", tool_calls, ["timeout: no response"])
        if saw_tool.is_set() and not turn_done.is_set():
            await asyncio.sleep(POST_TOOL_GRACE_S)
    finally:
        await session.close()

    transcript = " ".join(transcript_parts).strip()
    lowered = transcript.lower()
    failures: list[str] = []
    if s.expect_any and not any(w.lower() in lowered for w in s.expect_any):
        failures.append(f"transcript lacks all of {list(s.expect_any)}: {transcript!r}")
    for w in s.expect_none:
        if w.lower() in lowered:
            failures.append(f"transcript contains forbidden {w!r}: {transcript!r}")
    for t in s.forbid_tools:
        if t in tool_calls:
            failures.append(f"called forbidden tool {t}")
    for t in s.require_tools:
        if t not in tool_calls:
            failures.append(f"did not call required tool {t}")
    if not transcript and not tool_calls:
        failures.append("empty turn: no transcript and no tool calls")
    return ScenarioResult(s.name, not failures, transcript, tool_calls, failures)


async def _main(only: str | None, runs: int) -> int:
    api_key = load_api_key()
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env", file=sys.stderr)
        return 1
    os.environ["GEMINI_API_KEY"] = api_key

    scenarios = [s for s in SCENARIOS if only is None or s.name == only]
    if not scenarios:
        print(f"ERROR: no scenario named {only!r}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    all_passed = True
    for s in scenarios:
        print(f"\n=== {s.name} — {s.description}")
        frames = chunk_pcm(generate_speech_pcm(s.utterance))
        votes: list[ScenarioResult] = []
        for i in range(runs):
            r = await run_scenario(s, frames)
            votes.append(r)
            mark = "PASS" if r.passed else "FAIL"
            print(f"  run {i + 1}: {mark}  tools={r.tool_calls}  transcript={r.transcript!r}")
            for f in r.failures:
                print(f"         - {f}")
            if i < runs - 1:
                await asyncio.sleep(2.0)
        # Majority vote: live-model behavior is stochastic; a scenario passes when
        # more than half its runs pass.
        passed = sum(v.passed for v in votes) > runs / 2
        all_passed &= passed
        print(f"  => {'PASS' if passed else 'FAIL'} ({sum(v.passed for v in votes)}/{runs})")
        results.append({"scenario": s.name, "passed": passed, "runs": [v.as_dict() for v in votes]})

    out = Path(__file__).parent / "results" / "behavior_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "results": results}, indent=2))
    print(f"\n{'ALL PASS' if all_passed else 'FAILURES PRESENT'} — written to {out}")
    return 0 if all_passed else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Turn-level behavior eval for the live prompt.")
    p.add_argument("--only", default=None, help="Run a single scenario by name")
    p.add_argument("--runs", type=int, default=1, help="Runs per scenario (majority vote)")
    args = p.parse_args()
    return asyncio.run(_main(args.only, args.runs))


if __name__ == "__main__":
    raise SystemExit(main())
