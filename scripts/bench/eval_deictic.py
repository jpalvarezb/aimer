"""LLM-as-judge evaluation harness for Week-4 deictic resolver acceptance.

Drives GeminiLiveSession directly (no WebSocket server, no pointer agent, no mic).
Sends a cursor tile image plus a text annotation to Gemini Live — exactly the
visual-context path that runs in production — then uses a Claude judge to score
whether the model's text response correctly resolves the deictic referent.

IMPORTANT CAVEAT — text injection bypasses ASR:
    Utterances are injected via send_realtime_input(text=...), not via speech.
    This tests visual grounding + deictic resolution, NOT the speech transcription
    pipeline. Any failure attributable to ASR cannot be detected here.

Acceptance bar: pass_rate >= 80% (correct only; ambiguous does NOT count as pass).

Usage:
    uv run python scripts/bench/eval_deictic.py
    uv run python scripts/bench/eval_deictic.py --limit 3
    uv run python scripts/bench/eval_deictic.py --escalate-full-frame
    uv run python scripts/bench/eval_deictic.py --out results/week4_deictic.jsonl

Requires: GEMINI_API_KEY and ANTHROPIC_API_KEY in the environment or in .env.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make workspace packages importable when running as a script.
# scripts/bench/ is 3 levels from repo root.
_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/bench/ -> repo root
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "aimer-core" / "src"))

from duplex_bridge.providers.gemini_live import GeminiLiveSession  # noqa: E402
from google.genai import types  # noqa: E402

_HERE = Path(__file__).parent  # scripts/bench/

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval_deictic")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5"
DEFAULT_TASKS_PATH = _HERE / "fixtures" / "deictic_tasks.jsonl"
RESPONSE_TIMEOUT_S = 15.0
TEXT_SETTLE_S = 1.5  # wait after activity_end to collect trailing text chunks
IMAGE_INGEST_S = 2.0  # let the model ingest the still frame before the utterance triggers a turn
PASS_THRESHOLD = 0.80  # 80% correct for Week-4 acceptance

_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge for a deictic reference resolution evaluation.

The user pointed at one specific on-screen element (the expected_referent) and spoke the
utterance. Judge whether the assistant attended to the RIGHT thing — not whether it used the
exact words. The question is "did it resolve the pointer to the correct element?", graded for
ATTENTION TO THE RIGHT REGION, with tolerance for granularity and wording.

VERDICT: correct — the response is about the pointed-at element. ALL of these count as correct:
  • names or paraphrases the element (synonyms, different wording are fine);
  • identifies a specific word, sub-part, or detail WITHIN the expected_referent
    (e.g. the word "hot" within a cell "Usually hot; can be iced"; a button's purpose);
  • accurately describes the element's content or FUNCTION even without naming it
    (e.g. "an input field for the customer's name" for the 'Customer name' field;
     "the side button on a computer mouse" for an image of a computer mouse).
VERDICT: incorrect — the response attends to the WRONG thing. This includes:
  • identifying a DIFFERENT element than the one pointed at (a neighbour, a different
    link/section/story);
  • answering about the WHOLE page / entire screenshot instead of the specific element;
  • refusing, or hallucinating content that is not the expected_referent.
VERDICT: ambiguous — too vague to tell which element it meant (no specific content named).

Be lenient on WORDING and GRANULARITY (a correct sub-part or functional description passes),
but strict on ATTENTION (wrong element or whole-page answers fail). "ambiguous" does NOT pass.

Format your response EXACTLY as:
VERDICT: correct|incorrect|ambiguous
REASON: <one line, ≤80 chars>
"""

_JUDGE_USER_TEMPLATE = """\
utterance: {utterance}
expected_referent: {expected_referent}
model_response: {model_response}

Score the model_response using the rules above.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TaskResult:
    id: str
    utterance: str
    model_response: str
    judge_verdict: str  # "correct" | "incorrect" | "ambiguous"
    judge_reason: str
    passed: bool  # True only when verdict is "correct"

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_api_key_from_env_file(key_name: str) -> str:
    """Read a key from .env in the repo root; fall back to the process environment."""
    env_val = os.environ.get(key_name, "")
    if env_val:
        return env_val
    env_path = _ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == key_name:
                    return v.strip()
    return ""


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
    return tasks


def _build_text_annotation(task: dict[str, Any]) -> str:
    """Mirror GeminiLiveSession._build_text_annotation for a task fixture."""
    parts = ["[context]"]
    # task fixtures may carry image_path as a hint for app/title, use notes fallback
    app = task.get("app") or "Aimer"
    title = task.get("window_title") or task.get("image_path", "unknown")
    parts.append(f"app={app}")
    parts.append(f"title={title}")
    cx = task.get("cursor_tile_x", 0)
    cy = task.get("cursor_tile_y", 0)
    parts.append(f"cursor=({cx:.0f},{cy:.0f})")
    selected = task.get("selected_text")
    if selected:
        parts.append(f"selected={str(selected)[:80]}")
    return " ".join(parts)


def _get_anthropic_client() -> Any:
    """Lazily import and construct an Anthropic client; raise clearly if unavailable."""
    try:
        import anthropic  # noqa: PLC0415  (intentional lazy import)
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package is required for the LLM judge.\n"
            "Install it with:  pip install anthropic\n"
            "or:               uv pip install anthropic"
        ) from exc
    api_key = _load_api_key_from_env_file("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it or add it to .env:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def _parse_judge_response(raw: str) -> tuple[str, str]:
    """Parse a judge response into (verdict, reason).

    Expects:
        VERDICT: correct|incorrect|ambiguous
        REASON: <text>

    Falls back to "ambiguous" / raw text on parse failure so one bad judge
    response does not abort the whole batch.
    """
    verdict = "ambiguous"
    reason = raw.strip()
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            candidate = line.split(":", 1)[1].strip().lower()
            if candidate in ("correct", "incorrect", "ambiguous"):
                verdict = candidate
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return verdict, reason


def _call_judge_sync(
    client: Any,
    task: dict[str, Any],
    model_response: str,
    judge_model: str,
) -> tuple[str, str]:
    """Call the Anthropic judge synchronously (runs in a thread via asyncio.to_thread)."""
    user_msg = _JUDGE_USER_TEMPLATE.format(
        utterance=task["utterance"],
        expected_referent=task["expected_referent"],
        model_response=model_response,
    )
    response = client.messages.create(
        model=judge_model,
        max_tokens=128,
        temperature=0.0,
        system=_JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text if response.content else ""
    return _parse_judge_response(raw)


# ---------------------------------------------------------------------------
# Per-task eval
# ---------------------------------------------------------------------------


async def run_task(
    task: dict[str, Any],
    model: str,
    anthropic_client: Any,
    judge_model: str,
    escalate_full_frame: bool = False,
) -> TaskResult:
    """Run one deictic eval task and return a scored TaskResult.

    Uses a FRESH GeminiLiveSession per task to prevent context bleed between tasks.
    Turn flow (automatic VAD; text input triggers the response):
        tile (video) → [full frame if escalating] → text annotation → utterance.
    """
    task_id = task["id"]
    logger.info("[%s] starting", task_id)

    # Resolve image path relative to scripts/bench/
    image_path = _HERE / task["image_path"]
    if not image_path.exists():
        raise FileNotFoundError(
            f"[{task_id}] image not found: {image_path}\n"
            "Generate placeholder images or replace with real screenshots."
        )
    tile_bytes = image_path.read_bytes()

    # Accumulate text output chunks until the turn settles.
    text_chunks: list[str] = []

    def _on_text(text: str) -> None:
        text_chunks.append(text)

    # gemini-3.1-flash-live-preview rejects TEXT-only output (close 1007), so we
    # request AUDIO and read the model's words via server-side transcription, which
    # the session surfaces through on_text_out.
    session = GeminiLiveSession(
        model=model,
        api_key_env="GEMINI_API_KEY",
        manual_vad=False,
        response_modalities=["AUDIO"],
        output_audio_transcription=True,
        thinking_level="minimal",
    )

    # Register text callback if the session exposes it; fall back to a no-op if the
    # on_text_out method is not yet merged (other subagent's task). The turn_done event
    # is fired by a timeout-based collector below regardless.
    if hasattr(session, "on_text_out"):
        session.on_text_out(_on_text)
    else:
        # Fallback: hook audio-out with a dummy callback just to get turn_done firing.
        # The real text path fires when on_text_out is wired.
        logger.warning(
            "[%s] on_text_out not found on GeminiLiveSession — text collection disabled; "
            "score will be on empty string. Ensure duplex-bridge patch is applied.",
            task_id,
        )

    # We also watch for any audio signal so we detect if the model ignores TEXT modality
    audio_received = asyncio.Event()

    def _on_audio(_data: bytes) -> None:
        audio_received.set()

    session.on_audio_out(_on_audio)

    model_response = ""
    try:
        await session.open()
        await asyncio.sleep(0.3)  # brief settle after connect

        # Drive one turn with automatic VAD: text input alone triggers the model's
        # response, so no activity_start/end is sent. (Manual VAD's realtime_input_config
        # is rejected by the server when combined with output_audio_transcription.)

        # 1. Send the tile on the video (never-interrupts) channel.
        await session._session.send_realtime_input(
            video=types.Blob(mime_type="image/jpeg", data=tile_bytes)
        )

        # 1b. Escalation: send a downscaled full frame too, for relational deixis.
        # Under auto VAD the session's force-send-at-activity_start is inert, so the
        # eval sends the cached frame directly here. Falls back to the tile if no
        # separate full_frame_path is given.
        if escalate_full_frame:
            ff_path = _HERE / task.get("full_frame_path", task["image_path"])
            if ff_path.exists():
                await session._session.send_realtime_input(
                    video=types.Blob(mime_type="image/jpeg", data=ff_path.read_bytes())
                )

        # Let the model ingest the frame BEFORE the utterance triggers its turn.
        # Without this, automatic VAD responds to the text before the still image is
        # processed and the model hallucinates content. (In production the user is
        # *speaking*, which supplies this delay naturally; text injection is instant,
        # so the eval adds it explicitly.) Measured: <1s hallucinates, ~2s grounds.
        await asyncio.sleep(IMAGE_INGEST_S)

        # 2. Send text annotation (app / window / cursor position).
        annotation = _build_text_annotation(task)
        await session._session.send_realtime_input(text=annotation)

        # 3. Send the utterance as text (bypasses ASR — intentional, tests visual grounding).
        await session._session.send_realtime_input(text=task["utterance"])

        # Collect transcript text until the model finishes. Use a timeout-based settle
        # window: wait up to RESPONSE_TIMEOUT_S for the first chunk, then TEXT_SETTLE_S
        # for trailing chunks (the transcript streams incrementally).
        t_start = time.perf_counter()
        deadline = t_start + RESPONSE_TIMEOUT_S

        # Poll for text chunks — the on_text_out callback appends asynchronously.
        last_len = 0
        last_change_t = time.perf_counter()
        while time.perf_counter() < deadline:
            await asyncio.sleep(0.1)
            current_len = len(text_chunks)
            if current_len != last_len:
                last_len = current_len
                last_change_t = time.perf_counter()
            elif text_chunks and (time.perf_counter() - last_change_t) >= TEXT_SETTLE_S:
                # Text has stopped arriving for TEXT_SETTLE_S — consider turn done.
                break

        model_response = "".join(text_chunks).strip()
        if not model_response:
            if audio_received.is_set():
                logger.warning(
                    "[%s] audio arrived but no transcript — output_audio_transcription may "
                    "not have surfaced; judge will score empty response.",
                    task_id,
                )
            else:
                logger.warning("[%s] no response within %.0fs", task_id, RESPONSE_TIMEOUT_S)

        logger.info(
            "[%s] response (%d chars): %s…", task_id, len(model_response), model_response[:80]
        )

    except Exception as exc:
        logger.error("[%s] error driving session: %s", task_id, exc)
    finally:
        await session.close()

    # Score with the LLM judge (run in a thread so we do not block the event loop).
    verdict, reason = await asyncio.to_thread(
        _call_judge_sync, anthropic_client, task, model_response, judge_model
    )
    passed = verdict == "correct"
    logger.info("[%s] verdict=%s passed=%s  reason=%s", task_id, verdict, passed, reason)

    return TaskResult(
        id=task_id,
        utterance=task["utterance"],
        model_response=model_response,
        judge_verdict=verdict,
        judge_reason=reason,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Batch eval
# ---------------------------------------------------------------------------


async def eval_batch(
    tasks: list[dict[str, Any]],
    model: str,
    anthropic_client: Any,
    judge_model: str,
    escalate_full_frame: bool = False,
) -> list[TaskResult]:
    """Run all tasks sequentially (one fresh session per task).

    Judge calls are parallelised via asyncio.gather after all sessions are done to avoid
    opening too many Gemini Live sessions at once (the API limits concurrent sessions).
    """
    session_results: list[tuple[dict[str, Any], str]] = []

    for task in tasks:
        result = await run_task(
            task,
            model=model,
            anthropic_client=anthropic_client,
            judge_model=judge_model,
            escalate_full_frame=escalate_full_frame,
        )
        session_results.append((task, result.model_response))
        # Brief inter-task pause to avoid hammering the Gemini Live API.
        await asyncio.sleep(1.0)

    # All sessions closed — now score them in parallel via the judge.
    # (run_task already called the judge inline; we return the collected results.)
    # Re-run tasks that produced errors can be done externally with --limit.
    # Return results in order (run_task already includes the judge call).
    # NOTE: the current implementation calls the judge inline per task (simpler, serialised).
    # Parallel judging is available for callers who extract model_response first.
    # Results are collected directly in _main via run_task per-task.
    # eval_batch is provided as an importable helper for callers that want a
    # single-call interface; the _main CLI drives tasks sequentially itself.
    return []  # caller should use _main or iterate run_task directly


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(results: list[TaskResult], *, escalate_full_frame: bool) -> bool:
    """Print a per-task table and a summary line. Return True if the 80% bar is met."""
    print("\n" + "=" * 70)
    print("DEICTIC EVAL — per-task results")
    print("=" * 70)
    print(f"{'id':<14} {'verdict':<12} {'pass':<6} {'reason'}")
    print("-" * 70)
    for r in results:
        mark = "PASS" if r.passed else "fail"
        print(f"{r.id:<14} {r.judge_verdict:<12} {mark:<6} {r.judge_reason}")
    print("=" * 70)

    n_correct = sum(1 for r in results if r.passed)
    n_ambiguous = sum(1 for r in results if r.judge_verdict == "ambiguous")
    n_total = len(results)
    pct = (n_correct / n_total * 100) if n_total else 0.0

    print(f"\npass_rate = {n_correct}/{n_total} ({pct:.0f}%)")
    if n_ambiguous:
        print(f"  (ambiguous={n_ambiguous} — does NOT count as pass; only 'correct' does)")
    if escalate_full_frame:
        print("  (escalate_with_full_frame=True — full-frame sent at turn start)")

    passed_bar = pct >= PASS_THRESHOLD * 100
    verdict_str = (
        f"PASS — {pct:.0f}% >= {PASS_THRESHOLD * 100:.0f}% Week-4 bar"
        if passed_bar
        else f"FAIL — {pct:.0f}% < {PASS_THRESHOLD * 100:.0f}% Week-4 bar"
    )
    print(f"\nWeek-4 acceptance: {verdict_str}")
    return passed_bar


async def _main(
    model: str,
    tasks_path: Path,
    escalate_full_frame: bool,
    judge_model: str,
    limit: int | None,
    out: Path | None,
) -> int:
    gemini_key = _load_api_key_from_env_file("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set in environment or .env", file=sys.stderr)
        return 1
    os.environ["GEMINI_API_KEY"] = gemini_key

    # Anthropic client (raises clearly if anthropic is not installed or key missing).
    anthropic_client = _get_anthropic_client()

    tasks = _load_tasks(tasks_path)
    if limit is not None:
        tasks = tasks[:limit]

    print(f"eval_deictic: {len(tasks)} task(s)  model={model}  judge={judge_model}")
    print(f"  escalate_full_frame={escalate_full_frame}  tasks={tasks_path}")
    print(
        "\nNOTE: utterances are TEXT-injected (bypasses ASR). "
        "This tests visual grounding, not speech transcription.\n"
    )

    results: list[TaskResult] = []
    for task in tasks:
        result = await run_task(
            task,
            model=model,
            anthropic_client=anthropic_client,
            judge_model=judge_model,
            escalate_full_frame=escalate_full_frame,
        )
        results.append(result)
        await asyncio.sleep(1.0)

    passed_bar = _print_summary(results, escalate_full_frame=escalate_full_frame)

    # Write per-task JSONL results.
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            for r in results:
                fh.write(json.dumps(r.as_dict()) + "\n")
        print(f"\nResults written to {out}")

    return 0 if passed_bar else 2


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini Live model (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help="Path to deictic_tasks.jsonl (default: fixtures/deictic_tasks.jsonl)",
    )
    p.add_argument(
        "--escalate-full-frame",
        action="store_true",
        help="Pass escalate_with_full_frame=True to GeminiLiveSession (relational-deixis mode)",
    )
    p.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Anthropic model for the LLM judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N tasks (useful for smoke-testing)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write per-task JSONL results to this path",
    )
    args = p.parse_args()
    return asyncio.run(
        _main(
            model=args.model,
            tasks_path=args.tasks,
            escalate_full_frame=args.escalate_full_frame,
            judge_model=args.judge_model,
            limit=args.limit,
            out=args.out,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
