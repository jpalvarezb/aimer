"""Re-judge saved eval transcripts with Claude Sonnet via the ``claude`` CLI (Week 9).

Same 3-vote-majority shape as rejudge.py, but the judge calls go through ``claude -p``
(headless) instead of the Anthropic SDK — the repo's ANTHROPIC_API_KEY is dead, and the
authenticated Claude Code CLI is the working Sonnet access. Prompts and verdict parsing
are imported from eval_deictic.py so verdicts stay comparable across all judged runs.

Usage:
  uv run python scripts/bench/rejudge_subagent.py \
    --results scripts/bench/results/week9_decoupled.jsonl \
    --fixtures scripts/bench/fixtures/deictic_tasks_web_ax_resolved.jsonl --votes 3
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "duplex-bridge" / "src"))
sys.path.insert(0, str(_ROOT / "aimer-core" / "src"))

# Reuse the exact judge prompt + parser the eval uses, so verdicts are comparable.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_deictic", _HERE / "eval_deictic.py")
assert _spec is not None and _spec.loader is not None
ev = importlib.util.module_from_spec(_spec)
sys.modules["eval_deictic"] = ev
_spec.loader.exec_module(ev)


def _load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            out[record["id"]] = record
    return out


def _call_judge_cli(task: dict, response: str, model: str) -> tuple[str, str]:
    """One judge call through `claude -p --output-format json`; returns (verdict, reason)."""
    user_msg = ev._JUDGE_USER_TEMPLATE.format(
        utterance=task["utterance"],
        expected_referent=task["expected_referent"],
        keywords=", ".join(task.get("expected_referent_keywords", [])),
        model_response=response,
    )
    proc = subprocess.run(
        [
            "claude",
            "-p",
            user_msg,
            "--model",
            model,
            "--append-system-prompt",
            ev._JUDGE_SYSTEM_PROMPT,
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {proc.stderr[-300:]}")
    payload = json.loads(proc.stdout)
    return ev._parse_judge_response(str(payload.get("result", "")))


def _majority_judge(task: dict, response: str, model: str, votes: int) -> tuple[str, str]:
    verdicts = []
    last_reason = ""
    for _ in range(votes):
        try:
            verdict, reason = _call_judge_cli(task, response, model)
        except Exception as exc:  # noqa: BLE001 — one flaky vote must not kill the batch
            verdict, reason = "error", f"judge failed: {exc}"
        verdicts.append(verdict)
        last_reason = reason
    winner = collections.Counter(verdicts).most_common(1)[0][0]
    return winner, f"votes={verdicts} | {last_reason}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--judge-model", default="sonnet")
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    fixtures = _load_jsonl(args.fixtures)

    for res_path in args.results:
        results = _load_jsonl(res_path)
        ids = sorted(results)

        def _do(i: str, results: dict = results) -> tuple[str, str, str]:
            verdict, reason = _majority_judge(
                fixtures[i], results[i]["model_response"], args.judge_model, args.votes
            )
            return i, verdict, reason

        scored: dict[str, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, verdict, reason in pool.map(_do, ids):
                scored[i] = (verdict, reason)
                print(f"  {i}: {verdict}")

        n = len(ids)
        passed = sum(1 for i in ids if scored[i][0] == "correct")
        ambiguous = sum(1 for i in ids if scored[i][0] == "ambiguous")

        out_path = res_path.with_name(res_path.stem + "_rejudged.jsonl")
        with out_path.open("w") as fh:
            for i in ids:
                verdict, reason = scored[i]
                record = dict(results[i])
                record["rejudge_verdict"] = verdict
                record["rejudge_reason"] = reason
                record["rejudge_passed"] = verdict == "correct"
                fh.write(json.dumps(record) + "\n")

        print(f"\n=== {res_path.name} (judge=claude/{args.judge_model}, votes={args.votes}) ===")
        print(f"  pass: {passed}/{n} = {100 * passed / n:.1f}%  (ambiguous={ambiguous})")
        print(f"  -> {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
