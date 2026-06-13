"""Re-judge saved eval transcripts offline (no Gemini sessions) to separate judge noise
from model noise.

The per-task JSONL written by eval_deictic.py carries the model_response, so we can
re-score it without re-driving any Gemini Live session. This:
  - upgrades the judge to a stronger model, and
  - uses a 3-vote majority (temp 0) so a single flaky judge call can't flip a task.

Usage:
  uv run --package duplex-bridge python scripts/bench/rejudge.py \
    --results scripts/bench/results/week4_ax_off.jsonl scripts/bench/results/week4_ax_on.jsonl \
    --fixtures scripts/bench/fixtures/deictic_tasks_web_ax.jsonl \
    --judge-model claude-sonnet-4-6 --votes 3
"""

from __future__ import annotations

import argparse
import collections
import json
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
ev = importlib.util.module_from_spec(_spec)
sys.modules["eval_deictic"] = ev
_spec.loader.exec_module(ev)


def _load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _majority_judge(client, task, response, judge_model, votes):
    """Return (verdict, reason) via a majority vote of `votes` judge calls."""
    verdicts = []
    last_reason = ""
    for _ in range(votes):
        try:
            v, reason = ev._call_judge_sync(client, task, response, judge_model)
        except Exception as exc:  # noqa: BLE001
            v, reason = "error", f"judge failed: {exc}"
        verdicts.append(v)
        last_reason = reason
    winner = collections.Counter(verdicts).most_common(1)[0][0]
    return winner, f"votes={verdicts} | {last_reason}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--judge-model", default="claude-sonnet-4-6")
    p.add_argument("--votes", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    fixtures = _load_jsonl(args.fixtures)  # id -> task (expected_referent, utterance, ...)
    client = ev._get_anthropic_client()

    for res_path in args.results:
        results = _load_jsonl(res_path)
        ids = sorted(results)

        def _do(i: str, results=results):
            task = fixtures[i]
            resp = results[i]["model_response"]
            verdict, reason = _majority_judge(client, task, resp, args.judge_model, args.votes)
            return i, verdict, reason

        scored: dict[str, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, verdict, reason in pool.map(_do, ids):
                scored[i] = (verdict, reason)

        n = len(ids)
        passed = sum(1 for i in ids if scored[i][0] == "correct")
        amb = sum(1 for i in ids if scored[i][0] == "ambiguous")
        # Compare to the original single-judge verdict stored in the file.
        changed = [i for i in ids if (scored[i][0] == "correct") != results[i]["passed"]]

        out_path = res_path.with_name(res_path.stem + "_rejudged.jsonl")
        with out_path.open("w") as fh:
            for i in ids:
                v, reason = scored[i]
                rec = dict(results[i])
                rec["rejudge_verdict"] = v
                rec["rejudge_reason"] = reason
                rec["rejudge_passed"] = v == "correct"
                fh.write(json.dumps(rec) + "\n")

        print(f"\n=== {res_path.name} (judge={args.judge_model}, votes={args.votes}) ===")
        print(f"  re-judged pass: {passed}/{n} = {100 * passed / n:.1f}%  (ambiguous={amb})")
        print(f"  verdict changed vs original judge on {len(changed)} task(s): {changed}")
        print(f"  -> {out_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
