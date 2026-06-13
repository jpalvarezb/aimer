#!/usr/bin/env bash
# Drive the crash-resumable Week-5 entity eval: a fresh process per batch so a cumulative
# Metal GPU fault can't sink the whole run (each process exit resets the GPU). Resumes from
# results/week5_entity_eval.json until the eval prints "ALL DONE".
set -u
cd "$(dirname "$0")/../.."
LOG=scripts/bench/results/week5_entity_eval.log
: >"$LOG"
for i in $(seq 1 20); do
  echo "=== batch $i ===" >>"$LOG"
  ENTITY_EVAL_BATCH=3 uv run --no-sync python scripts/bench/eval_entities.py >>"$LOG" 2>&1 || true
  if grep -q "ALL DONE" "$LOG"; then
    echo "complete after $i batch invocation(s)"
    break
  fi
done
tail -25 "$LOG"
