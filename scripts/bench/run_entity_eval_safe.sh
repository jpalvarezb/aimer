#!/usr/bin/env bash
# HARDENED Week-5 entity-eval driver for GPU-fragile machines.
#
# This dev Mac's GPU faults under *sustained* back-to-back MLX VLM inference (it crashed the
# compositor twice). The fault correlates with continuous Metal command-buffer submission
# starving WindowServer, NOT with memory. The only cadence proven safe is occasional
# single-shot inference. This driver reproduces that cadence for a full eval:
#
#   - ONE tile per subprocess (ENTITY_EVAL_BATCH=1): every inference fully tears down the
#     Metal context on process exit, so nothing accumulates across tiles.
#   - A cooldown between tiles (COOLDOWN_SECONDS, default 30s): the GPU/compositor fully
#     recover before the next inference.
#   - MAX_TILES caps how many tiles this run will process (use 3 for the canary).
#   - Hard abort: if a subprocess exits non-zero or prints no progress marker, the GPU likely
#     faulted — stop immediately rather than hammering a wounded device.
#
# Usage:
#   MAX_TILES=3 ./scripts/bench/run_entity_eval_safe.sh        # canary
#   MAX_TILES=99 ./scripts/bench/run_entity_eval_safe.sh       # finish the rest
set -u
cd "$(dirname "$0")/../.."
LOG=scripts/bench/results/week5_entity_eval.log
COOLDOWN="${COOLDOWN_SECONDS:-30}"
MAX_TILES="${MAX_TILES:-99}"
: >"$LOG"
echo "driver: <=$MAX_TILES tile(s), ${COOLDOWN}s cooldown, 1 tile/process" | tee -a "$LOG"

processed=0
for i in $(seq 1 "$MAX_TILES"); do
  echo "=== tile attempt $i ===" >>"$LOG"
  out=$(ENTITY_EVAL_BATCH=1 uv run --no-sync python scripts/bench/eval_entities.py 2>&1)
  rc=$?
  printf '%s\n' "$out" | tee -a "$LOG"
  if printf '%s' "$out" | grep -q "ALL DONE"; then
    echo "COMPLETE: all tiles scored" | tee -a "$LOG"; break
  fi
  if [ "$rc" -ne 0 ] || ! printf '%s' "$out" | grep -q "REMAINING"; then
    echo "ABORT: tile attempt $i did not finish cleanly (rc=$rc) — GPU may have faulted; stopping" | tee -a "$LOG"
    exit 3
  fi
  processed=$((processed + 1))
  if [ "$i" -lt "$MAX_TILES" ]; then
    echo "--- cooldown ${COOLDOWN}s before next tile ---" | tee -a "$LOG"
    sleep "$COOLDOWN"
  fi
done
echo "DRIVER DONE: processed $processed tile(s) this run" | tee -a "$LOG"
