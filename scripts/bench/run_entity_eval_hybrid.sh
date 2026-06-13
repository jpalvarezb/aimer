#!/usr/bin/env bash
# Week-5 entity eval: LOCAL-4B primary with GEMINI fallback (vision-only).
#
# Phase 1 (local 4B): ONE tile per subprocess (full Metal teardown each exit) + a cooldown.
#   HARD-ABORTS on the first GPU fault (subprocess rc!=0 / no progress marker) so it never
#   hammers a wounded GPU into a compositor crash — that abort is the machine's safety net.
# Phase 2 (gemini fallback): GPU-free. Fills in EVERY tile local didn't finish — the one it
#   crashed on plus any it never reached — in a single process. Needs scripts/bench/.gemini_env.
#
# Both phases write to the SAME results file (ENTITY_EVAL_OUT) and tag each row with its
# backend, so the final summary shows the local-4B vs gemini split.
#
# Usage:
#   LOCAL_MAX_TILES=3 ./scripts/bench/run_entity_eval_hybrid.sh    # local-4B CANARY first
#   ./scripts/bench/run_entity_eval_hybrid.sh                      # full hybrid run
set -u
cd "$(dirname "$0")/../.."
OUT="${ENTITY_EVAL_OUT:-week5_entity_eval_4b_hybrid.json}"
COOLDOWN="${COOLDOWN_SECONDS:-30}"
MODEL_4B="${LOCAL_MODEL:-mlx-community/Qwen3-VL-4B-Instruct-4bit}"
LOCAL_MAX="${LOCAL_MAX_TILES:-99}"   # cap local attempts; small => canary
# Per-tile WALL-CLOCK timeout. A GPU *hang* can WEDGE the process (stuck in a Metal call,
# never exits) — without this, the driver's command substitution blocks forever (it once ran
# 4.5 h wedged). A healthy capped tile finishes well under this; a wedge is killed and treated
# as a fault, handing the tile to the Gemini fallback.
TILE_TIMEOUT="${TILE_TIMEOUT:-90}"
# Skip a faulted/wedged tile and continue (poison-guard skips it next loop), but stop after
# this many CONSECUTIVE faults — a run of back-to-back hangs means the GPU is wounded and we
# must not keep hammering it (that is what crashed the machine before).
MAX_CONSEC_FAULTS="${MAX_CONSEC_FAULTS:-3}"
LOG=scripts/bench/results/week5_entity_eval.log
: >"$LOG"

echo "PHASE 1: local 4B (vision-only), 1 tile/process, ${COOLDOWN}s cooldown, hard-abort on fault" | tee -a "$LOG"
echo "  model=$MODEL_4B  out=$OUT  local_max=$LOCAL_MAX" | tee -a "$LOG"
local_ok=0
faulted=0
consec=0
for i in $(seq 1 "$LOCAL_MAX"); do
  # Run the tile with a wall-clock watchdog: a wedged (hung-GPU) process gets SIGKILLed at
  # TILE_TIMEOUT and counts as a fault, instead of blocking the driver forever.
  tile_log=$(mktemp)
  ENTITY_EVAL_BACKEND=local ENTITY_EVAL_MODEL="$MODEL_4B" ENTITY_EVAL_OUT="$OUT" ENTITY_EVAL_BATCH=1 \
    uv run --no-sync python scripts/bench/eval_entities.py >"$tile_log" 2>&1 &
  pid=$!
  ( sleep "$TILE_TIMEOUT"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[watchdog] tile exceeded ${TILE_TIMEOUT}s — killing wedged GPU process" >>"$tile_log"
      kill -9 "$pid" 2>/dev/null; pkill -9 -P "$pid" 2>/dev/null; pkill -9 -f eval_entities.py 2>/dev/null
    fi ) &
  wd=$!
  wait "$pid"; rc=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  out=$(cat "$tile_log"); rm -f "$tile_log"
  printf '%s\n' "$out" | tee -a "$LOG"
  if printf '%s' "$out" | grep -q "ALL DONE"; then
    echo "local completed ALL tiles (no fallback needed)" | tee -a "$LOG"; break
  fi
  if [ "$rc" -ne 0 ] || ! printf '%s' "$out" | grep -q "REMAINING"; then
    consec=$((consec + 1))
    echo "LOCAL FAULT/WEDGE (rc=$rc) consecutive=$consec — tile crash-marked (skipped next loop); continuing" | tee -a "$LOG"
    if [ "$consec" -ge "$MAX_CONSEC_FAULTS" ]; then
      echo "hit $consec consecutive faults — GPU likely wounded, stopping local; Gemini can finish the rest" | tee -a "$LOG"
      faulted=1; break
    fi
    echo "--- post-fault cooldown ${COOLDOWN}s ---" | tee -a "$LOG"; sleep "$COOLDOWN"
    continue
  fi
  consec=0
  local_ok=$((local_ok + 1))
  if [ "$i" -lt "$LOCAL_MAX" ]; then echo "--- cooldown ${COOLDOWN}s ---" | tee -a "$LOG"; sleep "$COOLDOWN"; fi
done
echo "PHASE 1 done: local_ok=$local_ok faulted=$faulted" | tee -a "$LOG"

echo "PHASE 2: gemini fallback (vision-only) — filling any remaining/crashed tiles" | tee -a "$LOG"
if [ -f scripts/bench/.gemini_env ]; then . scripts/bench/.gemini_env; fi
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "SKIP gemini: GEMINI_API_KEY not set (cp scripts/bench/.gemini_env.example scripts/bench/.gemini_env and add your key). Local-only results stand." | tee -a "$LOG"
else
  ENTITY_EVAL_BACKEND=gemini ENTITY_EVAL_OUT="$OUT" ENTITY_EVAL_BATCH=999 \
    uv run --no-sync python scripts/bench/eval_entities.py 2>&1 | tee -a "$LOG"
fi
echo "HYBRID DONE (local_ok=$local_ok, faulted=$faulted)" | tee -a "$LOG"
