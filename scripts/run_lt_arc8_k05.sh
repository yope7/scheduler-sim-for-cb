#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
K05="$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_midcore_knee05.pth"
OUT="$ROOT/experiments/distributed_pcn/lt_arc8_$(date +%Y%m%d_%H%M%S)"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
mkdir -p "$OUT" "$PF"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$K05"
export DISTRIBUTED_PCN_LEARNING_RATE=0.001
export DISTRIBUTED_PCN_N_ITERATIONS=8
export DISTRIBUTED_PCN_EVAL_INTERVAL=8
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_COMMAND_BALANCE=1
export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0
export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT=0
export PCN_LOW_BAND_COND_MODE=arc
export PCN_LOW_BAND_COND_WEIGHT=0.04
export PCN_TRAIN_MID_STEP_WEIGHT=4
export PCN_MID_BAND_COND_WEIGHT=0.03
export PCN_TRAIN_KNEE_PF_WEIGHT=4
export PCN_CONDITIONING_SENS_WEIGHT=0.03

ray stop --force 2>/dev/null || true
echo "OUT=$OUT"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
test -f "$CKPT" || { echo "NO CKPT"; exit 1; }
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label arc8 --grid 10 --output "$OUT/bulge.json"
"$PYTHON" scripts/pf_left_tail_goal.py "$OUT/bulge.json" | tee "$OUT/goal.json"
if "$PYTHON" scripts/pf_left_tail_goal.py "$OUT/bulge.json"; then
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
    --checkpoint "$CKPT" --replay-snapshot "$EXEC/learner_replay_snapshot.pkl.gz" \
    --output "$PF" --label goal_arc8 --grid 16 --n-jobs 1024 --device cpu \
    --low-tail-frac 0.18 --low-tail-extra 14
  cp "$(ls -t "$PF"/uniform_cmd_pf_goal_arc8_*.png | head -1)" "$PF/SCALE1024_left_tail_goal.png"
  cp "$CKPT" "$PF/SCALE1024_left_tail_best.pth"
fi
