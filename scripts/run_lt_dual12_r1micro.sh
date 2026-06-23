#!/usr/bin/env bash
# dual12 ベース: 膝は維持しつつ low_slope のみ r1_sweep で軽く改善
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BASE="$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_left_tail_best.pth"
OUT="$ROOT/experiments/distributed_pcn/lt_d12r1_$(date +%Y%m%d_%H%M%S)"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
mkdir -p "$OUT" "$PF"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$BASE"
export DISTRIBUTED_PCN_LEARNING_RATE=0.0008
export DISTRIBUTED_PCN_N_ITERATIONS=10
export DISTRIBUTED_PCN_EVAL_INTERVAL=10
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=40
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
export PCN_LOW_BAND_COND_MODE=r1_sweep
export PCN_LOW_BAND_COND_WEIGHT=0.05
export PCN_TRAIN_MID_STEP_WEIGHT=3
export PCN_MID_BAND_COND_WEIGHT=0.02
export PCN_CONDITIONING_SENS_WEIGHT=0.025
export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=0.001

ray stop --force 2>/dev/null || true
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$OUT/pcn_run.log"
EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --grid 10 --output "$OUT/bulge.json"
"$PYTHON" scripts/pf_left_tail_goal.py "$OUT/bulge.json" | tee "$OUT/goal.json"
if "$PYTHON" scripts/pf_left_tail_goal.py "$OUT/bulge.json"; then
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
    --checkpoint "$CKPT" --replay-snapshot "$EXEC/learner_replay_snapshot.pkl.gz" \
    --output "$PF" --label goal_d12r1 --grid 16 --n-jobs 1024 --device cpu \
    --low-tail-frac 0.18 --low-tail-extra 14
  cp "$(ls -t "$PF"/uniform_cmd_pf_goal_d12r1_*.png | head -1)" "$PF/SCALE1024_left_tail_goal.png"
fi
