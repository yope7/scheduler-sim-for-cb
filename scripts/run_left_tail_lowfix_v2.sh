#!/usr/bin/env bash
# dual12 ベスト + 低域 step replay 重み + dual conditioning（プラトー対策 v2）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
BASE="$PF/SCALE1024_left_tail_best.pth"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/experiments/distributed_pcn/left_tail_lowfix2_${STAMP}"
LABEL="left_tail_lowfix2"

mkdir -p "$OUT" "$PF"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$BASE"
export DISTRIBUTED_PCN_LEARNING_RATE=0.001
export DISTRIBUTED_PCN_N_ITERATIONS=12
export DISTRIBUTED_PCN_EVAL_INTERVAL=12
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
export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT=2.0
export PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC=0.18
export PCN_LOW_BAND_COND_MODE=dual
export PCN_LOW_BAND_COND_WEIGHT=0.07
export PCN_LOW_BAND_COND_COST_LEVELS=14
export PCN_LOW_BAND_COND_KL_MARGIN=0.07
export PCN_LOW_BAND_DUAL_R1_FRAC=0.55

export PCN_TRAIN_MID_STEP_WEIGHT=3
export PCN_TRAIN_MID_PF_WEIGHT=2
export PCN_MID_BAND_COND_WEIGHT=0.02
export PCN_TRAIN_KNEE_PF_WEIGHT=3
export PCN_TRAIN_KNEE_STEP_WEIGHT=3
export PCN_CONDITIONING_SENS_WEIGHT=0.025
export PCN_COND_ADD_SCALE=0.22
export PCN_S_EMB_DROPOUT=0.06

ray stop --force 2>/dev/null || true
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
BULGE="$OUT/bulge.json"

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label "$LABEL" --grid 10 --output "$BULGE"
"$PYTHON" scripts/pf_left_tail_goal.py "$BULGE" | tee "$OUT/goal.json"

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF" --label "$LABEL" --grid 16 --n-jobs 1024 --device cpu \
  --low-tail-frac 0.18 --low-tail-extra 20

PNG="$(ls -t "$PF"/uniform_cmd_pf_"${LABEL}"_*.png | head -1)"
cp "$PNG" "$PF/SCALE1024_${LABEL}.png"
echo "[lowfix2] done PF=$PNG"
"$PYTHON" scripts/pf_left_tail_goal.py "$BULGE"
