#!/usr/bin/env bash
# knee05 から Phase3 +35: 左上先端（cost 0〜0.18·cmax）の wait 応答を厚く学習
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
REF_CKPT="${REF_CKPT:-$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_midcore_knee05.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/low_slope_${STAMP}}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"

mkdir -p "$OUT" "$PF"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$REF_CKPT"
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_N_ITERATIONS=35
export DISTRIBUTED_PCN_EVAL_INTERVAL=35
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_ADAPTIVE_RETURN_NORMALIZATION=1

# mid-core 継続（knee は弱め、low_slope を主役）
export PCN_COMMAND_BALANCE=1
export PCN_TRAIN_MID_STEP_WEIGHT=5
export PCN_TRAIN_EVALIKE_STEP_WEIGHT=3
export PCN_TRAIN_EVALIKE_STEP_FRAC=0.15
export PCN_TRAIN_MID_PF_WEIGHT=3
export PCN_MID_BAND_COND_WEIGHT=0.05
export PCN_MID_BAND_COND_WAIT_LEVELS=5
export PCN_MID_BAND_COND_COST_LEVELS=4
export PCN_MID_BAND_COND_FOCUS_FRAC=0.077
export PCN_MID_BAND_COND_FOCUS_HALF_WIDTH_FRAC=0.04
export PCN_TRAIN_KNEE_PF_WEIGHT=6
export PCN_TRAIN_KNEE_STEP_WEIGHT=6
export PCN_TRAIN_KNEE_COST_MIN_FRAC=0.04
export PCN_TRAIN_KNEE_COST_MAX_FRAC=0.12
export PCN_CONDITIONING_SENS_WEIGHT=0.04
export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=0.0015
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08

# 左上先端（プラトー〜膝前）
export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=14
export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT=14
export PCN_TRAIN_LOW_SLOPE_COST_MIN_FRAC=0.0
export PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC=0.18
export PCN_LOW_BAND_COND_WEIGHT=0.14
export PCN_LOW_BAND_COND_WAIT_LEVELS=10
export PCN_LOW_BAND_COND_COST_LEVELS=10
export PCN_LOW_BAND_COND_KL_MARGIN=0.08

ray stop --force 2>/dev/null || true
echo "[low_slope] OUT=$OUT INIT=$REF_CKPT"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
echo "[low_slope] eval ckpt=$CKPT"
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF" --label low_slope_iter240 --grid 16 --n-jobs 1024 --device cpu \
  --low-tail-frac 0.18 --low-tail-extra 16 \
  --focus-cost-frac 0.077 --focus-r1-extra 8
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label low_slope_iter240 --grid 12 \
  --output "$PF/pf_bulge_low_slope.json"
cp "$(ls -t "$PF"/uniform_cmd_pf_low_slope_*.png | head -1)" "$PF/SCALE1024_low_slope.png"
cp "$CKPT" "$PF/SCALE1024_low_slope.pth"
echo "Done. PF=$PF/SCALE1024_low_slope.png"
