#!/usr/bin/env bash
# mid-core から Phase3 +25: cost≈0.5e6 帯（frac 0.04–0.12）を replay / wait-KL で厚く
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
REF_CKPT="${REF_CKPT:-$ROOT/experiments/distributed_pcn/midcore_20260530_233226/20260530_233228/iteration_040/model_iter_040.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/midcore_knee05_${STAMP}}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"

mkdir -p "$OUT" "$PF"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$REF_CKPT"
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_N_ITERATIONS=25
export DISTRIBUTED_PCN_EVAL_INTERVAL=25
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_ADAPTIVE_RETURN_NORMALIZATION=1

# mid-core 継続
export PCN_COMMAND_BALANCE=1
export PCN_TRAIN_MID_STEP_WEIGHT=6
export PCN_TRAIN_EVALIKE_STEP_WEIGHT=4
export PCN_TRAIN_EVALIKE_STEP_FRAC=0.15
export PCN_TRAIN_MID_PF_WEIGHT=4
export PCN_MID_BAND_COND_WEIGHT=0.08
export PCN_MID_BAND_COND_WAIT_LEVELS=6
export PCN_MID_BAND_COND_COST_LEVELS=6
export PCN_MID_BAND_COND_FOCUS_FRAC=0.077
export PCN_MID_BAND_COND_FOCUS_HALF_WIDTH_FRAC=0.04
export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=0.002
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08

# cost≈0.5e6 帯（@cmax≈6.5e6 → 0.26M–0.78M、中心≈0.5M）
export PCN_TRAIN_KNEE_PF_WEIGHT=12
export PCN_TRAIN_KNEE_STEP_WEIGHT=10
export PCN_TRAIN_KNEE_COST_MIN_FRAC=0.04
export PCN_TRAIN_KNEE_COST_MAX_FRAC=0.12

ray stop --force 2>/dev/null || true
echo "[knee05] OUT=$OUT INIT=$REF_CKPT"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
echo "[knee05] eval ckpt=$CKPT"
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF" --label midcore_knee05_iter205 --grid 16 --n-jobs 1024 --device cpu \
  --focus-cost-frac 0.077 --focus-r1-extra 10
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label midcore_knee05_iter205 --grid 10 \
  --output "$PF/pf_bulge_midcore_knee05.json"
cp "$(ls -t "$PF"/uniform_cmd_pf_midcore_knee05_*.png | head -1)" "$PF/SCALE1024_midcore_knee05.png"
cp "$CKPT" "$PF/SCALE1024_midcore_knee05.pth"
echo "Done. PF=$PF/SCALE1024_midcore_knee05.png"
