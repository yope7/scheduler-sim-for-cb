#!/usr/bin/env bash
# 左上低コスト帯（Eval PF プラトー）: dual12 ベストから low-band dual で短い Phase3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
BASE="${BASE_CKPT:-$PF/SCALE1024_left_tail_best.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/left_tail_lowfix_${STAMP}}"
LABEL="${LABEL:-left_tail_lowfix}"

mkdir -p "$OUT" "$PF"
if [[ ! -f "$BASE" ]]; then
  echo "Missing BASE_CKPT: $BASE" >&2
  exit 1
fi

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$BASE"
export DISTRIBUTED_PCN_LEARNING_RATE="${LR:-0.0008}"
export DISTRIBUTED_PCN_N_ITERATIONS="${NIT:-15}"
export DISTRIBUTED_PCN_EVAL_INTERVAL="$DISTRIBUTED_PCN_N_ITERATIONS"
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES="${N_INIT:-40}"
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
export PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC=0.18
export PCN_LOW_BAND_COND_MODE=dual
export PCN_LOW_BAND_COND_WEIGHT="${LBW:-0.08}"
export PCN_LOW_BAND_COND_COST_LEVELS=14
export PCN_LOW_BAND_COND_WAIT_LEVELS=10
export PCN_LOW_BAND_COND_KL_MARGIN=0.06
export PCN_LOW_BAND_DUAL_R1_FRAC="${DUAL_R1:-0.65}"

export PCN_TRAIN_MID_STEP_WEIGHT=3
export PCN_TRAIN_MID_PF_WEIGHT=2
export PCN_MID_BAND_COND_WEIGHT=0.02
export PCN_TRAIN_KNEE_PF_WEIGHT=3
export PCN_TRAIN_KNEE_STEP_WEIGHT=3
export PCN_CONDITIONING_SENS_WEIGHT=0.025
export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=0.001
export PCN_COND_ADD_SCALE=0.22
export PCN_S_EMB_DROPOUT=0.06
# Eval 弱点帯域 → 次 iter 以降の replay step 重みを自動増幅（均等格子 PF）
export PCN_EVAL_GAP_FEEDBACK=1
export PCN_EVAL_GAP_FEEDBACK_GRID=12
export PCN_EVAL_GAP_REF_GAP=1200
export PCN_EVAL_GAP_BOOST_MAX=2.5

echo "[lowfix] OUT=$OUT BASE=$BASE nit=$DISTRIBUTED_PCN_N_ITERATIONS lbw=$PCN_LOW_BAND_COND_WEIGHT lr=$DISTRIBUTED_PCN_LEARNING_RATE"
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
cp "$CKPT" "$PF/SCALE1024_${LABEL}.pth"
cp "$BULGE" "$PF/pf_bulge_${LABEL}.json"

# ベスト更新
OLD_SCORE=$("$PYTHON" scripts/pf_left_tail_goal.py "$PF/pf_bulge_left_tail_best.json" 2>/dev/null | "$PYTHON" -c "import sys,json;print(json.load(sys.stdin)['score'])" 2>/dev/null || echo 1e18)
NEW_SCORE=$("$PYTHON" scripts/pf_left_tail_goal.py "$BULGE" | "$PYTHON" -c "import sys,json;print(json.load(sys.stdin)['score'])")
if awk -v a="$NEW_SCORE" -v b="$OLD_SCORE" 'BEGIN{exit !(a<b)}'; then
  cp "$CKPT" "$PF/SCALE1024_left_tail_best.pth"
  cp "$BULGE" "$PF/pf_bulge_left_tail_best.json"
  cp "$PNG" "$PF/SCALE1024_left_tail_best.png"
  echo "[lowfix] NEW BEST score=$NEW_SCORE (was $OLD_SCORE)"
fi
echo "[lowfix] PF=$PNG goal=$(grep -o '"goal": [^,]*' "$OUT/goal.json")"
