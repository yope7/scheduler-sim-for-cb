#!/usr/bin/env bash
# 中域向け本質設計（mid-core）: ステップ replay + Archive wait 条件付け + command バランス
# midfix から Phase3 +40。追加学習の「量」ではなく conditioning/replay 分布の修正が目的。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REF_CKPT="${REF_CKPT:-$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_midfix.pth}"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/midcore_${STAMP}}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"

mkdir -p "$OUT" "$PF"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$REF_CKPT"
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_N_ITERATIONS=40
export DISTRIBUTED_PCN_EVAL_INTERVAL=20
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_ADAPTIVE_RETURN_NORMALIZATION=1

# mid-core（CLI --mid-core と同値）
export PCN_COMMAND_BALANCE=1
export PCN_TRAIN_MID_STEP_WEIGHT=6
export PCN_TRAIN_EVALIKE_STEP_WEIGHT=4
export PCN_TRAIN_EVALIKE_STEP_FRAC=0.15
export PCN_TRAIN_MID_PF_WEIGHT=4
export PCN_MID_BAND_COND_WEIGHT=0.06
export PCN_MID_BAND_COND_WAIT_LEVELS=5
export PCN_MID_BAND_COND_COST_LEVELS=4
export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=0.002
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08

ray stop --force 2>/dev/null || true
echo "[midcore] OUT=$OUT INIT=$REF_CKPT"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
echo "[midcore] eval ckpt=$CKPT"
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF" --label midcore_iter180 --grid 16 --n-jobs 1024 --device cpu
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label midcore_iter180 --grid 10 \
  --output "$PF/pf_bulge_midcore.json"
cp "$(ls -t "$PF"/uniform_cmd_pf_midcore_*.png | head -1)" "$PF/SCALE1024_midcore.png"
echo "Done. PF=$PF/SCALE1024_midcore.png bulge=$PF/pf_bulge_midcore.json"
