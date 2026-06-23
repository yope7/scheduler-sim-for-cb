#!/usr/bin/env bash
# scale1024 iter30 から Phase3 を 70 iter 追加（実質 Phase3 累計 100 iter 相当）。
# Phase2 スキップ・iter30 重みロードで時間短縮。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PREV="$ROOT/experiments/distributed_pcn/scale1024_20260530_095643/20260530_095645"
REF_CKPT="${REF_CKPT:-$PREV/iteration_030/model_iter_030.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/scale1024_cont70_${STAMP}}"

mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$REF_CKPT"
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=1
export DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0
export DISTRIBUTED_PCN_QUICK=0
export DISTRIBUTED_PCN_N_ITERATIONS=70
export DISTRIBUTED_PCN_EVAL_INTERVAL=25
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1

export PCN_ADAPTIVE_RETURN_NORMALIZATION=1
export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_CONDITIONING_KL_MARGIN=0.08
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8
export PCN_EVAL_PF_GRID=64
export PCN_EVAL_STOCHASTIC=0

echo "[scale1024_cont70] OUT=$OUT"
echo "  INIT=$REF_CKPT  +70 Phase3 iter (cumulative ~100)  Phase2=skip"
date +%H:%M:%S
ray stop --force 2>/dev/null || true
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"
date +%H:%M:%S

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
echo "Done. exec=$EXEC"

# 均等 command PF（累積 iter100 相当 = 本 run の iter070）
CKPT_FINAL="$EXEC/iteration_070/model_iter_070.pth"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
PF_OUT="$ROOT/experiments/distributed_pcn/pf_best_current"
mkdir -p "$PF_OUT"
for IT in 050 070; do
  CK="$EXEC/iteration_${IT}/model_iter_${IT}.pth"
  [ -f "$CK" ] || continue
  CUM=$((30 + 10#${IT#0}))
  LABEL="scale1024_cumiter${CUM}"
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 .venv/bin/python -u scripts/eval_uniform_command_pf.py \
    --checkpoint "$CK" --replay-snapshot "$SNAP" \
    --output "$PF_OUT" --label "$LABEL" --grid 10 --n-jobs 1024 --device cpu \
    || PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 .venv/bin/python -u scripts/eval_uniform_command_pf.py \
    --checkpoint "$CK" --output "$PF_OUT" --label "$LABEL" --grid 10 --n-jobs 1024 --device cpu
done
# canonical 保存（累積 iter100）
if [ -f "$CKPT_FINAL" ]; then
  LATEST=$(ls -t "$PF_OUT"/uniform_cmd_pf_scale1024_cumiter100_*.png 2>/dev/null | head -1)
  [ -n "$LATEST" ] && cp "$LATEST" "$PF_OUT/SCALE1024_iter100.png"
  cp "$CKPT_FINAL" "$PF_OUT/SCALE1024_iter100.pth"
  echo "Saved $PF_OUT/SCALE1024_iter100.png"
fi
