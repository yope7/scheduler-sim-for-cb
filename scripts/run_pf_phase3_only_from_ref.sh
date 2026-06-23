#!/usr/bin/env bash
# Phase2 スキップ + Phase3 直前に参照 ckpt（Phase1 後のランダム学習は残る）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
REF_CKPT="${REF_CKPT:-$ROOT/experiments/distributed_pcn/value_repro_20260529_194337/20260529_194340/iteration_100/model_iter_100.pth}"
OUT="$ROOT/experiments/distributed_pcn/pf_phase3_only_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$REF_CKPT"
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_N_ITERATIONS=50
export DISTRIBUTED_PCN_EVAL_INTERVAL=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=0
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
export DISTRIBUTED_PCN_LEARNING_RATE=3e-5
export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_VALUE_REPRO_WEIGHT=0.1
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08

ray stop --force 2>/dev/null || true
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"
EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
PYTHONPATH="$ROOT" PCN_PF_ZOOM_COST_MAX=80000 "$PYTHON" scripts/plot_eval_pf_values.py \
  --checkpoint "$CKPT" --replay-snapshot "$EXEC/learner_replay_snapshot.pkl.gz" \
  --output "$OUT/pf_eval" --label phase3_only --n-eval 200
