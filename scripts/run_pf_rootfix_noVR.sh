#!/usr/bin/env bash
# 根本修正(正規化) + hinge-KL + cond_add + s_emb_dropout。value_repro は OFF。
# 仮説: 端点到達/密度の boost は cond_add 由来、iter100 崩壊は value_repro 由来。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/rootfix_noVR_${STAMP}}"

mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=1
export DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0
export DISTRIBUTED_PCN_QUICK=0
export DISTRIBUTED_PCN_N_ITERATIONS=100
export DISTRIBUTED_PCN_EVAL_INTERVAL=25
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100
export DISTRIBUTED_PCN_EVAL_SAMPLES=200
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1

export PCN_ADAPTIVE_RETURN_NORMALIZATION=1   # 中心化なし・スケールのみ（根本修正）

export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_CONDITIONING_KL_MARGIN=0.08
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08
export PCN_VALUE_REPRO_WEIGHT=0          # ← OFF（崩壊原因の検証）
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8
export PCN_EVAL_PF_GRID=64
export PCN_EVAL_STOCHASTIC=0

echo "[rootfix_noVR] OUT=$OUT  aids=hingeKL+cond_add+dropout  value_repro=OFF  eval@25"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
echo "Done. OUT=$OUT  exec=$EXEC"
