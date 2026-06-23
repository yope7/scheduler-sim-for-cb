#!/usr/bin/env bash
# 根本修正(正規化) + PCN準拠の conditioning 感度損失(hinge-KL)のみ。
# cond_add / s_emb_dropout / value_repro は OFF。iter25/50/75/100 で安定性を追跡。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/rootfix_kl_${STAMP}}"

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

# PCN 準拠の感度損失のみ（同 obs で command を変えたら方策も変える）。他パッチは OFF。
export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_CONDITIONING_KL_MARGIN=0.08
export PCN_COND_ADD_SCALE=0
export PCN_S_EMB_DROPOUT=0
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8
export PCN_EVAL_PF_GRID=64
export PCN_EVAL_STOCHASTIC=0

echo "[rootfix_kl] OUT=$OUT  norm=scale-only+horizon  aid=hinge-KL only  eval@25"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
echo "Done. OUT=$OUT  exec=$EXEC"
