#!/usr/bin/env bash
# 根本修正の検証: command 正規化を「中心化なし・目的ごとスケール + horizon を n_jobs で正規化」に変更。
# それ以外の conditioning パッチ（cond_add / s_emb_dropout / hinge-KL / value_repro）は OFF にして、
# 正規化の根本修正だけで PF が全域に出るかを確認する（最初から学習）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/rootfix_${STAMP}}"

mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=1
export DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0
export DISTRIBUTED_PCN_QUICK=0
export DISTRIBUTED_PCN_N_ITERATIONS=100
export DISTRIBUTED_PCN_EVAL_INTERVAL=50
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100
export DISTRIBUTED_PCN_EVAL_SAMPLES=200
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1

# --- 根本修正（既定で有効。明示しておく）---
export PCN_ADAPTIVE_RETURN_NORMALIZATION=1   # = 中心化なし・スケールのみ（コード側で実装）

# --- 補償パッチは OFF（根本修正だけで効くかを見る）---
export PCN_CONDITIONING_SENS_WEIGHT=0
export PCN_COND_ADD_SCALE=0
export PCN_S_EMB_DROPOUT=0
export PCN_VALUE_REPRO_WEIGHT=0
# 端点重みは PCN のデータ管理に相当（両端 12x 対称）。cost 端の追加重みは 8(=no-op) で対称維持。
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8

export PCN_EVAL_PF_GRID=64
export PCN_EVAL_STOCHASTIC=0

echo "[rootfix] OUT=$OUT  N_ITER=100 EVAL@50 SUP_EPOCH=100  patches=OFF  norm=scale-only+horizon"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
echo "Done. OUT=$OUT  exec=$EXEC"
