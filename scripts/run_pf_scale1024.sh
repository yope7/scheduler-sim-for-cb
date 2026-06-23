#!/usr/bin/env bash
# 極限スケール検証: n_jobs=1024。堅牢レシピ（正規化根本修正 + hinge-KL + cond_add + dropout, value_repro OFF）。
# 1024ステップ/エピソードのため、Phase1本数とeval本数を絞って現実的計算時間に収める。
# obs次元は固定(220)・正規化スケールはデータ由来でn_jobsに自動追従。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/scale1024_${STAMP}}"

mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024             # ← n_jobs=1024
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50   # 各Actor 50本 = 1600本（1024ステップで十分な掃引データ）
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=1
export DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0
export DISTRIBUTED_PCN_QUICK=0
export DISTRIBUTED_PCN_N_ITERATIONS=30
export DISTRIBUTED_PCN_EVAL_INTERVAL=10      # iter10/20/30 で保存（規模大ほどドリフトが早いので早期も取る）
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=50
export DISTRIBUTED_PCN_EVAL_SAMPLES=50       # 1024ステップ×本数のため評価本数を抑制
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1

export PCN_ADAPTIVE_RETURN_NORMALIZATION=1   # 中心化なし・スケールのみ（根本修正）
export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_CONDITIONING_KL_MARGIN=0.08
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08
export PCN_VALUE_REPRO_WEIGHT=0              # OFF（崩壊原因）
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8
export PCN_EVAL_PF_GRID=64
export PCN_EVAL_STOCHASTIC=0

echo "[scale1024] OUT=$OUT  n_jobs=1024  init_ep=50/actor  N_ITER=30 eval@10  SUP=50"
date +%H:%M:%S
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"
date +%H:%M:%S

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
echo "Done. OUT=$OUT  exec=$EXEC"
