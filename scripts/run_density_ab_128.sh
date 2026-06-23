#!/usr/bin/env bash
# ビンレス PF 密度逆数サンプリングの A/B (n_jobs=128, scale128 レシピを踏襲).
#   A=baseline : cost端ピンポイント強調(=8), density OFF        ← 現行の手動帯方式
#   B=density  : cost端OFF, 密度逆数版(weight=8 k=2 alpha=1)    ← ユーザー提案(件数逆比の連続版)
# 同一レシピで該当フラグだけ切替 → 純粋比較。各 ~19分、順次実行。
set -uo pipefail
ROOT=/home/noguchi/scheduler-sim-for-cb
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
BASE="$ROOT/experiments/distributed_pcn/density_ab"
rm -rf "$BASE"; mkdir -p "$BASE"
echo "[ab] BASE=$BASE START=$(date +%H:%M:%S)"

run_one() {
  local tag="$1"; shift
  local out="$BASE/$tag"
  mkdir -p "$out"
  echo "[ab:$tag] START $(date +%H:%M:%S)  extra: $*"
  env \
    DISTRIBUTED_PCN_OUTPUT_DIR="$out" \
    DISTRIBUTED_PCN_JOBS=128 \
    DISTRIBUTED_PCN_USE_EVENT_OBS=1 \
    SCHEDULER_LEARNER_BITMAP=0 \
    DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0 \
    DISTRIBUTED_PCN_EVAL_DIAG=1 \
    DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0 \
    DISTRIBUTED_PCN_QUICK=0 \
    DISTRIBUTED_PCN_N_ITERATIONS=50 \
    DISTRIBUTED_PCN_EVAL_INTERVAL=25 \
    DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100 \
    DISTRIBUTED_PCN_EVAL_SAMPLES=200 \
    DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0 \
    DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1 \
    PCN_ADAPTIVE_RETURN_NORMALIZATION=1 \
    PCN_CONDITIONING_SENS_WEIGHT=0.03 \
    PCN_CONDITIONING_KL_MARGIN=0.08 \
    PCN_COND_ADD_SCALE=0.25 \
    PCN_S_EMB_DROPOUT=0.08 \
    PCN_VALUE_REPRO_WEIGHT=0 \
    PCN_EVAL_PF_GRID=64 \
    PCN_EVAL_STOCHASTIC=0 \
    "$@" \
    "$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz > "$out/train.log" 2>&1
  local rc=$?
  echo "[ab:$tag] DONE  $(date +%H:%M:%S)  exit=$rc"
  grep -aE '総経過時間|MO_HV|hypervolume' "$out/train.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -3
}

run_one baseline PCN_TRAIN_COST_ENDPOINT_WEIGHT=8 PCN_TRAIN_PF_DENSITY_WEIGHT=0
run_one density  PCN_TRAIN_COST_ENDPOINT_WEIGHT=0 PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0

echo "[ab] ALL DONE $(date +%H:%M:%S) BASE=$BASE"
