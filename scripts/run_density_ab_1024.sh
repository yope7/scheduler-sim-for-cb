#!/usr/bin/env bash
# 1024ジョブ density A/B (run1024_frozen レシピ準拠). baseline と density を GPU0/GPU1 で並列学習.
#   baseline : 手動帯フル(KNEE8/LOW_SLOPE6/LOW_WAIT10 + cost端setdefault8)  ← 現行frozenレシピ
#   density  : 手動帯 全OFF明示 + 密度逆数版(weight8 k2 a1)                  ← ユーザー提案一本
# 安定化(urgency/frozen_clone/heuristic種まき/mid-core)は両条件 共通基盤.
set -u
ROOT=/home/noguchi/scheduler-sim-for-cb
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml
NITER="${1:-100}"
BASE="$ROOT/experiments/distributed_pcn/density_ab_1024"
rm -rf "$BASE"; mkdir -p "$BASE"
echo "[1024ab] BASE=$BASE NITER=$NITER START=$(date +%H:%M:%S)"

run_one() {  # tag extra...
  local tag="$1"; shift
  local out="$BASE/$tag"; mkdir -p "$out"
  echo "[1024:$tag] START $(date +%H:%M:%S)"
  env DISTRIBUTED_PCN_CONFIG=$CFG DISTRIBUTED_PCN_OUTPUT_DIR="$out" \
    SCHEDULER_OBS_URGENCY=1 \
    PCN_FROZEN_PF_CLONE=1 PCN_FROZEN_PF_MAX=256 \
    DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS=0,100,1000,10000,100000,500000 \
    DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES=10 \
    DISTRIBUTED_PCN_SUPERVISED_EPOCHS=50 \
    DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
    DISTRIBUTED_PCN_INITIAL_EPISODES=32 \
    DISTRIBUTED_PCN_EVAL_INTERVAL=10 DISTRIBUTED_PCN_EVAL_SAMPLES=64 \
    DISTRIBUTED_PCN_REPLAY_TX_BUDGET=1200000 \
    PCN_USE_AMP=0 PCN_OBS_LOG=1 \
    PCN_PHASE1_SWEEP_TRAIN_WEIGHT=10 PCN_PF_COMMAND_ANCHORS=16 \
    PCN_CHOOSE_COMMANDS_MODE=pf_archive DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
    "$@" \
    PYTHONUNBUFFERED=1 "$PY" -u -m src.distributed.distributed_pcn_event \
      --conditioning --mid-core --no-viz > "$out/train.log" 2>&1
  echo "[1024:$tag] DONE exit=$? $(date +%H:%M:%S)"
}

# 順次実行(各runがRay経由で両GPU活用). CUDA_VISIBLE_DEVICES は使わない(Ray gpu_id 解決と競合するため).
# baseline = 現行 frozen レシピ(手動帯フル), density OFF
run_one baseline \
  PCN_TRAIN_KNEE_PF_WEIGHT=8 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=6 \
  PCN_TRAIN_LOW_WAIT_PF_WEIGHT=10 PCN_TRAIN_LOW_WAIT_MAX=0 PCN_TRAIN_LOW_WAIT_FRAC=0.30 \
  PCN_TRAIN_PF_DENSITY_WEIGHT=0
echo "[1024ab] baseline done -> start density $(date +%H:%M:%S)"

# density = 手動帯 全OFF + 密度逆数版一本
run_one density \
  PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 \
  PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0 PCN_TRAIN_MID_PF_WEIGHT=0 \
  PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0
echo "[1024ab] ALL DONE $(date +%H:%M:%S)"
