#!/usr/bin/env bash
# front-loading 直接対策ラン。amplog レシピ + 3つの相補的介入:
#   1) SCHEDULER_OBS_URGENCY=1 : 現ジョブのオンプレ予測待ち(緊急度)を観測に1次元追加（220->221）。
#      → 方策が「混むジョブだけクラウド」の閾値判断を学べる信号を与える（front-loading の直接の叩き）。
#   2) PHASE1_HEURISTIC_THRESHOLDS : WaitTimeThreshold で「緊急度→行動」が正しい綺麗な低wait例を種まき。
#      → 方策が模倣すべき clean な教師（ランダムは偶然の整合しか持たない）。
#   3) PCN_TRAIN_LOW_WAIT_MAX=0 (+FRAC=0.30) : 低wait強調の不活性バグを修正し、その良例を重み付け。
# n_jobs は amplog 同条件（DISTRIBUTED_PCN_JOBS 未設定→24クロバーのまま。wait増幅が load-bearing なため）。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: run1024_urgency.sh NAME}"
NITER="${2:-100}"
OUT=experiments/distributed_pcn/run1024_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml

echo "[run1024_urgency] NAME=$NAME NITER=$NITER OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
SCHEDULER_OBS_URGENCY=1 \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS=0,100,1000,10000,100000,500000 \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES=6 \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${DISTRIBUTED_PCN_SUPERVISED_EPOCHS:-50}" \
DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
DISTRIBUTED_PCN_INITIAL_EPISODES="${DISTRIBUTED_PCN_INITIAL_EPISODES:-32}" \
DISTRIBUTED_PCN_EVAL_INTERVAL=10 \
DISTRIBUTED_PCN_EVAL_SAMPLES=64 \
PCN_TRAIN_KNEE_PF_WEIGHT=8 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=6 \
PCN_TRAIN_LOW_WAIT_PF_WEIGHT=10 PCN_TRAIN_LOW_WAIT_MAX=0 PCN_TRAIN_LOW_WAIT_FRAC=0.30 \
PCN_USE_AMP="${PCN_USE_AMP:-0}" PCN_OBS_LOG="${PCN_OBS_LOG:-1}" \
PCN_PHASE1_SWEEP_TRAIN_WEIGHT=10 PCN_PF_COMMAND_ANCHORS=16 \
PCN_CHOOSE_COMMANDS_MODE=pf_archive DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
TRAIN_EXIT=$?
EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "DONE NAME=$NAME EXEC=$EXEC TRAIN_EXIT=$TRAIN_EXIT END=$(date +%H:%M:%S)" | tee "$OUT/done.txt"
