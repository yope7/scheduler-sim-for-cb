#!/usr/bin/env bash
# frozen-PF cloning ラン: 検証済み urgency レシピ + OOM修正 に PCN_FROZEN_PF_CLONE=1 を追加。
#   狙い（自己強化崩壊の遮断）: best-ever 非支配フロントを凍結保持し phase-3 教師に常時含める。
#   方策が劣化しても「劣化しない良いフロント」を behavior-clone し続ける → loss上昇/command無視崩壊を防ぐ。
#   ヒューリスティック種まきが初期の正しい front を供給し、frozen がそれを恒久化する（相補）。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: run1024_frozen.sh NAME}"
NITER="${2:-100}"
OUT=experiments/distributed_pcn/run1024_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml

echo "[run1024_frozen] NAME=$NAME NITER=$NITER OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
SCHEDULER_OBS_URGENCY=1 \
PCN_FROZEN_PF_CLONE=1 PCN_FROZEN_PF_MAX=256 \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS=0,100,1000,10000,100000,500000 \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES=10 \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${DISTRIBUTED_PCN_SUPERVISED_EPOCHS:-50}" \
DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
DISTRIBUTED_PCN_INITIAL_EPISODES="${DISTRIBUTED_PCN_INITIAL_EPISODES:-32}" \
DISTRIBUTED_PCN_EVAL_INTERVAL=10 \
DISTRIBUTED_PCN_EVAL_SAMPLES=64 \
DISTRIBUTED_PCN_REPLAY_TX_BUDGET=1200000 \
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
