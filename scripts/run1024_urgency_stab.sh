#!/usr/bin/env bash
# 安定化ラン: urgency obs + OOM修正 に加え、Phase-1 を「ヒューリスティック優勢のカリキュラム」にする。
#   狙い: 正しい urgency→action のデモを多数(優勢)与え、Phase-2 教師ありで方策を「良いベイスン」に置く
#         → 後続のランダム rollout に front-loading へ引き戻されにくくする（確率的崩壊の分散を低減）。
#   - INITIAL(random) を絞り、HEURISTIC を増やす（ただし random は全域&端点カバーのため残す）
#   - 閾値は中〜高cost域（front-loading被害が最大の領域）の正しいデモを厚く
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: run1024_urgency_stab.sh NAME}"
NITER="${2:-100}"
OUT=experiments/distributed_pcn/run1024_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml

echo "[run1024_urgency_stab] NAME=$NAME NITER=$NITER OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
SCHEDULER_OBS_URGENCY=1 \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS=0,10000,100000,500000,1000000,2000000 \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES=20 \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${DISTRIBUTED_PCN_SUPERVISED_EPOCHS:-80}" \
DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
DISTRIBUTED_PCN_INITIAL_EPISODES="${DISTRIBUTED_PCN_INITIAL_EPISODES:-16}" \
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
