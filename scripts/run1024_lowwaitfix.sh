#!/usr/bin/env bash
# run1024.sh と完全同一レシピ（n_jobs=24クロバーのまま＝amplogと同条件）で、
# 唯一 PCN_TRAIN_LOW_WAIT_MAX のみ修正する A/B テスト。
#   診断結果: amplog_b iter100 は achieved wait > commanded wait の系統的上方バイアス（ニーで~2x）。
#   原因: PCN_TRAIN_LOW_WAIT_MAX=600 は avg_wait[1.4e5,1.6e6] スケールで0点しか選ばず低wait強調が不活性、
#         かつ >0 なので scale-robust な FRAC=0.30 パーセンタイル選択を上書きしていた。
#   修正: MAX=0 にして FRAC=0.30（下位30%のwaitを重み10で強調）を有効化。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: run1024_lowwaitfix.sh NAME}"
NITER="${2:-100}"
OUT=experiments/distributed_pcn/run1024_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml

echo "[run1024_lowwaitfix] NAME=$NAME NITER=$NITER OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
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
