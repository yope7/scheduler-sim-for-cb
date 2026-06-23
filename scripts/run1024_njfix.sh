#!/usr/bin/env bash
# run1024.sh の修正版（B+A）。amplog 系との before/after 比較用に別ファイルにしてある。
#  B: DISTRIBUTED_PCN_JOBS=1024 を明示 → agent側 n_jobs が 24 にクロバーされる致命バグを回避
#     （calibration / horizon scale 1/n_jobs / PCN_VALUE_WAIT_SCALE=wait_max*n_jobs が正しく1024で計算される）
#     + trace24 参照PF(PCN_REF_PF_NPZ)を空にして 1024 学習コマンドプールへの混入を排除
#  A: 主eval を EVAL_SAMPLES 64->256、live uniform PF の cost 軸解像度を上げて PF 点数/HVの過小評価を是正
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: run1024_njfix.sh NAME}"
NITER="${2:-100}"
OUT=experiments/distributed_pcn/run1024_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml

echo "[run1024_njfix] NAME=$NAME NITER=$NITER OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_JOBS=1024 \
PCN_REF_PF_NPZ= \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${DISTRIBUTED_PCN_SUPERVISED_EPOCHS:-50}" \
DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
DISTRIBUTED_PCN_INITIAL_EPISODES="${DISTRIBUTED_PCN_INITIAL_EPISODES:-32}" \
DISTRIBUTED_PCN_EVAL_INTERVAL=10 \
DISTRIBUTED_PCN_EVAL_SAMPLES=256 \
DISTRIBUTED_PCN_LIVE_UNIFORM_PF_GRID=24 \
PCN_TRAIN_KNEE_PF_WEIGHT=8 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=6 \
PCN_TRAIN_LOW_WAIT_PF_WEIGHT=10 PCN_TRAIN_LOW_WAIT_MAX=600 \
PCN_USE_AMP="${PCN_USE_AMP:-0}" PCN_OBS_LOG="${PCN_OBS_LOG:-1}" \
PCN_PHASE1_SWEEP_TRAIN_WEIGHT=10 PCN_PF_COMMAND_ANCHORS=16 \
PCN_CHOOSE_COMMANDS_MODE=pf_archive DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
TRAIN_EXIT=$?
EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "DONE NAME=$NAME EXEC=$EXEC TRAIN_EXIT=$TRAIN_EXIT END=$(date +%H:%M:%S)" | tee "$OUT/done.txt"
