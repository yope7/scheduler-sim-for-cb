#!/usr/bin/env bash
# 合成ジョブ(job_type=1) + urgency観測 の実験。NAME と JOBS(24/1024) を引数で受ける。
# 検証済み urgency レシピ（SCHEDULER_OBS_URGENCY=1 + 種まき + low-wait修正 + OOM修正）。
# 合成は env も agent も n_jobs を一致して使う（DISTRIBUTED_PCN_JOBS で指定）。
# checkpoint は model_iter_XXX.pth / final_model.pth が自動保存される。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: run_synthetic_urgency.sh NAME JOBS [NITER]}"
JOBS="${2:?usage: run_synthetic_urgency.sh NAME JOBS [NITER]}"
NITER="${3:-100}"
OUT=experiments/distributed_pcn/run_synth${JOBS}_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
CFG="${DISTRIBUTED_PCN_CONFIG:-experiments/distributed_pcn/job_synthetic_pcn.yml}"

# スケール依存 α: 件数正規化(各PF領域に均等な総重量)の強さを n_jobs で逓増させる。
# 大規模ほど cheap/expensive のエピソード件数が偏り expensive 支配→安い角を学べないので強い正規化(α→1)が要る。
# 小規模はバランスが元々良いので弱い正規化(α=0.5)で領域重視を保つ。非単調トレードを解く一本のルール。
# α=clip(0.5 + 0.5*(log2(JOBS)-8), 0.5, 1.0): 256→0.5, 512→1.0, 1024→1.0(検証実測で各スケール良好)。
PF_BALANCE_ALPHA_AUTO=$(awk -v j="$JOBS" 'BEGIN{a=0.5+0.5*(log(j)/log(2)-8); if(a<0.5)a=0.5; if(a>1)a=1; printf "%.3f",a}')

echo "[run_synth] NAME=$NAME JOBS=$JOBS NITER=$NITER OUT=$OUT ALPHA=$PF_BALANCE_ALPHA_AUTO START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_JOBS=$JOBS \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
SCHEDULER_OBS_URGENCY="${SCHEDULER_OBS_URGENCY:-1}" \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS="${DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS:-0,50,150,500,999999}" \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES=8 \
PCN_TRAIN_PF_BALANCE_REF="${PCN_TRAIN_PF_BALANCE_REF:-32}" \
PCN_TRAIN_PF_BALANCE_ALPHA="${PCN_TRAIN_PF_BALANCE_ALPHA:-$PF_BALANCE_ALPHA_AUTO}" \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${DISTRIBUTED_PCN_SUPERVISED_EPOCHS:-50}" \
DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
DISTRIBUTED_PCN_INITIAL_EPISODES="${DISTRIBUTED_PCN_INITIAL_EPISODES:-32}" \
DISTRIBUTED_PCN_EVAL_INTERVAL="${DISTRIBUTED_PCN_EVAL_INTERVAL:-10}" \
DISTRIBUTED_PCN_EVAL_SAMPLES="${DISTRIBUTED_PCN_EVAL_SAMPLES:-64}" \
DISTRIBUTED_PCN_REPLAY_TX_BUDGET=1200000 \
PCN_TRAIN_KNEE_PF_WEIGHT="${PCN_TRAIN_KNEE_PF_WEIGHT:-8}" PCN_TRAIN_LOW_SLOPE_PF_WEIGHT="${PCN_TRAIN_LOW_SLOPE_PF_WEIGHT:-6}" \
PCN_TRAIN_LOW_WAIT_PF_WEIGHT="${PCN_TRAIN_LOW_WAIT_PF_WEIGHT:-10}" PCN_TRAIN_LOW_WAIT_MAX=0 PCN_TRAIN_LOW_WAIT_FRAC=0.30 \
PCN_USE_AMP="${PCN_USE_AMP:-0}" PCN_OBS_LOG="${PCN_OBS_LOG:-1}" \
PCN_FROZEN_PF_CLONE="${PCN_FROZEN_PF_CLONE:-0}" PCN_FROZEN_PF_MAX="${PCN_FROZEN_PF_MAX:-128}" \
PCN_EMA_DECAY="${PCN_EMA_DECAY:-0}" \
PCN_LR_DECAY="${PCN_LR_DECAY:-0}" PCN_LR_DECAY_FINAL="${PCN_LR_DECAY_FINAL:-0.1}" \
PCN_PHASE1_SWEEP_TRAIN_WEIGHT=10 PCN_PF_COMMAND_ANCHORS=16 \
PCN_CHOOSE_COMMANDS_MODE="${PCN_CHOOSE_COMMANDS_MODE:-pf_archive}" DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz ${DISTRIBUTED_PCN_EXTRA_ARGS:-} > "$OUT/train.log" 2>&1
TRAIN_EXIT=$?
EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "DONE NAME=$NAME JOBS=$JOBS EXEC=$EXEC TRAIN_EXIT=$TRAIN_EXIT END=$(date +%H:%M:%S)" | tee "$OUT/done.txt"
