#!/usr/bin/env bash
# 交絡排除: synth256 を trace256と完全同一レシピ(run_trace256_lever.sh)で回す。
# CFG だけ synth に差し替え SYNTH_TAIL_LEVEL を付ける。trace不安定がworkload由来かrecipe由来かの切り分け。
# 使い方: scripts/run_synth256_tracerecipe.sh <TAG> "<extra env(CUDA/SYNTH_TAIL_LEVEL等)>" [NITER]
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAG="${1:?set TAG}"; EXTRA="${2:-}"; NITER="${3:-100}"
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml
OUT="experiments/distributed_pcn/run_synth256_${TAG}"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[synth_tracerecipe] TAG=$TAG EXTRA=[$EXTRA] START=$(date +%H:%M:%S)"
env $EXTRA \
PCN_FAST_UPDATE=1 \
PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_FILM=1 \
DISTRIBUTED_PCN_CONFIG=$CFG DISTRIBUTED_PCN_JOBS=256 DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
SCHEDULER_OBS_URGENCY=1 \
DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS=0,50,150,500 DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES=8 \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS=50 DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
PCN_EVAL_ACTOR_POOL=8 \
DISTRIBUTED_PCN_INITIAL_EPISODES=32 DISTRIBUTED_PCN_EVAL_INTERVAL=10 DISTRIBUTED_PCN_EVAL_SAMPLES=64 \
DISTRIBUTED_PCN_REPLAY_TX_BUDGET=1200000 \
PCN_TRAIN_KNEE_PF_WEIGHT=8 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=6 \
PCN_TRAIN_LOW_WAIT_PF_WEIGHT=10 PCN_TRAIN_LOW_WAIT_MAX=0 PCN_TRAIN_LOW_WAIT_FRAC=0.30 \
PCN_USE_AMP=0 PCN_OBS_LOG=1 \
PCN_PHASE1_SWEEP_TRAIN_WEIGHT=10 PCN_PF_COMMAND_ANCHORS=16 \
PCN_CHOOSE_COMMANDS_MODE=pf_archive DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
echo "[synth_tracerecipe] TAG=$TAG DONE=$(date +%H:%M:%S) exit=$?"
