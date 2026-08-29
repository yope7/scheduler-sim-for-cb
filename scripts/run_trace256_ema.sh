#!/usr/bin/env bash
# trace256 の振動対策B: run_trace256_lc.sh に PCN_EMA_DECAY を入れただけ。
# weight_decay(重みノルム抑制)は振動を悪化させた(反証)ので、eval重みをEMA平均して
# 「良いPF点が定着せず振動する」を直接抑える。save_model は EMA重みを保存するので
# iteration_XXX/model_iter_XXX.pth をそのまま learning_curve.py で評価できる。
# 使い方: scripts/run_trace256_ema.sh [NITER] [EMA]  例: scripts/run_trace256_ema.sh 100 0.999
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${1:-100}"
EMA="${2:-0.999}"
CFG=experiments/distributed_pcn/job_trace_256_pcn.yml
OUT="experiments/distributed_pcn/run_trace256_ema${EMA}"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[trace256_ema] NITER=$NITER EMA=$EMA OUT=$OUT START=$(date +%H:%M:%S)"
PCN_EMA_DECAY=$EMA \
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
echo "[trace256_ema] DONE=$(date +%H:%M:%S) exit=$?"
ls -d "$OUT"/*/iteration_* 2>/dev/null | wc -l
