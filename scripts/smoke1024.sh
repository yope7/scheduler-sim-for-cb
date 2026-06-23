#!/usr/bin/env bash
# 1024-job PCN smoke test: verify it runs (no obs-dim errors) and measure timing.
set -u
cd /home/noguchi/scheduler-sim-for-cb
OUT=experiments/distributed_pcn/smoke1024_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
echo "OUTPUT_DIR=$OUT"
DISTRIBUTED_PCN_CONFIG=experiments/distributed_pcn/job_trace_1024_pcn.yml \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0 \
DISTRIBUTED_PCN_N_ITERATIONS=4 \
DISTRIBUTED_PCN_EVAL_INTERVAL=4 \
DISTRIBUTED_PCN_INITIAL_EPISODES=32 \
DISTRIBUTED_PCN_EVAL_SAMPLES=48 \
PCN_CHOOSE_COMMANDS_MODE=pf_archive \
DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
echo "EXIT=$? OUTPUT_DIR=$OUT" | tee "$OUT/done.txt"
