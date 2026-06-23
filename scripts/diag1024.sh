#!/usr/bin/env bash
set -u
cd /home/noguchi/scheduler-sim-for-cb
OUT=experiments/distributed_pcn/run1024_diag
rm -rf "$OUT"; mkdir -p "$OUT"
export PCN_DIAG_BATCH=1
export DISTRIBUTED_PCN_CONFIG=experiments/distributed_pcn/job_trace_1024_pcn.yml
export DISTRIBUTED_PCN_OUTPUT_DIR=$OUT
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=2
export DISTRIBUTED_PCN_N_ITERATIONS=2
export DISTRIBUTED_PCN_INITIAL_EPISODES=4
export DISTRIBUTED_PCN_EVAL_INTERVAL=2
export DISTRIBUTED_PCN_EVAL_SAMPLES=16
export PCN_TRAIN_KNEE_PF_WEIGHT=8
export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=6
export PCN_TRAIN_LOW_WAIT_PF_WEIGHT=10
export PCN_TRAIN_LOW_WAIT_MAX=600
export PCN_USE_AMP=0
export PCN_PHASE1_SWEEP_TRAIN_WEIGHT=10
export PCN_PF_COMMAND_ANCHORS=16
export PYTHONUNBUFFERED=1
.venv/bin/python -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
echo "DIAG_DONE exit=$?" | tee "$OUT/done.txt"
