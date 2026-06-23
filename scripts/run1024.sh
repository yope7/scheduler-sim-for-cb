#!/usr/bin/env bash
# 1024-job clean single-shot PCN run with the best-known 24J config
# (ctrl band weights + sweep upweight + anchored command pool). Small initial_episodes
# to keep phase-1 tractable. Training only; eval (no-ref uniform-command PF) done separately.
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: run1024.sh NAME}"
NITER="${2:-100}"
OUT=experiments/distributed_pcn/run1024_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml

echo "[run1024] NAME=$NAME NITER=$NITER OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${DISTRIBUTED_PCN_SUPERVISED_EPOCHS:-50}" \
DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
DISTRIBUTED_PCN_INITIAL_EPISODES="${DISTRIBUTED_PCN_INITIAL_EPISODES:-32}" \
DISTRIBUTED_PCN_EVAL_INTERVAL=10 \
DISTRIBUTED_PCN_EVAL_SAMPLES=64 \
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
# archive trajectory
PYTHONPATH=. .venv/bin/python - "$EXEC" <<'PY' || true
import json,sys
from pathlib import Path
ex=Path(sys.argv[1]); f=ex/"training_iteration_summary.json"
if f.exists():
    rows=json.loads(f.read_text())["rows"]
    pts=[(r["iteration"],r["pareto_front_size"]) for r in rows if r.get("pareto_front_size") is not None]
    print("[archive_nd] "+" ".join(f"{it}:{sz}" for it,sz in pts))
PY
