#!/usr/bin/env bash
# 24-job PCN stabilization experiment harness: clean single-shot train (100 iter) + score ALL checkpoints
# vs the 934-pt reference. All tuning knobs (PCN_TRAIN_*, LR, COMMAND_ALPHA, ...) are passed via the
# caller's exported environment so each experiment isolates a lever. Run in the background.
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: stab_run.sh NAME}"
OUT=experiments/distributed_pcn/stab_${NAME}
rm -rf "$OUT"; mkdir -p "$OUT"
REF=experiments/distributed_pcn/trace24_no_outlier_sampled_exhaustive_pf.npz
CFG=experiments/distributed_pcn/job_trace_24_scratch_pass.yml

echo "[stab_run] NAME=$NAME OUT=$OUT START=$(date +%H:%M:%S)"
echo "[stab_run] knob env:"; env | grep -E '^PCN_TRAIN_|^PCN_COMMAND_|^PCN_PF_COMMAND_|^PCN_RETURN_NORM_|^PCN_CHOOSE_|^PCN_COND|^DISTRIBUTED_PCN_LEARNING_RATE' | sort

# --- training (structural knobs only; all tuning knobs inherited from caller env) ---
DISTRIBUTED_PCN_CONFIG=$CFG \
DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0 \
DISTRIBUTED_PCN_N_ITERATIONS="${DISTRIBUTED_PCN_N_ITERATIONS:-100}" \
PCN_EVAL_GAP_BOOST_MAX="${PCN_EVAL_GAP_BOOST_MAX:-5.0}" \
PCN_CHOOSE_COMMANDS_MODE="${PCN_CHOOSE_COMMANDS_MODE:-pf_archive}" \
DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
TRAIN_EXIT=$?
EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "[stab_run] TRAIN_EXIT=$TRAIN_EXIT EXEC=$EXEC TRAIN_DONE=$(date +%H:%M:%S)"

# --- archive trajectory (collapse signal) ---
PYTHONPATH=. .venv/bin/python - "$EXEC" <<'PY' || true
import json,sys
from pathlib import Path
ex=Path(sys.argv[1]); f=ex/"training_iteration_summary.json"
if f.exists():
    rows=json.loads(f.read_text())["rows"]
    pts=[(r["iteration"],r["pareto_front_size"]) for r in rows if r.get("pareto_front_size") is not None]
    print("[archive_nd] "+" ".join(f"{it}:{sz}" for it,sz in pts))
PY

# --- eval ALL checkpoints vs reference ---
if [ -n "$EXEC" ] && [ "$TRAIN_EXIT" = "0" ]; then
  EXEC_DIR="$EXEC" REF_NPZ="$REF" DISTRIBUTED_PCN_CONFIG="$CFG" \
  EVAL_GRID=32 PCN_EVAL_COMMAND_MODE=pf_ref BEST_CKPT_TOPK=99 BEST_CKPT_MIN_ITER_FRAC="${BEST_CKPT_MIN_ITER_FRAC:-0}" \
  PCN_SCORE_LOW_WAIT_MAX=0 PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC=0 \
  PYTHONPATH=. .venv/bin/python -u -m scripts.eval_best_checkpoint > "$OUT/besteval.log" 2>&1
  echo "[stab_run] EVAL_EXIT=$?"
fi
echo "DONE NAME=$NAME EXEC=$EXEC TRAIN_EXIT=$TRAIN_EXIT END=$(date +%H:%M:%S)" | tee "$OUT/done.txt"
