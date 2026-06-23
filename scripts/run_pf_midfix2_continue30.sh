#!/usr/bin/env bash
# midfix からさらに Phase3 +30（中域ギャップ詰め: 強め MidPF + wait-KL）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
REF_CKPT="${REF_CKPT:-$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_midfix.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/midfix2_${STAMP}}"

mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$REF_CKPT"
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_N_ITERATIONS=30
export DISTRIBUTED_PCN_EVAL_INTERVAL=30
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1

export PCN_ADAPTIVE_RETURN_NORMALIZATION=1
export PCN_CONDITIONING_SENS_WEIGHT=0.08
export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=0.002
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_TRAIN_MID_PF_WEIGHT=32
export PCN_TRAIN_MID_COST_MIN_FRAC=0.05
export PCN_TRAIN_MID_COST_MAX_FRAC=0.42

ray stop --force 2>/dev/null || true
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF" --label midfix2_iter170 --grid 16 --n-jobs 1024 --device cpu

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label midfix2_iter170 --grid 10 \
  --output "$PF/pf_bulge_midfix2.json"

cp "$(ls -t "$PF"/uniform_cmd_pf_midfix2_*.png | head -1)" "$PF/SCALE1024_midfix2.png"
echo "Done $CKPT"
