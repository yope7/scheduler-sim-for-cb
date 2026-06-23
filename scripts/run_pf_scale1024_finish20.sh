#!/usr/bin/env bash
# cont70 の iter050（累積 Phase3≈80）から +20 iter → 累積 100。EVAL_INTERVAL=20 で ckpt 保存。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONT="$ROOT/experiments/distributed_pcn/scale1024_cont70_20260530_154043/20260530_154046"
REF_CKPT="${REF_CKPT:-$CONT/iteration_050/model_iter_050.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/scale1024_finish20_${STAMP}}"

mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$REF_CKPT"
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES=50
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=1
export DISTRIBUTED_PCN_N_ITERATIONS=20
export DISTRIBUTED_PCN_EVAL_INTERVAL=20
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
export PCN_ADAPTIVE_RETURN_NORMALIZATION=1
export PCN_CONDITIONING_SENS_WEIGHT=0.03
export PCN_CONDITIONING_KL_MARGIN=0.08
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8

ray stop --force 2>/dev/null || true
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$EXEC/iteration_020/model_iter_020.pth"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
PF_OUT="$ROOT/experiments/distributed_pcn/pf_best_current"
mkdir -p "$PF_OUT"
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF_OUT" --label scale1024_iter100 --grid 10 --n-jobs 1024 --device cpu
LATEST=$(ls -t "$PF_OUT"/uniform_cmd_pf_scale1024_iter100_*.png | head -1)
cp "$LATEST" "$PF_OUT/SCALE1024_iter100.png"
cp "$CKPT" "$PF_OUT/SCALE1024_iter100.pth"
echo "Saved $PF_OUT/SCALE1024_iter100.png"
