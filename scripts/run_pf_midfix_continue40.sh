#!/usr/bin/env bash
# iter100 から Phase3 +40: 中域 PF 膨らみ対策（MidPF 重み + wait-command hinge-KL）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
REF_CKPT="${REF_CKPT:-$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_iter100.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/midfix_${STAMP}}"

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
export DISTRIBUTED_PCN_N_ITERATIONS=40
export DISTRIBUTED_PCN_EVAL_INTERVAL=20
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export DISTRIBUTED_PCN_EVAL_SAMPLES=50
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1

export PCN_ADAPTIVE_RETURN_NORMALIZATION=1
export PCN_CONDITIONING_SENS_WEIGHT=0.05
export PCN_CONDITIONING_KL_MARGIN=0.08
export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=5e-4
export PCN_COND_ADD_SCALE=0.25
export PCN_S_EMB_DROPOUT=0.08
export PCN_VALUE_REPRO_WEIGHT=0
export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8
# 中域 cost 帯（batch max の 6%〜38% ≒ 0.4M〜2.5M @1024）
export PCN_TRAIN_MID_PF_WEIGHT=16
export PCN_TRAIN_MID_COST_MIN_FRAC=0.06
export PCN_TRAIN_MID_COST_MAX_FRAC=0.38

ray stop --force 2>/dev/null || true
echo "[midfix] OUT=$OUT INIT=$REF_CKPT"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
PF_OUT="$ROOT/experiments/distributed_pcn/pf_best_current"

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF_OUT" --label midfix_iter140 --grid 16 --n-jobs 1024 --device cpu

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label midfix_iter140 --grid 16 \
  --output "$PF_OUT/pf_bulge_midfix.json"

cp "$(ls -t "$PF_OUT"/uniform_cmd_pf_midfix_iter140_*.png | head -1)" "$PF_OUT/SCALE1024_midfix.png"
echo "Done exec=$EXEC ckpt=$CKPT"
