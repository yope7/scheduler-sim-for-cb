#!/usr/bin/env bash
# 現状ベスト戦略（dual12: low-band dual + mid-core/conditioning）で 0→100 iter フル学習
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/dual12_scratch100_${STAMP}}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
LABEL="dual12_scratch100"

mkdir -p "$OUT" "$PF"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
unset DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3 || true

ray stop --force 2>/dev/null || true
echo "[dual12_scratch100] OUT=$OUT (0→100 iter, --left-tail)"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --left-tail --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
echo "[dual12_scratch100] ckpt=$CKPT"

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF" --label "$LABEL" --grid 16 --n-jobs 1024 --device cpu \
  --low-tail-frac 0.18 --low-tail-extra 16

BULGE="$OUT/bulge_${LABEL}.json"
PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label "$LABEL" --grid 10 --output "$BULGE"

"$PYTHON" scripts/pf_left_tail_goal.py "$BULGE" | tee "$OUT/goal.json"

PNG="$(ls -t "$PF"/uniform_cmd_pf_"${LABEL}"_*.png | head -1)"
cp "$PNG" "$PF/SCALE1024_${LABEL}.png"
cp "$CKPT" "$PF/SCALE1024_${LABEL}.pth"
cp "$BULGE" "$PF/pf_bulge_${LABEL}.json"
echo "[dual12_scratch100] PF=$PF/SCALE1024_${LABEL}.png"
