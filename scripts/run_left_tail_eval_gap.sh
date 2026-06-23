#!/usr/bin/env bash
# dual12 ベスト + Eval 弱点帯域フィードバック（均等格子 PF → replay 重み）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
BASE="${BASE_CKPT:-$PF/SCALE1024_left_tail_best.pth}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/left_tail_eval_gap_${STAMP}}"
LABEL="${LABEL:-left_tail_eval_gap}"

mkdir -p "$OUT" "$PF"
if [[ ! -f "$BASE" ]]; then
  echo "Missing BASE_CKPT: $BASE" >&2
  exit 1
fi

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_JOBS=1024
export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$BASE"
export DISTRIBUTED_PCN_LEARNING_RATE="${LR:-0.0008}"
export DISTRIBUTED_PCN_N_ITERATIONS="${NIT:-12}"
export DISTRIBUTED_PCN_EVAL_INTERVAL="${EVAL_INT:-4}"
export DISTRIBUTED_PCN_N_ACTORS=32
export DISTRIBUTED_PCN_INITIAL_EPISODES="${N_INIT:-40}"

echo "[eval_gap] OUT=$OUT BASE=$BASE nit=$DISTRIBUTED_PCN_N_ITERATIONS eval_int=$DISTRIBUTED_PCN_EVAL_INTERVAL"
ray stop --force 2>/dev/null || true
"$PYTHON" -u -m src.distributed.distributed_pcn_event --left-tail --no-viz 2>&1 | tee "$OUT/pcn_run.log"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -name 'model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
BULGE="$OUT/bulge.json"

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
  --checkpoint "$CKPT" --label "$LABEL" --grid 10 --output "$BULGE"
"$PYTHON" scripts/pf_left_tail_goal.py "$BULGE" | tee "$OUT/goal.json"

PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
  --checkpoint "$CKPT" --replay-snapshot "$SNAP" \
  --output "$PF" --label "$LABEL" --grid 16 --n-jobs 1024 --device cpu \
  --low-tail-frac 0.18 --low-tail-extra 20

PNG="$(ls -t "$PF"/uniform_cmd_pf_"${LABEL}"_*.png | head -1)"
cp "$PNG" "$PF/SCALE1024_${LABEL}.png"

OLD_SCORE=$("$PYTHON" scripts/pf_left_tail_goal.py "$PF/pf_bulge_left_tail_best.json" 2>/dev/null | "$PYTHON" -c "import sys,json;print(json.load(sys.stdin)['score'])" 2>/dev/null || echo 1e18)
NEW_SCORE=$("$PYTHON" scripts/pf_left_tail_goal.py "$BULGE" | "$PYTHON" -c "import sys,json;print(json.load(sys.stdin)['score'])")
echo "[eval_gap] score old=$OLD_SCORE new=$NEW_SCORE"
if awk -v n="$NEW_SCORE" -v o="$OLD_SCORE" 'BEGIN{exit !(n<o)}'; then
  cp "$CKPT" "$PF/SCALE1024_left_tail_best.pth"
  cp "$BULGE" "$PF/pf_bulge_left_tail_best.json"
  cp "$PNG" "$PF/SCALE1024_left_tail_best.png"
  echo "[eval_gap] BEST updated"
fi
echo "[eval_gap] done PF=$PNG"
