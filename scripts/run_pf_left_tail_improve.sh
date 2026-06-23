#!/usr/bin/env bash
# knee05 基準: 低域を壊さず膝だけ改善する短試行（ゴール達成まで繰り返し）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
K05="$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_midcore_knee05.pth"
K05_SNAP="$ROOT/experiments/distributed_pcn/midcore_knee05_20260531_023754/20260531_023757/learner_replay_snapshot.pkl.gz"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
LOG="$ROOT/experiments/distributed_pcn/left_tail_improve_log.jsonl"
mkdir -p "$PF"
touch "$LOG"

BEST_CKPT="$K05"
BEST_SCORE=$($PYTHON scripts/pf_left_tail_goal.py "$PF/pf_bulge_midcore_knee05.json" | $PYTHON -c "import sys,json;print(json.load(sys.stdin)['score'])")

# name|mode|lbw|lr|nit
trials=(
  "arc8|arc|0.04|0.001|8"
  "arc10|arc|0.05|0.0012|10"
  "r1micro8|r1_sweep|0.04|0.001|8"
  "dual8|dual|0.04|0.001|8"
  "arc12|arc|0.06|0.0015|12"
  "r1micro12|r1_sweep|0.05|0.0012|12"
)

run_one() {
  local name="$1" mode="$2" lbw="$3" lr="$4" nit="$5"
  local stamp out exec ckpt bulge
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="$ROOT/experiments/distributed_pcn/lt_${name}_${stamp}"
  mkdir -p "$out"

  export DISTRIBUTED_PCN_OUTPUT_DIR="$out"
  export DISTRIBUTED_PCN_JOBS=1024
  export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$BEST_CKPT"
  export DISTRIBUTED_PCN_LEARNING_RATE="$lr"
  export DISTRIBUTED_PCN_N_ACTORS=32
  export DISTRIBUTED_PCN_INITIAL_EPISODES=50
  export DISTRIBUTED_PCN_USE_EVENT_OBS=1
  export SCHEDULER_LEARNER_BITMAP=0
  export DISTRIBUTED_PCN_N_ITERATIONS="$nit"
  export DISTRIBUTED_PCN_EVAL_INTERVAL="$nit"
  export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
  export DISTRIBUTED_PCN_EVAL_SAMPLES=50
  export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
  export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
  export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
  export PCN_VALUE_REPRO_WEIGHT=0
  export PCN_ADAPTIVE_RETURN_NORMALIZATION=1
  export PCN_COMMAND_BALANCE=1
  # knee05 相当の mid/knee は弱く維持
  export PCN_TRAIN_MID_STEP_WEIGHT=4
  export PCN_TRAIN_MID_PF_WEIGHT=2
  export PCN_MID_BAND_COND_WEIGHT=0.03
  export PCN_MID_BAND_COND_FOCUS_FRAC=0.077
  export PCN_TRAIN_KNEE_PF_WEIGHT=4
  export PCN_TRAIN_KNEE_STEP_WEIGHT=4
  export PCN_CONDITIONING_SENS_WEIGHT=0.03
  export PCN_COND_ADD_SCALE=0.25
  export PCN_S_EMB_DROPOUT=0.08
  # 低域 replay 重みは付けない（低域悪化の主因）
  export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0
  export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT=0
  export PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC=0.18
  export PCN_LOW_BAND_COND_MODE="$mode"
  export PCN_LOW_BAND_COND_WEIGHT="$lbw"
  export PCN_LOW_BAND_COND_COST_LEVELS=10
  export PCN_LOW_BAND_COND_KL_MARGIN=0.08

  echo "======== $name mode=$mode lbw=$lbw lr=$lr nit=$nit FROM=$BEST_CKPT ========"
  ray stop --force 2>/dev/null || true
  "$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$out/pcn_run.log"

  exec="$(find "$out" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
  ckpt="$(find "$exec" -name 'model_iter_*.pth' | sort -V | tail -1)"
  bulge="$out/bulge.json"
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
    --checkpoint "$ckpt" --label "$name" --grid 10 --output "$bulge"

  local metrics goal_ok=1
  if metrics="$($PYTHON scripts/pf_left_tail_goal.py "$bulge")"; then
    goal_ok=0
  else
    metrics="$($PYTHON scripts/pf_left_tail_goal.py "$bulge" 2>/dev/null || echo '{}')"
  fi
  echo "$metrics" | $PYTHON -c "import sys,json; r=json.load(sys.stdin); r.update({'trial':'$name','ckpt':'$ckpt','bulge':'$bulge'}); print(json.dumps(r))" >> "$LOG"

  local sc beats
  sc="$(echo "$metrics" | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['score'])")"
  beats="$(echo "$metrics" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('beats_knee05_both',False))")"

  if awk -v a="$sc" -v b="$BEST_SCORE" 'BEGIN{exit !(a<b)}'; then
    BEST_SCORE="$sc"
    BEST_CKPT="$ckpt"
    cp "$ckpt" "$PF/SCALE1024_left_tail_best.pth"
    cp "$bulge" "$PF/pf_bulge_left_tail_best.json"
    echo "[BEST] $name score=$sc"
  fi

  if [[ "$goal_ok" -eq 0 ]]; then
    echo "[GOAL] $name"
    PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
      --checkpoint "$ckpt" --replay-snapshot "$exec/learner_replay_snapshot.pkl.gz" \
      --output "$PF" --label "goal_${name}" --grid 16 --n-jobs 1024 --device cpu \
      --low-tail-frac 0.18 --low-tail-extra 14
    cp "$(ls -t "$PF"/uniform_cmd_pf_goal_"${name}"_*.png | head -1)" "$PF/SCALE1024_left_tail_goal.png"
    exit 0
  fi
}

if [[ ! -f "$PF/pf_bulge_midcore_knee05.json" ]]; then
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
    --checkpoint "$K05" --replay-snapshot "$K05_SNAP" \
    --label knee05 --grid 10 --output "$PF/pf_bulge_midcore_knee05.json"
fi

for spec in "${trials[@]}"; do
  IFS='|' read -r name mode lbw lr nit <<< "$spec"
  run_one "$name" "$mode" "$lbw" "$lr" "$nit" || true
done
echo "[DONE] best score=$BEST_SCORE ckpt=$BEST_CKPT"
exit 1
