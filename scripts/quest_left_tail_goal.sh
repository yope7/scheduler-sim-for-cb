#!/usr/bin/env bash
# dual12 ベースでゴール達成まで試行（1本ずつ・bulge 評価・ベスト更新）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
# 進捗を /tmp/quest_left_tail.log にも残す（nohup 時）
exec > >(tee -a /tmp/quest_left_tail.log) 2>&1
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
LOG="$ROOT/experiments/distributed_pcn/quest_left_tail_log.jsonl"
mkdir -p "$PF"
touch "$LOG"

BEST_CKPT="${BEST_CKPT:-$PF/SCALE1024_left_tail_best.pth}"
if [[ ! -f "$BEST_CKPT" ]]; then
  BEST_CKPT="$PF/SCALE1024_midcore_knee05.pth"
fi
BEST_SCORE=1e18
if [[ -f "$PF/pf_bulge_left_tail_best.json" ]]; then
  read -r BEST_SCORE <<< "$($PYTHON scripts/pf_left_tail_goal.py "$PF/pf_bulge_left_tail_best.json" | $PYTHON -c "import sys,json;print(json.load(sys.stdin)['score'])")"
elif [[ -f "$ROOT/experiments/distributed_pcn/left_tail_dual12_20260531_221137/bulge_dual12.json" ]]; then
  read -r BEST_SCORE <<< "$($PYTHON scripts/pf_left_tail_goal.py "$ROOT/experiments/distributed_pcn/left_tail_dual12_20260531_221137/bulge_dual12.json" | $PYTHON -c "import sys,json;print(json.load(sys.stdin)['score'])")"
fi
if [[ -f "$LOG" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    _bs=$(echo "$line" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('score',1e18))")
    _bc=$(echo "$line" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('ckpt',''))")
    if [[ -n "$_bc" && -f "$_bc" ]] && awk -v a="$_bs" -v b="$BEST_SCORE" 'BEGIN{exit !(a<b)}'; then
      BEST_SCORE="$_bs"
      BEST_CKPT="$_bc"
    fi
  done < "$LOG"
fi
echo "[QUEST] start BEST=$BEST_CKPT score=$BEST_SCORE"

# name|mode|lbw|lr|nit|n_init_ep|low_step_w|dual_r1_frac
# r1_sweep 単独は dual12 から低域が悪化しやすい → dual/arc 中心
TRIALS=(
  "dual12b|dual|0.07|0.001|12|40|0|"
  "dual8|dual|0.05|0.0008|8|40|0|"
  "dual6|dual|0.04|0.0006|6|40|0|"
  "arc6|arc|0.04|0.0006|6|40|0|"
  "arc8|arc|0.05|0.0008|8|40|0|"
  "dual10|dual|0.06|0.001|10|40|1.5|"
  "dual10s|dual|0.06|0.001|10|40|2.0|0.58"
  "dual10m|dual|0.055|0.0007|10|40|1.2|0.55"
  "arc10|arc|0.06|0.001|10|40|0|"
  "dual12l|dual|0.065|0.0008|12|40|0|0.55"
  "arc12|arc|0.07|0.001|12|40|0|"
  "dual5|dual|0.035|0.0005|5|35|0|"
  "arc5|arc|0.035|0.0005|5|35|0|"
  "dual8l|dual|0.045|0.0007|8|40|0|"
  "arc8l|arc|0.045|0.0007|8|40|0|"
)

run_trial() {
  local name="$1" mode="$2" lbw="$3" lr="$4" nit="$5" ninit="$6"
  local lstep="${7:-0}" dual_r1="${8:-}"
  local stamp out exec ckpt bulge
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="$ROOT/experiments/distributed_pcn/quest_${name}_${stamp}"
  mkdir -p "$out"

  export DISTRIBUTED_PCN_OUTPUT_DIR="$out"
  export DISTRIBUTED_PCN_JOBS=1024
  export DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$BEST_CKPT"
  export DISTRIBUTED_PCN_LEARNING_RATE="$lr"
  export DISTRIBUTED_PCN_N_ITERATIONS="$nit"
  export DISTRIBUTED_PCN_EVAL_INTERVAL="$nit"
  export DISTRIBUTED_PCN_EVAL_SAMPLES=50
  export DISTRIBUTED_PCN_N_ACTORS=32
  export DISTRIBUTED_PCN_INITIAL_EPISODES="$ninit"
  export DISTRIBUTED_PCN_USE_EVENT_OBS=1
  export SCHEDULER_LEARNER_BITMAP=0
  export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
  export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
  export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
  export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
  export PCN_VALUE_REPRO_WEIGHT=0
  export PCN_COMMAND_BALANCE=1
  export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0
  export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT=0
  export PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC=0.18
  export PCN_LOW_BAND_COND_MODE="$mode"
  export PCN_LOW_BAND_COND_WEIGHT="$lbw"
  if [[ -n "$dual_r1" ]]; then
    export PCN_LOW_BAND_DUAL_R1_FRAC="$dual_r1"
  else
    unset PCN_LOW_BAND_DUAL_R1_FRAC || true
  fi
  if awk -v x="$lstep" 'BEGIN{exit !(x>0)}'; then
    export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT="$lstep"
  else
    export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT=0
  fi
  export PCN_LOW_BAND_COND_COST_LEVELS=12
  export PCN_LOW_BAND_COND_WAIT_LEVELS=10
  export PCN_LOW_BAND_COND_KL_MARGIN=0.07
  export PCN_TRAIN_MID_STEP_WEIGHT=3
  export PCN_TRAIN_MID_PF_WEIGHT=2
  export PCN_MID_BAND_COND_WEIGHT=0.02
  export PCN_TRAIN_KNEE_PF_WEIGHT=3
  export PCN_TRAIN_KNEE_STEP_WEIGHT=3
  export PCN_CONDITIONING_SENS_WEIGHT=0.025
  export PCN_CONDITIONING_SENS_WAIT_DR_THRESH=0.001
  export PCN_COND_ADD_SCALE=0.22
  export PCN_S_EMB_DROPOUT=0.06
  export PCN_EVAL_GAP_FEEDBACK=1
  export PCN_EVAL_GAP_FEEDBACK_GRID=12
  export PCN_EVAL_GAP_REF_GAP=1200
  export PCN_EVAL_GAP_BOOST_MAX=2.5

  echo "======== QUEST $name mode=$mode lbw=$lbw lr=$lr nit=$nit FROM=$BEST_CKPT ========"
  ray stop --force 2>/dev/null || true
  "$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$out/pcn_run.log"

  exec="$(find "$out" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
  ckpt="$(find "$exec" -name 'model_iter_*.pth' | sort -V | tail -1)"
  if [[ ! -f "$ckpt" ]]; then
    echo "[FAIL] $name no checkpoint"; return 1
  fi
  bulge="$out/bulge.json"
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS="${QUEST_BULGE_JOBS:-1024}" "$PYTHON" scripts/analyze_pf_bulge.py \
    --checkpoint "$ckpt" --label "$name" --grid 10 --output "$bulge"

  local metrics
  metrics="$($PYTHON scripts/pf_left_tail_goal.py "$bulge")"
  echo "$metrics" | $PYTHON -c "import sys,json; r=json.load(sys.stdin); r.update({'trial':'$name','ckpt':'$ckpt','bulge':'$bulge'}); print(json.dumps(r))" >> "$LOG"
  echo "$metrics"

  local sc goal_ok=1
  sc="$(echo "$metrics" | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['score'])")"
  if echo "$metrics" | $PYTHON -c "import sys,json; import sys; sys.exit(0 if json.load(sys.stdin).get('goal') else 1)"; then
    goal_ok=0
  fi

  if awk -v a="$sc" -v b="$BEST_SCORE" 'BEGIN{exit !(a<b)}'; then
    BEST_SCORE="$sc"
    BEST_CKPT="$ckpt"
    cp "$ckpt" "$PF/SCALE1024_left_tail_best.pth"
    cp "$bulge" "$PF/pf_bulge_left_tail_best.json"
    echo "[BEST] $name score=$sc knee_ratio=$(echo "$metrics" | $PYTHON -c "import sys,json;print(json.load(sys.stdin)['knee_ratio'])") ls_ratio=$(echo "$metrics" | $PYTHON -c "import sys,json;print(json.load(sys.stdin)['ls_ratio'])")"
  fi

  if [[ "$goal_ok" -eq 0 ]]; then
    echo "[GOAL] ACHIEVED trial=$name"
    PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
      --checkpoint "$ckpt" --replay-snapshot "$exec/learner_replay_snapshot.pkl.gz" \
      --output "$PF" --label "GOAL_${name}" --grid 16 --n-jobs 1024 --device cpu \
      --low-tail-frac 0.18 --low-tail-extra 16
    cp "$(ls -t "$PF"/uniform_cmd_pf_GOAL_"${name}"_*.png | head -1)" "$PF/SCALE1024_left_tail_goal.png"
    cp "$ckpt" "$PF/SCALE1024_left_tail_goal.pth"
    exit 0
  fi
  return 0
}

cycle=0
while true; do
  cycle=$((cycle + 1))
  echo "[QUEST] === cycle $cycle ==="
  for spec in "${TRIALS[@]}"; do
    IFS='|' read -r name mode lbw lr nit ninit lstep dual_r1 <<< "$spec"
    run_trial "$name" "$mode" "$lbw" "$lr" "$nit" "$ninit" "${lstep:-0}" "${dual_r1:-}" || true
  done
  echo "[QUEST] cycle $cycle done, best score=$BEST_SCORE — repeating"
done
