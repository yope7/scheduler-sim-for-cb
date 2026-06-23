#!/usr/bin/env bash
# 左上先端ゴールまで試行錯誤ループ（各試行: Phase3 短 run + bulge 評価）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
REF="${REF_CKPT:-$ROOT/experiments/distributed_pcn/pf_best_current/SCALE1024_midcore_knee05.pth}"
PF="$ROOT/experiments/distributed_pcn/pf_best_current"
LOG="$ROOT/experiments/distributed_pcn/left_tail_loop_log.jsonl"
mkdir -p "$PF" "$(dirname "$LOG")"
touch "$LOG"

BEST_CKPT="$REF"
BEST_SCORE=1e18
if [[ -f "$PF/SCALE1024_left_tail_best.pth" ]]; then
  BEST_CKPT="$PF/SCALE1024_left_tail_best.pth"
fi
BEST_SCORE=$($PYTHON scripts/pf_left_tail_goal.py "$PF/pf_bulge_midcore_knee05.json" 2>/dev/null | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['score'])" 2>/dev/null || echo "6000")
if [[ -f "$LOG" ]]; then
  read -r _BSC _BCK <<< "$($PYTHON -c "
import json
best_s, best_c = 1e18, ''
for line in open('$LOG'):
    if not line.strip(): continue
    r = json.loads(line)
    if r.get('score', 1e18) < best_s:
        best_s, best_c = r['score'], r.get('ckpt','')
if best_c: print(best_s, best_c)
" 2>/dev/null || echo "")"
  if [[ -n "${_BCK:-}" ]]; then
    BEST_SCORE="$_BSC"
    BEST_CKPT="$_BCK"
  fi
fi
echo "[LOOP] start BEST_SCORE=$BEST_SCORE ckpt=$BEST_CKPT"

trials=(
  "r1_20|r1_sweep|0.09|0.0015|20|0|0"
  "r1_18|r1_sweep|0.07|0.0012|18|0|0"
  "dual18|dual|0.05|0.0015|18|0|0"
  "dual10|dual|0.04|0.0012|10|0|0"
  "r1_15|r1_sweep|0.10|0.002|15|0|0"
  "arc8|arc|0.05|0.0012|8|0|0"
)

run_trial() {
  local name="$1" mode="$2" lbw="$3" lr="$4" nit="$5" lpf="$6" lsw="$7"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local out="$ROOT/experiments/distributed_pcn/left_tail_${name}_${stamp}"
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
  export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
  export PCN_VALUE_REPRO_WEIGHT=0
  export PCN_ADAPTIVE_RETURN_NORMALIZATION=1
  export PCN_COMMAND_BALANCE=1
  export PCN_TRAIN_MID_STEP_WEIGHT=4
  export PCN_TRAIN_MID_PF_WEIGHT=2
  export PCN_MID_BAND_COND_WEIGHT=0.04
  export PCN_MID_BAND_COND_FOCUS_FRAC=0.077
  export PCN_TRAIN_KNEE_PF_WEIGHT=4
  export PCN_TRAIN_KNEE_STEP_WEIGHT=4
  export PCN_CONDITIONING_SENS_WEIGHT=0.03
  export PCN_COND_ADD_SCALE=0.25
  export PCN_S_EMB_DROPOUT=0.08
  export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT="$lpf"
  export PCN_TRAIN_LOW_SLOPE_STEP_WEIGHT="$lsw"
  export PCN_TRAIN_LOW_SLOPE_COST_MAX_FRAC=0.18
  export PCN_LOW_BAND_COND_MODE="$mode"
  export PCN_LOW_BAND_COND_WEIGHT="$lbw"
  export PCN_LOW_BAND_COND_COST_LEVELS=10
  export PCN_LOW_BAND_COND_KL_MARGIN=0.08

  echo "======== TRIAL $name mode=$mode lbw=$lbw lr=$lr nit=$nit ========"
  ray stop --force 2>/dev/null || true
  "$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz 2>&1 | tee "$out/pcn_run.log"

  local exec ckpt bulge_json
  exec="$(find "$out" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
  ckpt="$(find "$exec" -name 'model_iter_*.pth' | sort -V | tail -1)"
  bulge_json="$out/bulge_${name}.json"
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
    --checkpoint "$ckpt" --label "${name}" --grid 10 --output "$bulge_json"

  local metrics goal_exit=1
  if metrics="$($PYTHON scripts/pf_left_tail_goal.py "$bulge_json" 2>/dev/null)"; then
    goal_exit=0
  else
    goal_exit=1
    metrics="${metrics:-{\"score\":1e18,\"goal\":false}}"
  fi
  local sc
  sc="$(echo "$metrics" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('score',1e18))" 2>/dev/null || echo 1e18)"

  echo "$metrics" | $PYTHON -c "
import sys, json
m = json.load(sys.stdin)
rec = {'trial': '$name', 'ckpt': '$ckpt', 'bulge': '$bulge_json', **m}
print(json.dumps(rec))
" >> "$LOG"

  if awk -v a="$sc" -v b="$BEST_SCORE" 'BEGIN{exit !(a<b)}'; then
    BEST_SCORE="$sc"
    BEST_CKPT="$ckpt"
    cp "$ckpt" "$PF/SCALE1024_left_tail_best.pth"
    echo "[BEST] $name score=$sc ckpt=$ckpt"
  fi

  if [[ "$goal_exit" -eq 0 ]]; then
    echo "[GOAL] 達成 trial=$name"
    PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" -u scripts/eval_uniform_command_pf.py \
      --checkpoint "$ckpt" --replay-snapshot "$exec/learner_replay_snapshot.pkl.gz" \
      --output "$PF" --label "left_tail_goal_${name}" --grid 16 --n-jobs 1024 --device cpu \
      --low-tail-frac 0.18 --low-tail-extra 14
    cp "$(ls -t "$PF"/uniform_cmd_pf_left_tail_goal_"${name}"_*.png | head -1)" "$PF/SCALE1024_left_tail_goal.png"
    exit 0
  fi
  return 0
}

# knee05 bulge が無ければ生成
if [[ ! -f "$PF/pf_bulge_midcore_knee05.json" ]]; then
  K05_SNAP="$ROOT/experiments/distributed_pcn/midcore_knee05_20260531_023754/20260531_023757/learner_replay_snapshot.pkl.gz"
  PYTHONPATH=. DISTRIBUTED_PCN_JOBS=1024 "$PYTHON" scripts/analyze_pf_bulge.py \
    --checkpoint "$REF" --replay-snapshot "$K05_SNAP" \
    --label knee05 --grid 10 --output "$PF/pf_bulge_midcore_knee05.json"
fi

for spec in "${trials[@]}"; do
  IFS='|' read -r name mode lbw lr nit lpf lsw <<< "$spec"
  name="${name// /}"
  mode="${mode// /}"
  run_trial "$name" "$mode" "$lbw" "$lr" "$nit" "$lpf" "$lsw" || true
done

echo "[LOOP] 全試行終了。ベスト score=$BEST_SCORE ckpt=$BEST_CKPT"
exit 1
