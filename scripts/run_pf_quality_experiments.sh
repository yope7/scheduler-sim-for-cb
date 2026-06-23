#!/usr/bin/env bash
# PF 品質の試行: 既存 ckpt の再評価 + 短い学習 ablation
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="$ROOT/experiments/distributed_pcn/pf_quality_trials_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT"

plot_pf() {
  local ckpt="$1" snap="$2" label="$3" sub="$4"
  local d="$OUT/$sub"
  mkdir -p "$d"
  PCN_PF_ZOOM_COST_MAX=80000 "$PYTHON" scripts/plot_eval_pf_values.py \
    --checkpoint "$ckpt" \
    --replay-snapshot "$snap" \
    --output "$d" \
    --label "$label" \
    --n-eval 200 2>&1 | tee "$d/plot.log"
}

REF_CKPT="$ROOT/experiments/distributed_pcn/value_repro_20260529_194337/20260529_194340/iteration_100/model_iter_100.pth"
REF_SNAP="$ROOT/experiments/distributed_pcn/value_repro_20260529_194337/20260529_194340/learner_replay_snapshot.pkl.gz"
NEW_CKPT="$ROOT/experiments/distributed_pcn/value_repro_20260529_234559/20260529_234602/iteration_100/model_iter_100.pth"
NEW_SNAP="$ROOT/experiments/distributed_pcn/value_repro_20260529_234559/20260529_234602/learner_replay_snapshot.pkl.gz"

echo "=== A: 既存 ckpt 再評価（eval 選択 fix 後） ===" | tee "$OUT/summary.txt"
plot_pf "$REF_CKPT" "$REF_SNAP" "ref_reeval" "A_ref_reeval"
plot_pf "$NEW_CKPT" "$NEW_SNAP" "new_reeval" "B_new_reeval"

run_train() {
  local tag="$1"
  shift
  local d="$OUT/train_$tag"
  mkdir -p "$d"
  echo "=== train $tag ===" | tee -a "$OUT/summary.txt"
  export DISTRIBUTED_PCN_OUTPUT_DIR="$d"
  export DISTRIBUTED_PCN_USE_EVENT_OBS=1
  export SCHEDULER_LEARNER_BITMAP=0
  export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
  export DISTRIBUTED_PCN_EVAL_DIAG=0
  export DISTRIBUTED_PCN_QUICK=0
  export DISTRIBUTED_PCN_N_ITERATIONS="${DISTRIBUTED_PCN_N_ITERATIONS:-100}"
  export DISTRIBUTED_PCN_EVAL_INTERVAL=50
  export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100
  export DISTRIBUTED_PCN_EVAL_SAMPLES=200
  export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
  export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
  export PCN_CONDITIONING_SENS_WEIGHT=0.03
  export PCN_CONDITIONING_KL_MARGIN=0.08
  export PCN_VALUE_REPRO_WEIGHT=0.1
  export PCN_TRAIN_COST_ENDPOINT_WEIGHT=8
  export PCN_EVAL_STOCHASTIC=0
  "$@" 
  "$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$d/pcn_run.log"
  local exec
  exec="$(find "$d" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
  plot_pf "$exec/iteration_100/model_iter_100.pth" "$exec/learner_replay_snapshot.pkl.gz" "$tag" "train_$tag"
  python3 -c "import json; s=json.load(open('$OUT/train_$tag/eval_stats_${tag}.json')); print('$tag', 'eval_PF', s['n_eval_pf'], 'unique', s['n_unique_values'], 'n_eval', s['n_eval_points'])" | tee -a "$OUT/summary.txt"
}

ray stop --force 2>/dev/null || true

# C: 適応正規化 OFF（command 潰れ抑制）
run_train "no_adapt_norm" env PCN_ADAPTIVE_RETURN_NORMALIZATION=0 PCN_COND_ADD_SCALE=0.25 PCN_S_EMB_DROPOUT=0.08

ray stop --force 2>/dev/null || true

# D: cond add / dropout 弱め
run_train "mild_cond" env PCN_ADAPTIVE_RETURN_NORMALIZATION=1 PCN_COND_ADD_SCALE=0.1 PCN_S_EMB_DROPOUT=0.04

echo "Done. OUT=$OUT" | tee -a "$OUT/summary.txt"
