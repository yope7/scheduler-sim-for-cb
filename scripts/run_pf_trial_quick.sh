#!/usr/bin/env bash
# PF 試行: 1 本だけ学習（環境変数 TAG で切替）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TAG="${1:-no_adapt_norm}"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/pf_trial_${TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=0
export DISTRIBUTED_PCN_QUICK=0
export DISTRIBUTED_PCN_N_ITERATIONS=100
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

case "$TAG" in
  no_adapt_norm)
    export PCN_ADAPTIVE_RETURN_NORMALIZATION=0
    export PCN_COND_ADD_SCALE=0.25
    export PCN_S_EMB_DROPOUT=0.08
    ;;
  mild_cond)
    export PCN_ADAPTIVE_RETURN_NORMALIZATION=1
    export PCN_COND_ADD_SCALE=0.1
    export PCN_S_EMB_DROPOUT=0.04
    ;;
  strong_value_repro)
    export PCN_ADAPTIVE_RETURN_NORMALIZATION=0
    export PCN_COND_ADD_SCALE=0.15
    export PCN_S_EMB_DROPOUT=0.05
    export PCN_VALUE_REPRO_WEIGHT=0.2
    ;;
  *)
    echo "Unknown TAG=$TAG"; exit 1
    ;;
esac

ray stop --force 2>/dev/null || true
echo "[pf_trial] TAG=$TAG OUT=$OUT"
"$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"
EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
mkdir -p "$OUT/pf_eval"
PYTHONPATH="$ROOT" PCN_PF_ZOOM_COST_MAX=80000 "$PYTHON" scripts/plot_eval_pf_values.py \
  --checkpoint "$EXEC/iteration_100/model_iter_100.pth" \
  --replay-snapshot "$EXEC/learner_replay_snapshot.pkl.gz" \
  --output "$OUT/pf_eval" \
  --label "$TAG" --n-eval 200 2>&1 | tee "$OUT/pf_eval/plot.log"
cat "$OUT/pf_eval/eval_stats_${TAG}.json"
