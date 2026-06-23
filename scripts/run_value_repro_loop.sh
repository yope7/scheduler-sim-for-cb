#!/usr/bin/env bash
# 学習 → PF 描画 → 品質チェック。失敗時は最大 MAX_ATTEMPTS まで再試行。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="$ROOT/experiments/distributed_pcn/value_repro_loop_${STAMP}"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  OUT="${BASE_OUT}/attempt_${attempt}"
  mkdir -p "$OUT"
  echo "======== attempt ${attempt}/${MAX_ATTEMPTS} OUT=$OUT ========"

  export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
  export DISTRIBUTED_PCN_USE_EVENT_OBS=1
  export SCHEDULER_LEARNER_BITMAP=0
  export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
  export DISTRIBUTED_PCN_EVAL_DIAG=1
  export DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0
  export DISTRIBUTED_PCN_QUICK=0
  export DISTRIBUTED_PCN_N_ITERATIONS=100
  export DISTRIBUTED_PCN_EVAL_INTERVAL=50
  export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100
  export DISTRIBUTED_PCN_EVAL_SAMPLES=200
  export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
  export PCN_MODEL_NAN_WARN_LIMIT=20

  if ! "$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz 2>&1 | tee "$OUT/pcn_run.log"; then
    echo "[attempt $attempt] training failed (exit code)"
    ray stop --force 2>/dev/null || true
    continue
  fi

  if grep -q "損失 = 0.0000" "$OUT/pcn_run.log" && grep -q "エポック 3 完了: 平均損失 = 0.0000" "$OUT/pcn_run.log"; then
    echo "[attempt $attempt] Phase2 collapsed (zero loss epochs) — retry"
    ray stop --force 2>/dev/null || true
    continue
  fi

  EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
  CKPT="$(find "$EXEC" -name 'model_iter_100.pth' | sort -V | tail -1)"
  SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
  PF="$OUT/pf_eval"
  mkdir -p "$PF"

  if [[ ! -f "$CKPT" || ! -f "$SNAP" ]]; then
    echo "[attempt $attempt] missing ckpt or snapshot"
    continue
  fi

  PCN_PF_ZOOM_COST_MAX=80000 PYTHONPATH="$ROOT" "$PYTHON" scripts/plot_eval_pf_values.py \
    --checkpoint "$CKPT" \
    --replay-snapshot "$SNAP" \
    --output "$PF" \
    --label "value_repro_iter100" \
    --n-eval 200 2>&1 | tee "$PF/plot.log"

  PNG="$(find "$PF" -name 'pareto_front_values_*.png' | sort | tail -1)"
  EVAL_PF=$(python3 -c "import json; print(json.load(open('$PF/eval_stats_value_repro_iter100.json'))['n_eval_pf'])" 2>/dev/null || echo 0)
  PHASE3_OK=0
  grep -q "フェーズ3完了" "$OUT/pcn_run.log" && PHASE3_OK=1
  NAN_OK=1
  if [[ $(grep -c "状態埋め込みsにNaN" "$OUT/pcn_run.log" 2>/dev/null || echo 0) -gt 100 ]]; then
    NAN_OK=0
  fi
  echo "[attempt $attempt] eval_PF=$EVAL_PF phase3_ok=$PHASE3_OK nan_ok=$NAN_OK png=$PNG"

  if [[ "$PHASE3_OK" == "1" && "$NAN_OK" == "1" && "$EVAL_PF" -ge 80 ]]; then
    echo "SUCCESS attempt=$attempt OUT=$OUT PF=$PNG"
    exit 0
  fi
  ray stop --force 2>/dev/null || true
done

echo "FAILED after $MAX_ATTEMPTS attempts. See $BASE_OUT"
exit 1
