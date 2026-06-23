#!/usr/bin/env bash
# Phase3 5iter の壁時計比較（Phase2 ハイパラ固定）。第1引数: on|off
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODE="${1:-on}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/experiments/distributed_pcn/bench_phase3_${MODE}_${STAMP}"
mkdir -p "$OUT"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_PROFILE=1
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=0
export DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0
export DISTRIBUTED_PCN_QUICK=0
export DISTRIBUTED_PCN_N_ITERATIONS=5
export DISTRIBUTED_PCN_EVAL_INTERVAL=999
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100
export DISTRIBUTED_PCN_SUPERVISED_UPDATES_PER_EPOCH=100

if [[ "$MODE" == "off" ]]; then
  export DISTRIBUTED_PCN_REPLAY_ZERO_COPY=0
  export DISTRIBUTED_PCN_ACTOR_RAY_PUT=0
else
  export DISTRIBUTED_PCN_REPLAY_ZERO_COPY=1
  export DISTRIBUTED_PCN_ACTOR_RAY_PUT=1
fi

echo "[bench] mode=$MODE OUT=$OUT"
/usr/bin/time -f 'elapsed_sec=%e' "$PYTHON" -u -m src.distributed.distributed_pcn --profile --no-enable-visualization 2>&1 | tee "$OUT/bench.log"
grep -E 'PROFILE Phase3|フェーズ2|フェーズ3|elapsed_sec' "$OUT/bench.log" || true
