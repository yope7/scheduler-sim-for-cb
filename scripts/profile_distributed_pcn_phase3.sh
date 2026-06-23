#!/usr/bin/env bash
# Phase2 ハイパラは本番既定のまま、Phase3 のみ短く回して内訳を計測する。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/experiments/distributed_pcn/profile_${STAMP}"
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
export DISTRIBUTED_PCN_LOG_RAY_TRANSFER=1
# Phase2 本番既定（縮めない）
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100
export DISTRIBUTED_PCN_SUPERVISED_UPDATES_PER_EPOCH=100

echo "[profile] OUT=$OUT"
"$PYTHON" -u -m src.distributed.distributed_pcn --profile --no-enable-visualization 2>&1 | tee "$OUT/profile.log"
echo "[profile] log=$OUT/profile.log"
