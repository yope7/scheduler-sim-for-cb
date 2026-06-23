#!/usr/bin/env bash
# left_tail プロファイルの壁時計（短縮: phase3=10iter, phase2=20ep）。本番比は線形外挿のみ。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/experiments/distributed_pcn/bench_lt_${STAMP}"
mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_PROFILE=1
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export PYTHONPATH=.

echo "[bench] OUT=$OUT"
/usr/bin/time -f 'elapsed_sec=%e' "$PYTHON" -u -m src.distributed.distributed_pcn_event \
  --left-tail --no-viz --jobs 24 \
  --n-iterations 10 --initial-episodes 100 --eval-interval 10 --eval-samples 50 \
  --supervised-epochs 20 2>&1 | tee "$OUT/bench.log"
grep -E 'フェーズ[123]完了|総経過時間|elapsed_sec|PROFILE Phase3' "$OUT/bench.log" || true
