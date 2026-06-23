#!/usr/bin/env bash
# 単一プロセスの update ループ(bench_update_b2_128.py)を py-spy でプロファイル。
# Ray を介さないので idle worker 滞留もサンプリング遅延もなく、Learner update の純粋なスタックが採れる。
set -u
cd /home/noguchi/scheduler-sim-for-cb
N="${1:-220}"
ART=docs/figures/pyspy
mkdir -p "$ART"
PYSPY=/home/noguchi/.local/bin/py-spy
export PYTHONPATH=. DEVICE=cuda N="$N" WARMUP=8 PCN_FAST_UPDATE="${PCN_FAST_UPDATE:-0}"

echo "[harness-pyspy] N=$N FAST=$PCN_FAST_UPDATE flamegraph..."
"$PYSPY" record --rate 250 --format flamegraph --output "$ART/learner_flame.svg" -- \
  .venv/bin/python scripts/bench_update_b2_128.py 2>&1 | grep -aiE 'RESULT|bench|error|Trace' | tail -8
echo "[harness-pyspy] speedscope..."
"$PYSPY" record --rate 250 --format speedscope --output "$ART/learner.speedscope.json" -- \
  .venv/bin/python scripts/bench_update_b2_128.py 2>&1 | grep -aiE 'RESULT|bench|error' | tail -4
echo "[harness-pyspy] DONE"; ls -la "$ART"/learner_flame.svg "$ART"/learner.speedscope.json 2>/dev/null