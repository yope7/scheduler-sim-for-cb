#!/usr/bin/env bash
# 本番投入前の1点スモーク: v10の実効フラグ(=main100と同一)に今日の高速化3点を乗せて2 iterだけ回す。
#   高速化3点: ①CSRキャッシュ(既定ON・自動) ②rawブロックカーネル(既定ON・Phase1)
#              ③配列エピソード(PCN_ACTOR_ARRAY_EPISODE=1・要指定)
# Phase1は1本だけにして短縮する(転送は新規16本ぶんなので Phase3 の比較には影響しない)。
# usage: smoke_v10_fast.sh [OUTDIR]
set -u
cd /home/noguchi/scheduler-sim-for-cb
OUT="${1:-/tmp/smoke_v10_fast}"
rm -rf "$OUT"; mkdir -p "$OUT"

source tools/cuda_env.sh

set -a
source experiments/distributed_pcn/run_j50000_v10_main100/v9_env_export.sh >/dev/null
set +a

export PCN_CMD_TRACK_WAIT_WEIGHT="0.3"
export PCN_COND_WAIT_ROBUST="logexpand"
export PCN_COND_WAIT_Z0="3e-2"
export PCN_WAIT_SENS_PROBE=1

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_N_ITERATIONS=2
export DISTRIBUTED_PCN_INITIAL_EPISODES=1
export DISTRIBUTED_PCN_EVAL_INTERVAL=99
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=2
export DISTRIBUTED_PCN_PROFILE=1
export PCN_ACTOR_ARRAY_EPISODE=1

echo "[smoke] START=$(date +%H:%M:%S) OUT=$OUT"
PYTHONUNBUFFERED=1 PYTHONPATH=. timeout 3000 .venv/bin/python -u \
  -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz \
  > "$OUT/train.log" 2>&1
echo "[smoke] exit=$? END=$(date +%H:%M:%S)"

grep -aE 'PCN_GPU_RAW_BLOCK|ブロック' "$OUT/train.log" | head -3
grep -aE 'PROFILE Phase3|per-iter mean' "$OUT/train.log" | tail -2
grep -aE 'Learner\+Actor並列待機|Actor実行:' "$OUT/train.log" | tail -2
grep -acE 'Traceback|Error|FAILED' "$OUT/train.log"
