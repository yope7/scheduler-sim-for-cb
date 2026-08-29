#!/bin/bash
# N混合学習スクリーン本番: 混合1モデル(N∈{128,256,512}, 基準N=512) vs 各N専用学習。
# 3rep×4腕=12run を直列実行(OOM回避の掟)。腕を横に回してからrepを進める(早い段階で全腕が1本ずつ揃う)。
# 学習はGPU0のみ(GPU1は別作業が使用)。
set -e
cd /home/noguchi/scheduler-sim-for-cb
export CUDA_VISIBLE_DEVICES=0
NITER=${NITER:-100}

for r in 1 2 3; do
  echo "=== rep $r: nmix (mix 128,256,512 base512) ==="
  PCN_MIX_JOBS="128,256,512" bash scripts/run_synthetic_urgency.sh nmix_r$r 512 $NITER
  for n in 512 256 128; do
    echo "=== rep $r: dedicated N=$n ==="
    bash scripts/run_synthetic_urgency.sh nded${n}_r$r $n $NITER
  done
done
echo "NMIX_SCREEN_DONE"
