#!/usr/bin/env bash
# warm-start各runの中間ckptを評価し「rep4の効率谷からの離脱の時系列」を取る。
# 維持(warm1) と 離脱(warm2,5) を iter 10..100 で eval → 単調劣化か非単調かを判定。
# 単調劣化なら early-stop/低LR で救える。非単調なら谷が本質的に不安定。
# 出力: results/eval_pf/truepf_trace512_warm{i}_iter{X}_s0.npz, /tmp/warmtraj.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE=512; CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml; NPROC="${NPROC:-32}"
RUNS="${RUNS:-1 2 5}"; ITERS="${ITERS:-010 030 050 070 090 100}"
MARK=/tmp/warmtraj.marker; rm -f "$MARK"
echo "[traj] START $(date +%H:%M:%S) RUNS=$RUNS ITERS=$ITERS"
for i in $RUNS; do
  base=$(find experiments/distributed_pcn/run_synth${SCALE}_warm${i}/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  for X in $ITERS; do
    ck="$base/iteration_${X}/model_iter_${X}.pth"
    if [ ! -f "$ck" ]; then echo "[traj] miss warm${i} iter${X}"; continue; fi
    out="results/eval_pf/truepf_trace${SCALE}_warm${i}_iter${X}_s0.npz"
    CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
      OUT="$out" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/warmtraj_${i}_${X}.out 2>&1
    echo "[traj] warm${i} iter${X} exit=$? $(date +%H:%M:%S)"
  done
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[traj] ALL DONE $(date +%H:%M:%S)"
