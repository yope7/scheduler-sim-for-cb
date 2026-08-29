#!/bin/bash
# N混合スクリーンのN別評価: 混合モデル(nmix_r1..3)を N∈{128,256,512} で、
# 専用モデル(nded{N}_r1..3)を自分のNで、同一プロトコル(eval_b2_compare)で評価。
# フラグは学習(run_synthetic_urgency.sh 既定+素PCN)と完全一致:
#   urgency=1 / occupancy=0 / FILM=0 / FOURIER=0 / defer無し / OBS_LOG=1
set -e
cd /home/noguchi/scheduler-sim-for-cb
export PYTHONPATH=.
export OBS_URGENCY=1 OBS_OCCUPANCY=0
export PCN_FILM=0 PCN_FOURIER_CMD=0 PCN_HIDDEN_DIM=512 PCN_OBS_LOG=1
export NCMD=40 KSAMP=5 NPROC=16 SEEDS=0
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml

ck() { ls "$1"/*/iteration_100/model_iter_100.pth | tail -1; }

for r in 1 2 3; do
  MIX=$(ck experiments/distributed_pcn/run_synth512_nmix_r$r)
  for n in 128 256 512; do
    echo "=== nmix_r$r -> N=$n ==="
    CKPT=$MIX CFG=$CFG NJ=$n OUT=truepf_nm_mix_r${r}_n${n}.npz .venv/bin/python scripts/eval_b2_compare.py
  done
  for n in 128 256 512; do
    DED=$(ck experiments/distributed_pcn/run_synth${n}_nded${n}_r$r)
    echo "=== nded${n}_r$r -> N=$n ==="
    CKPT=$DED CFG=$CFG NJ=$n OUT=truepf_nm_ded${n}_r${r}.npz .venv/bin/python scripts/eval_b2_compare.py
  done
done
echo "NMIX_EVAL_DONE"
