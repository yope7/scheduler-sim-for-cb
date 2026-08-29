#!/bin/bash
# H1b: 転移時に指令正規化(desired_return_scale/center)とscaling_factorだけを
# ターゲットスケールのネイティブ値(専用モデルcheckpointの焼き込み値)に上書きして評価。
# 「指令正規化さえ合わせれば1024→256転移が復活するか」= Step 0(指令正規化)仮説の検証。
set -e
cd /home/noguchi/scheduler-sim-for-cb
export PYTHONPATH=.
export OBS_URGENCY=0 OBS_OCCUPANCY=1 SCHEDULER_ALLOW_DEFER=1
export PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_HIDDEN_DIM=512
export NCMD=40 KSAMP=5 NPROC=16 SEEDS=0

W1024=experiments/distributed_pcn/run_synth1024_win/20260707_124235/iteration_100/model_iter_100.pth
W512=experiments/distributed_pcn/run_synth512_win/20260707_120617/iteration_100/model_iter_100.pth
CB05=experiments/distributed_pcn/run_synth256_cb05/20260707_030420/iteration_100/model_iter_100.pth
CFG256=experiments/distributed_pcn/job_trace_256_pcn.yml
CFG512=experiments/distributed_pcn/job_trace_512_pcn.yml

echo "=== [1/2] H1b: win1024 -> 256 (正規化=cb05ネイティブ) ==="
CKPT=$W1024 OVERRIDE_DR_FROM=$CB05 CFG=$CFG256 NJ=256 OUT=truepf_h1b_1024to256.npz .venv/bin/python scripts/eval_b2_compare.py
echo "=== [2/2] H1b: win1024 -> 512 (正規化=win512ネイティブ) ==="
CKPT=$W1024 OVERRIDE_DR_FROM=$W512 CFG=$CFG512 NJ=512 OUT=truepf_h1b_1024to512.npz .venv/bin/python scripts/eval_b2_compare.py
echo "H1B_ALL_DONE"
