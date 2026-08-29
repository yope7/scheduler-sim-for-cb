#!/bin/bash
# H1即席テスト: 「大は小を兼ねる」仮説の検証
# win1024(trace1024学習) を フレッシュな256/512インスタンス(同一CSVの入れ子)に
# horizon初期値+指令スケールの線形縮小(NJ指定でeval_b2_compareが自動でやる)だけで適用し、
# 各スケール専用学習(cb05@256, win512@512)と同一プロトコルで比較する。
# 注: checkpointに焼き込まれた正規化(desired_return_scale/scaling_factor=1024基準)は
#     素のH1では上書きしない(=「1024エピソードの終盤」としての解釈をそのまま試す)。
set -e
cd /home/noguchi/scheduler-sim-for-cb
export PYTHONPATH=.

# 学習時と一致させるimport時env(trace系レシピ: FILM+FOURIER bands4, occupancy=1, urgency=0, defer=3行動)
# 鉄則: eval は FILM/Fourier/occupancy/defer を学習と全一致させる(defer忘れ→fc.2が[3]vs[2]で乱数のまま)
export OBS_URGENCY=0 OBS_OCCUPANCY=1 SCHEDULER_ALLOW_DEFER=1
export PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_HIDDEN_DIM=512
export NCMD=40 KSAMP=5 NPROC=16 SEEDS=0

W1024=experiments/distributed_pcn/run_synth1024_win/20260707_124235/iteration_100/model_iter_100.pth
W512=experiments/distributed_pcn/run_synth512_win/20260707_120617/iteration_100/model_iter_100.pth
CB05=experiments/distributed_pcn/run_synth256_cb05/20260707_030420/iteration_100/model_iter_100.pth
CFG256=experiments/distributed_pcn/job_trace_256_pcn.yml
CFG512=experiments/distributed_pcn/job_trace_512_pcn.yml

echo "=== [1/4] H1: win1024 -> 256 ==="
CKPT=$W1024 CFG=$CFG256 NJ=256 OUT=truepf_h1_1024to256.npz .venv/bin/python scripts/eval_b2_compare.py
echo "=== [2/4] control: cb05 -> 256 ==="
CKPT=$CB05 CFG=$CFG256 NJ=256 OUT=truepf_h1_cb05_256.npz .venv/bin/python scripts/eval_b2_compare.py
echo "=== [3/4] H1: win1024 -> 512 ==="
CKPT=$W1024 CFG=$CFG512 NJ=512 OUT=truepf_h1_1024to512.npz .venv/bin/python scripts/eval_b2_compare.py
echo "=== [4/4] control: win512 -> 512 ==="
CKPT=$W512 CFG=$CFG512 NJ=512 OUT=truepf_h1_win512_512.npz .venv/bin/python scripts/eval_b2_compare.py
echo "H1_ALL_DONE"
