#!/usr/bin/env bash
# 指定ジョブ数で baseline(film) と b2(fourier) を同条件で新規学習し、seed0 で gap 比較図を作る。
# film128/fourier128 と同じレシピ（run_synthetic_urgency.sh）で、違いは PCN_FOURIER_CMD のみ＝clean A/B。
# 学習は Ray+GPU（CUDA_VISIBLE_DEVICES は設定しない）。eval は単体プロセスで GPU1 を使う。
# usage: scripts/run_b2_jobsize.sh JOBS [NITER]
set -u
cd /home/noguchi/scheduler-sim-for-cb
NJ="${1:?usage: run_b2_jobsize.sh JOBS [NITER]}"
NITER="${2:-100}"
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml
echo "[b2_jobsize] NJ=$NJ START=$(date +%H:%M:%S)"

# --- train: baseline(film, Fourier無し) と b2(fourier) ---
PCN_FILM=1 PCN_FOURIER_CMD=0 PCN_FOURIER_BANDS=4 bash scripts/run_synthetic_urgency.sh film${NJ} ${NJ} ${NITER}
PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 bash scripts/run_synthetic_urgency.sh fourier${NJ} ${NJ} ${NITER}

EXF=$(find experiments/distributed_pcn/run_synth${NJ}_film${NJ} -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
EXG=$(find experiments/distributed_pcn/run_synth${NJ}_fourier${NJ} -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "[b2_jobsize] NJ=$NJ EXF=$EXF EXG=$EXG"

# --- eval seed0（in-sample）both policies ---
PCN_FILM=1 PCN_FOURIER_CMD=1 DEVICE=cuda CUDA_VISIBLE_DEVICES=1 \
  CKPT="$EXG/iteration_${NITER}/model_iter_${NITER}.pth" CFG=$CFG NJ=$NJ SEEDS=0 NCMD=40 KSAMP=10 \
  OUT=truepf_fourier_${NJ}_s0.npz OBS_URGENCY=1 PYTHONPATH=. .venv/bin/python -u scripts/eval_b2_compare.py
PCN_FILM=1 PCN_FOURIER_CMD=0 DEVICE=cuda CUDA_VISIBLE_DEVICES=1 \
  CKPT="$EXF/iteration_${NITER}/model_iter_${NITER}.pth" CFG=$CFG NJ=$NJ SEEDS=0 NCMD=40 KSAMP=10 \
  OUT=truepf_film_${NJ}_s0.npz OBS_URGENCY=1 PYTHONPATH=. .venv/bin/python -u scripts/eval_b2_compare.py

# --- plot ---
FILM=truepf_film_${NJ}_s0.npz FOUR=truepf_fourier_${NJ}_s0.npz SEEDS=0 TAG="n_jobs=${NJ}" \
  OUT=pf_b2_compare_${NJ}_s0.png PYTHONPATH=. .venv/bin/python -u scripts/plot_b2_compare.py
echo "[b2_jobsize] NJ=$NJ DONE=$(date +%H:%M:%S)"
