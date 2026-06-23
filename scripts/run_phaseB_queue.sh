#!/usr/bin/env bash
# Phase B: Phase A スイープ完了を待って、C3(レバレッジ観測)の決定打テストを trace512 で自動実行。
#  B1 lev1024 : cap1024(h1024, 崩壊3/5 のベースライン) + SCHEDULER_OBS_OCCUPANCY=1, EARLYSTOP=1
#               → 崩壊が止まるか（崖の増幅器に対する直接テスト）
#  B2 lev512f : 標準レシピ(h512, nup200) + OCCUPANCY, EARLYSTOP=0 最終ckpt
#               → early-stop に頼らず最終ckptが baseline(~51%) を超えるか（本質改善テスト）
set -u
cd /home/noguchi/scheduler-sim-for-cb
echo "[phaseB] waiting for Phase A sweep to finish..."
until grep -q PHASE_A_SWEEP_DONE /tmp/phaseA_sweep.log 2>/dev/null; do sleep 120; done
echo "[phaseB] Phase A done. START B1 $(date +%H:%M:%S)"

RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 DISTRIBUTED_PCN_N_UPDATES=200"

# B1: h1024 collapse amplifier + leverage obs (compare vs cap1024 baseline 38%/collapse3)
TAG=lev1024 SCALE=512 REPS=5 NITER=100 MAXJOBS=3 NPROC=24 \
  CFG=experiments/distributed_pcn/job_trace_512_pcn.yml \
  LV="$RECIPE PCN_HIDDEN_DIM=1024 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1" \
  bash scripts/run_512_final.sh

echo "[phaseB] B1 done. START B2 $(date +%H:%M:%S)"

# B2: standard recipe h512, EARLYSTOP=0 final ckpt + leverage obs (compare vs ~51% baseline)
TAG=lev512f SCALE=512 REPS=5 NITER=100 MAXJOBS=3 NPROC=24 ESTOP=0 \
  CFG=experiments/distributed_pcn/job_trace_512_pcn.yml \
  LV="$RECIPE SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1" \
  bash scripts/run_synth_tail.sh

echo "PHASE_B_QUEUE_DONE $(date +%H:%M:%S)"
