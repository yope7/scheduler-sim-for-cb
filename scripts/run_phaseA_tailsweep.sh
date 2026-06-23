#!/usr/bin/env bash
# Phase A: 裾レベルを 0.0→1.0 で振り、各 5seed・early-stop OFF・最終ckpt で崩壊の崖を特定。
# recipe は syncap(合成で安定だった設定)と同一にし、L=0 が安定ベースラインを再現するようにする。
set -u
cd /home/noguchi/scheduler-sim-for-cb
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 DISTRIBUTED_PCN_N_UPDATES=200"
LEVELS="${LEVELS:-0.0 0.2 0.4 0.6 0.8 1.0}"
for L in $LEVELS; do
  TAGN="tailL$(echo "$L" | tr -d '.')"
  echo "===== SWEEP L=$L TAG=$TAGN $(date +%H:%M:%S) ====="
  TAG="$TAGN" SCALE=512 REPS=5 NITER=100 MAXJOBS=3 NPROC=24 ESTOP=0 \
    LV="SYNTH_TAIL_LEVEL=$L $RECIPE" bash scripts/run_synth_tail.sh
done
echo "PHASE_A_SWEEP_DONE $(date +%H:%M:%S)"
