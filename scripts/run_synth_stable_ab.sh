#!/usr/bin/env bash
# 自走ゴール: synth(必須)で安定PFを最小変更で。plain synth が既に安定かを5seedで確証し、
# EMA(Polyak)が普遍的最小レバー(synthも締める/害なし)になるか測る。
# 判定 = 学習中 uniform_cmd_pf の n_pf 分布(忠実指標) + CV + 崩壊率。
# 使い方: SCALE=256 REPS=5 NITER=100 bash scripts/run_synth_stable_ab.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-256}"; REPS="${REPS:-5}"; NITER="${NITER:-100}"
MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-24}"
CFG="${CFG:-experiments/distributed_pcn/job_synthetic_pcn.yml}"
MARK=/tmp/synth_stable_ab.marker; rm -f "$MARK"
echo "[SYN-AB] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER CFG=$CFG"

# Arm A: plain minimal (purepcn相当: FILM/Fourier/occ/tail なし, urgency off → obs220)
echo "[SYN-AB] === Arm A plain (synbase) ==="
TAG=synbase SCALE=$SCALE REPS=$REPS NITER=$NITER ESTOP=0 MAXJOBS=$MAXJOBS NPROC=$NPROC \
  CFG="$CFG" LV="SCHEDULER_OBS_URGENCY=0" bash scripts/run_synth_tail.sh

# Arm B: plain + EMA
echo "[SYN-AB] === Arm B plain+EMA0.999 (synema) ==="
TAG=synema SCALE=$SCALE REPS=$REPS NITER=$NITER ESTOP=0 MAXJOBS=$MAXJOBS NPROC=$NPROC \
  CFG="$CFG" LV="SCHEDULER_OBS_URGENCY=0 PCN_EMA_DECAY=0.999" bash scripts/run_synth_tail.sh

echo "SYNTH_STABLE_AB_DONE $(date +%H:%M:%S)" > "$MARK"
echo "[SYN-AB] DONE $(date +%H:%M:%S)"
