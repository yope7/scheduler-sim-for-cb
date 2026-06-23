#!/usr/bin/env bash
# cmd-track Loss の多seed A/B（[[pcn-cmd-track-loss]]）。
# 崩壊が出る正準レシピ(trace + FILM+Fourier+occupancy+N_UPDATES=200)を baseline とし、
# 唯一の差 = PCN_CMD_TRACK_WEIGHT を ON にして、崩壊seed率 / n_pf分布 が改善するか測る。
# 判定指標 = 学習中 uniform_cmd_pf の n_pf（[[pcn-eval-scale-mismatch]]で検証した忠実指標）。
# 使い方: SCALE=128 REPS=5 NITER=100 W=1.0 bash scripts/run_cmdtrack_ab.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-128}"; REPS="${REPS:-5}"; NITER="${NITER:-100}"; W="${W:-1.0}"
MAXJOBS="${MAXJOBS:-4}"; NPROC="${NPROC:-24}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1"
WTAG="ctw${W/./}"
MARK=/tmp/cmdtrack_ab.marker; rm -f "$MARK"
echo "[AB] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER W=$W CFG=$CFG"

# Arm A: baseline（cmd_track=0）
echo "[AB] === Arm A baseline (ctbase) ==="
TAG=ctbase SCALE=$SCALE REPS=$REPS NITER=$NITER ESTOP=0 MAXJOBS=$MAXJOBS NPROC=$NPROC \
  CFG="$CFG" LV="$RECIPE" bash scripts/run_synth_tail.sh

# Arm B: cmd-track ON
echo "[AB] === Arm B cmd-track W=$W ($WTAG) ==="
TAG=$WTAG SCALE=$SCALE REPS=$REPS NITER=$NITER ESTOP=0 MAXJOBS=$MAXJOBS NPROC=$NPROC \
  CFG="$CFG" LV="$RECIPE PCN_CMD_TRACK_WEIGHT=$W" bash scripts/run_synth_tail.sh

echo "[AB] === n_pf summary (trusted metric) ==="
for grp in ctbase${SCALE} ${WTAG}${SCALE}; do
  echo "  --- $grp ---"
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_${grp}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    echo "    rep=$i n_pf=${np:-NA}"
  done
done
echo "AB_DONE $(date +%H:%M:%S)" > "$MARK"
echo "[AB] DONE $(date +%H:%M:%S)"
