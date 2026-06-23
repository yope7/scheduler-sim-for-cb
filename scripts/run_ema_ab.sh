#!/usr/bin/env bash
# 構造ロバスト化(2b): モデル重みEMA(Polyak) で「後期decay」を平滑化し崩壊率/分散を下げるか。
# baseline = 既存 ctbase128 (EMA=0)。本scriptは EMA>0 の2 arm を回す。
# 判定 = 学習中 uniform_cmd_pf の n_pf 分布(忠実指標) + 崩壊率 + CV。
# 使い方: SCALE=128 REPS=5 NITER=100 bash scripts/run_ema_ab.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-128}"; REPS="${REPS:-5}"; NITER="${NITER:-100}"
MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-24}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1"
MARK=/tmp/ema_ab.marker; rm -f "$MARK"
echo "[EMA-AB] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER CFG=$CFG"

for D in 0.999 0.9995; do
  DT=$(echo "$D" | tr -d '.')
  echo "[EMA-AB] === EMA_DECAY=$D (ema${DT}) ==="
  TAG=ema${DT} SCALE=$SCALE REPS=$REPS NITER=$NITER ESTOP=0 MAXJOBS=$MAXJOBS NPROC=$NPROC \
    CFG="$CFG" LV="$RECIPE PCN_EMA_DECAY=$D" bash scripts/run_synth_tail.sh
done

echo "[EMA-AB] === n_pf summary ==="
for grp in ctbase${SCALE} ema999${SCALE} ema9995${SCALE}; do
  echo "  --- $grp ---"
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_${grp}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    echo "    rep=$i n_pf=${np:-NA}"
  done
done
echo "EMA_AB_DONE $(date +%H:%M:%S)" > "$MARK"
echo "[EMA-AB] DONE $(date +%H:%M:%S)"
