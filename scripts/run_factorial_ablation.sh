#!/usr/bin/env bash
# 全数(2^4=16)アブレーション: 4機能 ON/OFF の全組合せで精度(n_pf)表を埋める。
# 機能: F=フーリエ(PCN_FOURIER_CMD) / W=重みサンプリング(中域PF重み群) / E=探索場所チューニング(PCN_EVAL_GAP_FEEDBACK) / D=後回しdefer(GIANT_DEFER)
# ケチる癖(cost_hold)は対象外=常にOFF。本物データ trace。各セル REPS seed。
# 使い方: SCALE=256 REPS=3 NITER=100 MAXJOBS=12 bash scripts/run_factorial_ablation.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-256}"; REPS="${REPS:-3}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-12}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"
MARK=/tmp/factorial_ablation.marker; rm -f "$MARK"

BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"

lv_for(){ # $1..$4 = F W E D (0/1)
  local F="$1" W="$2" E="$3" D="$4"; local lv="$BASE"
  if [ "$F" = 1 ]; then lv="$lv PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"; else lv="$lv PCN_FOURIER_CMD=0"; fi
  if [ "$W" = 0 ]; then lv="$lv $WOFF"; fi
  if [ "$E" = 0 ]; then lv="$lv PCN_EVAL_GAP_FEEDBACK=0"; fi
  if [ "$D" = 1 ]; then lv="$lv SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"; fi
  echo "$lv"
}

echo "[fct] START $(date +%H:%M:%S) 16セル×${REPS}seed SCALE=$SCALE NITER=$NITER MAXJOBS=$MAXJOBS"
run_cell(){ local tag="$1" lv="$2" i="$3"; local t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${tag}_${i}" "$SCALE" "$NITER" > /tmp/fct_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[fct] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for F in 0 1; do for W in 0 1; do for E in 0 1; do for D in 0 1; do
  tag="fc${F}${W}${E}${D}"
  lv="$(lv_for $F $W $E $D)"
  for i in $(seq 1 "$REPS"); do
    while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    run_cell "$tag" "$lv" "$i" &
  done
done; done; done; done
wait
echo "[fct] === 精度表 n_pf (F W E D = フーリエ 重みサンプル 探索チューニング 後回し) ==="
for F in 0 1; do for W in 0 1; do for E in 0 1; do for D in 0 1; do
  tag="fc${F}${W}${E}${D}"; vals=""
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_${tag}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    vals="$vals ${np:-NA}"
  done
  echo "  F=$F W=$W E=$E D=$D : n_pf=$vals"
done; done; done; done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[fct] ALL DONE $(date +%H:%M:%S)"
