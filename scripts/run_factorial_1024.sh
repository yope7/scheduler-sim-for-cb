#!/usr/bin/env bash
# trace1024 完全16セル(2^4)アブレーション。Ver.2ベース: W=距離ベース密度重み(手動帯OFF)、D=defer offset1。
# OOM対処: Phase3 cache を CPU化(DISTRIBUTED_PCN_PHASE3_GPU_CACHE=0) + MAXJOBS=4(前回1024 ladderでOOM 0件実証)。
# 上書きせず新タグ fdk_{FWED}。各 REPS seed(1024は重いため既定2)。
# 使い方: REPS=2 MAXJOBS=4 bash scripts/run_factorial_1024.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
REPS="${REPS:-2}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-4}"
CFG="experiments/distributed_pcn/job_trace_1024_pcn.yml"; SCALE=1024; PFX="fdk"
MARK=/tmp/factorial_1024.marker; rm -f "$MARK"

BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 DISTRIBUTED_PCN_PHASE3_GPU_CACHE=0"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENSITY="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
FOUR="PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"

lv_for(){ # $1..$4 = F W E D
  local F="$1" W="$2" E="$3" D="$4"; local lv="$BASE $WOFF"
  if [ "$F" = 1 ]; then lv="$lv $FOUR"; else lv="$lv PCN_FOURIER_CMD=0"; fi
  if [ "$W" = 1 ]; then lv="$lv $DENSITY"; fi
  if [ "$E" = 0 ]; then lv="$lv PCN_EVAL_GAP_FEEDBACK=0"; fi
  if [ "$D" = 1 ]; then lv="$lv $DEFER"; fi
  echo "$lv"
}

echo "[fk1024] START $(date +%H:%M:%S) 16セル×${REPS}seed SCALE=1024 MAXJOBS=$MAXJOBS (密度W+offset1, cache CPU化)"
run_cell(){ local tag="$1" lv="$2" i="$3" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${tag}_${i}" "$SCALE" "$NITER" > /tmp/fk_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[fk1024] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for F in 0 1; do for W in 0 1; do for E in 0 1; do for D in 0 1; do
  tag="${PFX}${F}${W}${E}${D}"; lv="$(lv_for $F $W $E $D)"
  for i in $(seq 1 "$REPS"); do
    while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "[fk1024] launch $tag rep=$i $(date +%H:%M:%S)"
    run_cell "$tag" "$lv" "$i" &
  done
done; done; done; done
wait
echo "[fk1024] === n_pf 速報 ==="
for F in 0 1; do for W in 0 1; do for E in 0 1; do for D in 0 1; do
  tag="${PFX}${F}${W}${E}${D}"; vals=""
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth1024_${tag}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    vals="$vals ${np:-NA}"
  done
  echo "  $tag : n_pf=$vals"
done; done; done; done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[fk1024] ALL DONE $(date +%H:%M:%S)"
