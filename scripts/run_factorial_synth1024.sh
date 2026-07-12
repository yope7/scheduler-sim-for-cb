#!/usr/bin/env bash
# 合成1024(job_type=1) アブレーション。探索E除外=2^3=8セル(F/W/D, E=OFF固定)。
# レシピは run_factorial_1024.sh(trace1024)と完全に揃え、データセットだけ合成(job_synthetic_pcn.yml)に変更。
#   → 合成 vs trace を同一レシピで比較できる。
# OOM対処: Phase3 cache CPU化 + MAXJOBS=4(trace1024で実証, 1024同ジョブ数=同メモリ profile)。
# タグ fsk_{F}{W}{E}{D}(E=0固定)。各 REPS seed(既定3=trace1024と一致)。
# 使い方: REPS=3 MAXJOBS=4 bash scripts/run_factorial_synth1024.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
REPS="${REPS:-3}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-4}"
CFG="experiments/distributed_pcn/job_synthetic_pcn.yml"; SCALE=1024; PFX="fsk"
MARK=/tmp/factorial_synth1024.marker; rm -f "$MARK"

# trace1024 と同一 BASE(URGENCY=0/OCCUPANCY=1/N_UPDATES=200/cache CPU化)
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 DISTRIBUTED_PCN_PHASE3_GPU_CACHE=0"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENSITY="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
FOUR="PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"

lv_for(){ # $1..$3 = F W D  (E は常に OFF)
  local F="$1" W="$2" D="$3"; local lv="$BASE $WOFF PCN_EVAL_GAP_FEEDBACK=0"
  if [ "$F" = 1 ]; then lv="$lv $FOUR"; else lv="$lv PCN_FOURIER_CMD=0"; fi
  if [ "$W" = 1 ]; then lv="$lv $DENSITY"; fi
  if [ "$D" = 1 ]; then lv="$lv $DEFER"; fi
  echo "$lv"
}

echo "[fsk] START $(date +%H:%M:%S) 8セル(E除外)×${REPS}seed SCALE=1024 合成 MAXJOBS=$MAXJOBS"
run_cell(){ local tag="$1" lv="$2" i="$3" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${tag}_${i}" "$SCALE" "$NITER" > /tmp/fsk_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[fsk] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for F in 0 1; do for W in 0 1; do for D in 0 1; do
  tag="${PFX}${F}${W}0${D}"   # E=0 を tag に明示(fdkと同じ4桁FWED形式)
  lv="$(lv_for $F $W $D)"
  for i in $(seq 1 "$REPS"); do
    while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "[fsk] launch $tag rep=$i $(date +%H:%M:%S)"
    run_cell "$tag" "$lv" "$i" &
  done
done; done; done
wait
echo "[fsk] === n_pf 速報 ==="
for F in 0 1; do for W in 0 1; do for D in 0 1; do
  tag="${PFX}${F}${W}0${D}"; vals=""
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth1024_${tag}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    vals="$vals ${np:-NA}"
  done
  echo "  $tag : n_pf=$vals"
done; done; done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[fsk] ALL DONE $(date +%H:%M:%S)"
