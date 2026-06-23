#!/usr/bin/env bash
# defer offset=1(ユーザー設計「deferは1つ後ろのみ」)で D=ON の8セルを再実験し、
# 旧 offset=4 の結果を置換する。F/W/E は全組合せ、D=1固定。各 REPS seed。
# 使い方: MAXJOBS=8 bash scripts/run_defer_offset1_redo.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-256}"; REPS="${REPS:-3}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-8}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"
MARK=/tmp/defer_offset1_redo.marker; rm -f "$MARK"

BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"

lv_for(){ # $1 F  $2 W  $3 E  (D=1固定)
  local F="$1" W="$2" E="$3"; local lv="$BASE $DEFER"
  if [ "$F" = 1 ]; then lv="$lv PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"; else lv="$lv PCN_FOURIER_CMD=0"; fi
  if [ "$W" = 0 ]; then lv="$lv $WOFF"; fi
  if [ "$E" = 0 ]; then lv="$lv PCN_EVAL_GAP_FEEDBACK=0"; fi
  echo "$lv"
}
run_cell(){ local tag="$1" lv="$2" i="$3" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${tag}_${i}" "$SCALE" "$NITER" > /tmp/dro_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[dro] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

echo "[dro] START $(date +%H:%M:%S) D=ON 8セル×${REPS}seed offset=1 SCALE=$SCALE NITER=$NITER MAXJOBS=$MAXJOBS"
for F in 0 1; do for W in 0 1; do for E in 0 1; do
  D=1; tag="fc${F}${W}${E}${D}"; lv="$(lv_for $F $W $E)"
  for i in $(seq 1 "$REPS"); do
    while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "[dro] launch $tag rep=$i $(date +%H:%M:%S)"
    run_cell "$tag" "$lv" "$i" &
  done
done; done; done
wait
echo "[dro] === offset=1 再実験 完了。n_pf 速報 ==="
for F in 0 1; do for W in 0 1; do for E in 0 1; do
  tag="fc${F}${W}${E}1"; vals=""
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_${tag}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    vals="$vals ${np:-NA}"
  done
  echo "  $tag : n_pf=$vals"
done; done; done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[dro] ALL DONE $(date +%H:%M:%S)"
