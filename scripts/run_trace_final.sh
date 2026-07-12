#!/usr/bin/env bash
# 最終フェーズ: ct03(cmd-track w0.3)を trace256で5seed化 + trace512でスケール検証(base vs ct03)。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${NITER:-60}"; MAXJOBS="${MAXJOBS:-4}"
MARK=/tmp/trfinal.marker; rm -f "$MARK"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"
CT03="PCN_CMD_TRACK_WEIGHT=0.3 DISTRIBUTED_PCN_CMD_OUTCOMES=1"

# (name, scale, lv, seed)  name は run dir/eval tag に使う
JOBS=()
for i in 4 5; do JOBS+=("ct03|256|$BASE $CT03|$i"); done       # 256 ct03 を5seed化
for i in 1 2 3; do JOBS+=("base512|512|$BASE|$i"); done         # 512 base
for i in 1 2 3; do JOBS+=("ct03512|512|$BASE $CT03|$i"); done   # 512 ct03

echo "[trfinal] START $(date +%H:%M:%S) ${#JOBS[@]}runs NITER=$NITER MAXJOBS=$MAXJOBS"
train_one(){ local name="$1" scale="$2" lv="$3" i="$4" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=experiments/distributed_pcn/job_trace_${scale}_pcn.yml \
    DISTRIBUTED_PCN_N_ACTORS=8 \
    bash scripts/run_synthetic_urgency.sh "scr_${name}_${i}" "$scale" "$NITER" > /tmp/trfinal_${name}_${i}.out 2>&1
  t1=$(date +%s); echo "[trfinal] $name rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name scale lv i <<< "$spec"
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[trfinal] launch $name($scale) rep=$i $(date +%H:%M:%S)"
  train_one "$name" "$scale" "$lv" "$i" &
done
wait
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[trfinal] ALL DONE $(date +%H:%M:%S)"
