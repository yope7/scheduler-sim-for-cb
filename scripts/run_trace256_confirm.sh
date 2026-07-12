#!/usr/bin/env bash
# trace256 確認スクリーン: 崩壊率と ct の安定性を seed 追加で確認 + HV回復(ct弱め)を試す。
#   base seeds 3-5 / ct seeds 3-5 (= 各5seed化) / ct03(cmd-track weight 0.3) seeds 1-3
# 使い方: MAXJOBS=4 bash scripts/run_trace256_confirm.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${NITER:-60}"; MAXJOBS="${MAXJOBS:-4}"; SCALE=256
CFG="experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml"
MARK=/tmp/tr256cfm.marker; rm -f "$MARK"

WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"
CT="PCN_CMD_TRACK_WEIGHT=1.0 DISTRIBUTED_PCN_CMD_OUTCOMES=1"
CT03="PCN_CMD_TRACK_WEIGHT=0.3 DISTRIBUTED_PCN_CMD_OUTCOMES=1"

# (tag, LV, seed) のジョブ列
JOBS=()
for i in 3 4 5; do JOBS+=("base|$BASE|$i"); done
for i in 3 4 5; do JOBS+=("ct|$BASE $CT|$i"); done
for i in 1 2 3; do JOBS+=("ct03|$BASE $CT03|$i"); done

echo "[tr256cfm] START $(date +%H:%M:%S) ${#JOBS[@]}runs NITER=$NITER MAXJOBS=$MAXJOBS"
train_one(){ local tag="$1" lv="$2" i="$3" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG DISTRIBUTED_PCN_N_ACTORS=8 \
    bash scripts/run_synthetic_urgency.sh "scr_${tag}_${i}" "$SCALE" "$NITER" > /tmp/tr256cfm_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[tr256cfm] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag lv i <<< "$spec"
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[tr256cfm] launch $tag rep=$i $(date +%H:%M:%S)"
  train_one "$tag" "$lv" "$i" &
done
wait
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[tr256cfm] ALL DONE $(date +%H:%M:%S)"
