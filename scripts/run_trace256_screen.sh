#!/usr/bin/env bash
# trace256 小規模スクリーニング: 命令追従/PF品質を上げる機構的レバーの方向当て。
# 共通=Ver.2の良レシピ(FILM+occupancy+N_UPDATES=200+Fourier(bands4)+密度W+offset1 defer, 手動帯OFF)。
# 各config=共通 + 1レバー。trace256・REPS seed・NITER iter・final_model評価(rich_eval)。
# 使い方: REPS=2 NITER=60 MAXJOBS=4 bash scripts/run_trace256_screen.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
REPS="${REPS:-2}"; NITER="${NITER:-60}"; MAXJOBS="${MAXJOBS:-4}"; SCALE=256
CFG="experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml"
MARK=/tmp/tr256scr.marker; rm -f "$MARK"

WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"

declare -A LV
LV[base]="$BASE"
LV[fb2]="$BASE PCN_FOURIER_BANDS=2"
LV[wd]="$BASE PCN_WEIGHT_DECAY=1e-4"
LV[ct]="$BASE PCN_CMD_TRACK_WEIGHT=1.0 DISTRIBUTED_PCN_CMD_OUTCOMES=1"

echo "[tr256scr] START $(date +%H:%M:%S) trace256 configs=${!LV[@]} REPS=$REPS NITER=$NITER MAXJOBS=$MAXJOBS"
train_one(){ local tag="$1" lv="$2" i="$3" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG DISTRIBUTED_PCN_N_ACTORS=8 \
    bash scripts/run_synthetic_urgency.sh "scr_${tag}_${i}" "$SCALE" "$NITER" > /tmp/tr256scr_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[tr256scr] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for tag in base fb2 wd ct; do
  for i in $(seq 1 "$REPS"); do
    while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "[tr256scr] launch $tag rep=$i $(date +%H:%M:%S)"
    train_one "$tag" "${LV[$tag]}" "$i" &
  done
done
wait
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[tr256scr] ALL TRAIN DONE $(date +%H:%M:%S)"
