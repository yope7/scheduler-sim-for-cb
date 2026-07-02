#!/usr/bin/env bash
# trace1024 で cmd-track(w0.3) を確認: base vs ct03 を各3seed。正準レシピは run_trace_final.sh と同一、
# scale=1024(job_trace_1024_pcn.yml)。serial 実行(較正ランと同一条件=確実に動く / 2並列は CUDA_VISIBLE_DEVICES
# をコードが device 文字列に誤読する+Ray GPU 配置 OOM の二重リスクで断念)。6本 × ~26分 ≈ 2.6h。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${NITER:-60}"
MARK=/tmp/ctk1024.marker; rm -f "$MARK"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENS="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 $WOFF $DENS $DEFER"
CT03="PCN_CMD_TRACK_WEIGHT=0.3 DISTRIBUTED_PCN_CMD_OUTCOMES=1"

train_one(){ local name="$1" lv="$2" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 \
    DISTRIBUTED_PCN_CONFIG=experiments/distributed_pcn/job_trace_1024_pcn.yml \
    DISTRIBUTED_PCN_N_ACTORS=8 \
    bash scripts/run_synthetic_urgency.sh "$name" 1024 "$NITER" > /tmp/ctk1024_${name}.out 2>&1
  t1=$(date +%s); echo "[ctk1024] $name DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

SEEDS="${SEEDS:-1 2 3}"
echo "[ctk1024] START $(date +%H:%M:%S) NITER=$NITER SEEDS='$SEEDS' (base + ct03, serial, name suffix _n$NITER)"
for i in $SEEDS; do
  echo "[ctk1024] seed=$i base $(date +%H:%M:%S)";  train_one "ctk1024_base_${i}_n${NITER}" "$BASE"
  echo "[ctk1024] seed=$i ct03 $(date +%H:%M:%S)";  train_one "ctk1024_ct03_${i}_n${NITER}" "$BASE $CT03"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[ctk1024] ALL DONE $(date +%H:%M:%S)"
