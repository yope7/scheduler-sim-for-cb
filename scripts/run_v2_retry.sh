#!/usr/bin/env bash
# Ver.2 で exit=1 失敗した3 run を再学習(offset1・密度W設定)。失敗ディレクトリは run_synthetic_urgency が rm -rf。
set -u
cd /home/noguchi/scheduler-sim-for-cb
CFG=experiments/distributed_pcn/job_trace_256_pcn.yml
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENSITY="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
rm -f /tmp/v2retry.marker
retry(){ local name="$1" lv="$2"
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG bash scripts/run_synthetic_urgency.sh "$name" 256 100 > /tmp/v2retry_${name}.out 2>&1
  echo "[retry] $name DONE exit=$? $(date +%H:%M:%S)"; }
echo "[retry] START $(date +%H:%M:%S)"
retry fd1001_2 "$BASE $WOFF PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_EVAL_GAP_FEEDBACK=0 $DEFER" &       # F1 W0 E0 D1
retry fd1101_1 "$BASE $WOFF $DENSITY PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_EVAL_GAP_FEEDBACK=0 $DEFER" &  # F1 W1 E0 D1
retry fd0111_3 "$BASE $WOFF $DENSITY PCN_FOURIER_CMD=0 $DEFER" &                                          # F0 W1 E1 D1
wait
echo "DONE $(date +%H:%M:%S)" > /tmp/v2retry.marker
echo "[retry] ALL DONE $(date +%H:%M:%S)"
