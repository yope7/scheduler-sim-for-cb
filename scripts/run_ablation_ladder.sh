#!/usr/bin/env bash
# 段階的削減アブレーション(節約版): 過去最良 F+W+D(探索Eは元OFF)から効果の小さい順に1つずつOFF。
#   段階0 FWD = フーリエ+密度重み+offset1 defer (=fd1101相当・最良)
#   段階1 FD  = 密度W OFF
#   段階2 F   = offset1 defer も OFF (フーリエのみ)
#   段階3 base= 全OFF
# DATASET=synth(合成256, job_synthetic_pcn.yml) | trace1024(trace1024, job_trace_1024_pcn.yml)
# 上書きせず新タグ lad_s_* / lad_t_*。各 REPS seed。
# 使い方: DATASET=synth MAXJOBS=12 bash scripts/run_ablation_ladder.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
DATASET="${DATASET:-synth}"
REPS="${REPS:-3}"; NITER="${NITER:-100}"
if [ "$DATASET" = synth ]; then
  CFG="experiments/distributed_pcn/job_synthetic_pcn.yml"; SCALE=256; PFX="lad_s"; MAXJOBS="${MAXJOBS:-12}"; BASE_EXTRA=""
elif [ "$DATASET" = trace1024 ]; then
  CFG="experiments/distributed_pcn/job_trace_1024_pcn.yml"; SCALE=1024; PFX="lad_t"; MAXJOBS="${MAXJOBS:-4}"
  BASE_EXTRA="DISTRIBUTED_PCN_PHASE3_GPU_CACHE=0"   # 1024はPhase3 cacheをCPU化してGPU OOM回避
else
  echo "DATASET must be synth|trace1024"; exit 1
fi
MARK=/tmp/ablation_ladder_${DATASET}.marker; rm -f "$MARK"

BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0 PCN_EVAL_GAP_FEEDBACK=0 ${BASE_EXTRA:-}"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENSITY="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"
DEFER="SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"
FOUR="PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"

STAGES="FWD FD F base"
lv_for(){
  case "$1" in
    FWD)  echo "$BASE $WOFF $FOUR $DENSITY $DEFER";;
    FD)   echo "$BASE $WOFF $FOUR $DEFER";;
    F)    echo "$BASE $WOFF $FOUR";;
    base) echo "$BASE $WOFF PCN_FOURIER_CMD=0";;
  esac
}

echo "[ladder] START $(date +%H:%M:%S) DATASET=$DATASET SCALE=$SCALE CFG=$CFG MAXJOBS=$MAXJOBS stages='$STAGES'"
run_cell(){ local tag="$1" lv="$2" i="$3" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${tag}_${i}" "$SCALE" "$NITER" > /tmp/lad_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[ladder] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for st in $STAGES; do
  tag="${PFX}_${st}"; lv="$(lv_for $st)"
  for i in $(seq 1 "$REPS"); do
    while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "[ladder] launch $tag rep=$i $(date +%H:%M:%S)"
    run_cell "$tag" "$lv" "$i" &
  done
done
wait
echo "[ladder] === $DATASET n_pf 速報 ==="
for st in $STAGES; do
  tag="${PFX}_${st}"; vals=""
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_${tag}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    vals="$vals ${np:-NA}"
  done
  echo "  $tag : n_pf=$vals"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[ladder] ALL DONE $(date +%H:%M:%S)"
