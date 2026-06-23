#!/usr/bin/env bash
# 全数アブレーション Ver.2(新規作成・旧Ver.1は温存): タグ fd{FWED}。
#   W=ON = 距離ベース密度重み(近傍k番目距離 r_k の逆比, 手動帯なし) PCN_TRAIN_PF_DENSITY_WEIGHT=8/K=2/ALPHA=1.0
#   D=ON = defer offset=1(1つ後ろ・ユーザー設計)
# 旧Ver.1(fcタグ: 手動帯W + offset4)は上書きしない(別タグ別ディレクトリ)。
# 使い方: MAXJOBS=12 bash scripts/run_factorial_v2.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-256}"; REPS="${REPS:-3}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-12}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"
MARK=/tmp/factorial_v2.marker; rm -f "$MARK"

# 新Ver.は手動帯を常にOFF(WOFF)。W軸は「密度重みのON/OFF」で分離する。
BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 PCN_COST_HOLD=0"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"
DENSITY="PCN_TRAIN_PF_DENSITY_WEIGHT=8 PCN_TRAIN_PF_DENSITY_K=2 PCN_TRAIN_PF_DENSITY_ALPHA=1.0"

lv_for(){ # $1..$4 = F W E D (0/1)
  local F="$1" W="$2" E="$3" D="$4"; local lv="$BASE $WOFF"
  if [ "$F" = 1 ]; then lv="$lv PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"; else lv="$lv PCN_FOURIER_CMD=0"; fi
  if [ "$W" = 1 ]; then lv="$lv $DENSITY"; fi                                   # W=ON: 密度重み(W=OFFは密度デフォ0=等確率)
  if [ "$E" = 0 ]; then lv="$lv PCN_EVAL_GAP_FEEDBACK=0"; fi
  if [ "$D" = 1 ]; then lv="$lv SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_PHASE1_GIANT_DEFER=0.9"; fi
  echo "$lv"
}

echo "[v2] START $(date +%H:%M:%S) 16セル×${REPS}seed SCALE=$SCALE NITER=$NITER MAXJOBS=$MAXJOBS (密度W+offset1D)"
run_cell(){ local tag="$1" lv="$2" i="$3" t0 t1; t0=$(date +%s)
  env $lv DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${tag}_${i}" "$SCALE" "$NITER" > /tmp/v2_${tag}_${i}.out 2>&1
  t1=$(date +%s); echo "[v2] $tag rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }

for F in 0 1; do for W in 0 1; do for E in 0 1; do for D in 0 1; do
  tag="fd${F}${W}${E}${D}"
  lv="$(lv_for $F $W $E $D)"
  for i in $(seq 1 "$REPS"); do
    while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "[v2] launch $tag rep=$i $(date +%H:%M:%S)"
    run_cell "$tag" "$lv" "$i" &
  done
done; done; done; done
wait
echo "[v2] === Ver.2 n_pf 速報 ==="
for F in 0 1; do for W in 0 1; do for E in 0 1; do for D in 0 1; do
  tag="fd${F}${W}${E}${D}"; vals=""
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_${tag}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    vals="$vals ${np:-NA}"
  done
  echo "  $tag : n_pf=$vals"
done; done; done; done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[v2] ALL DONE $(date +%H:%M:%S)"
