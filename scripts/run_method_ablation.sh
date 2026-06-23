#!/usr/bin/env bash
# 手法アブレーション(leave-one-out): 既存3手法(フーリエ/重みサンプリング/探索場所チューニング)+新(ケチる癖直し=cost_hold)
# を全部ONの full から1つずつ外し、本物データ(trace)でどれが効くか n_pf で測る。
# 対応: フーリエ=PCN_FOURIER_CMD / 重みサンプリング=中域PF重み群 / 探索チューニング=PCN_EVAL_GAP_FEEDBACK / ケチる癖直し=PCN_COST_HOLD
# 全(条件×seed)を1プールで並列(MAXJOBS)。使い方: SCALE=256 REPS=3 NITER=100 MAXJOBS=8 bash scripts/run_method_ablation.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-256}"; REPS="${REPS:-3}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-8}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"
MARK=/tmp/method_ablation.marker; rm -f "$MARK"

BASE="PCN_FILM=1 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_URGENCY=0 DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1"
FOUR="PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"
WOFF="PCN_TRAIN_MID_PF_WEIGHT=0 PCN_TRAIN_KNEE_PF_WEIGHT=0 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 PCN_TRAIN_COST_ENDPOINT_WEIGHT=0"

# 条件名 -> LV
declare -A LV
LV[full]="$BASE $FOUR PCN_COST_HOLD=1"
LV[no_fourier]="$BASE PCN_COST_HOLD=1"
LV[no_weight]="$BASE $FOUR PCN_COST_HOLD=1 $WOFF"
LV[no_exptune]="$BASE $FOUR PCN_COST_HOLD=1 PCN_EVAL_GAP_FEEDBACK=0"
LV[no_costhold]="$BASE $FOUR PCN_COST_HOLD=0"
CONDS="full no_fourier no_weight no_exptune no_costhold"

echo "[abl] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER MAXJOBS=$MAXJOBS"
train_one(){ local cond="$1" i="$2"; local t0 t1; t0=$(date +%s)
  env ${LV[$cond]} DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "abl_${cond}_${i}" "$SCALE" "$NITER" > /tmp/abl_${cond}_${i}.out 2>&1
  t1=$(date +%s); echo "[abl] $cond rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }
for cond in $CONDS; do for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  train_one "$cond" "$i" &
done; done
wait
echo "[abl] === n_pf 結果(本物データ trace${SCALE}) ==="
for cond in $CONDS; do
  vals=""
  for i in $(seq 1 "$REPS"); do
    sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_abl_${cond}_${i}/2026* 2>/dev/null | tail -1)
    np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
    vals="$vals ${np:-NA}"
  done
  echo "  $cond: n_pf=$vals"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[abl] ALL DONE $(date +%H:%M:%S)"
