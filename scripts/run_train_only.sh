#!/usr/bin/env bash
# 軽量trainer: 学習のみ(eval_b2_compareを省く=無駄削減)。n_pfは学習中 uniform_cmd_stats から取得。
# MAXJOBS を上げて64コアを活用。使い方:
#   TAG=syn SCALE=512 REPS=5 NITER=100 MAXJOBS=5 CFG=.. LV=".." bash scripts/run_train_only.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-256}"; TAG="${TAG:-t}"; REPS="${REPS:-5}"; NITER="${NITER:-100}"
MAXJOBS="${MAXJOBS:-5}"; ESTOP="${ESTOP:-0}"; LV="${LV:-}"
CFG="${CFG:-experiments/distributed_pcn/job_synthetic_pcn.yml}"
GRP="${TAG}${SCALE}"
MARK=/tmp/${GRP}_trainonly.marker; rm -f "$MARK"
echo "[$GRP] TRAIN-ONLY START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER MAXJOBS=$MAXJOBS CFG=$CFG LV=[$LV]"
train_one(){ local i="$1"; local t0 t1; t0=$(date +%s)
  env $LV DISTRIBUTED_PCN_EARLYSTOP=$ESTOP DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${GRP}_${i}" "$SCALE" "$NITER" > /tmp/${GRP}_train_${i}.out 2>&1
  local ex=$?; t1=$(date +%s); echo "[$GRP] rep=$i DONE exit=$ex sec=$((t1-t0)) $(date +%H:%M:%S)"; }
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[$GRP] launch rep=$i $(date +%H:%M:%S)"; train_one "$i" &
done
wait
echo "[$GRP] === n_pf (uniform_cmd_pf, 忠実指標) ==="
for i in $(seq 1 "$REPS"); do
  sub=$(ls -d experiments/distributed_pcn/run_synth${SCALE}_${GRP}_${i}/2026* 2>/dev/null | tail -1)
  np=$(ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*')
  echo "  rep=$i n_pf=${np:-NA}"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[$GRP] TRAIN-ONLY DONE $(date +%H:%M:%S)"
