#!/usr/bin/env bash
# Phase A: 合成の裾ダイヤル(SYNTH_TAIL_LEVEL)を振り、崩壊の発生点(崖)を特定する。
# early-stop OFF・最終ckpt 評価が主指標（ckpt選択でなく最適化自体の安定性を測る）。
# 使い方: TAG=tailL08 LV="SYNTH_TAIL_LEVEL=0.8 <recipe levers>" bash scripts/run_synth_tail.sh
# 出力: results/eval_pf/truepf_trace${SCALE}_${TAG}${SCALE}_${i}_s0.npz, /tmp/${TAG}${SCALE}.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-512}"; TAG="${TAG:-tail}"; REPS="${REPS:-5}"; NITER="${NITER:-100}"
MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-32}"; ESTOP="${ESTOP:-0}"
LV="${LV:-}"
CFG="${CFG:-experiments/distributed_pcn/job_synthetic_pcn.yml}"
GRP="${TAG}${SCALE}"
TIMES=/tmp/${GRP}_times.txt; MARK=/tmp/${GRP}.marker; rm -f "$MARK" "$TIMES"
echo "[$GRP] START $(date +%H:%M:%S) SCALE=$SCALE EARLYSTOP=$ESTOP LEVERS=[$LV] CFG=$CFG"
train_one(){ local i="$1"; local t0 t1; t0=$(date +%s)
  env $LV DISTRIBUTED_PCN_EARLYSTOP=$ESTOP DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${GRP}_${i}" "$SCALE" "$NITER" > /tmp/${GRP}_train_${i}.out 2>&1
  local ex=$?; t1=$(date +%s); echo "train rep=$i exit=$ex sec=$((t1-t0))" >> "$TIMES"
  echo "[$GRP] train rep=$i DONE exit=$ex sec=$((t1-t0)) $(date +%H:%M:%S)"; }
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[$GRP] launch rep=$i $(date +%H:%M:%S)"; train_one "$i" &
done
wait
echo "[$GRP] ALL TRAIN DONE $(date +%H:%M:%S)"
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_${GRP}_${i}
  EXEC=$(find "$OUT"/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  # 最終ckpt を採用（early-stop OFF＝最適化自体の着地を測る）
  ck="$EXEC/final_model.pth"
  [ -f "$ck" ] || ck=$(find "$EXEC"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ] || [ ! -f "$ck" ]; then echo "[$GRP] MISSING ckpt rep=$i"; continue; fi
  echo "[$GRP] rep=$i use=$(basename "$ck")"
  t0=$(date +%s)
  env $LV CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_${GRP}_${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/${GRP}_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s); echo "eval rep=$i exit=$ex sec=$((t1-t0))" >> "$TIMES"
  [ "$ex" = 0 ] && echo "[$GRP] eval rep=$i saved" || echo "[$GRP] eval rep=$i FAIL"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "[$GRP] ALL DONE $(date +%H:%M:%S)"; echo "===== TIMES ====="; cat "$TIMES"
