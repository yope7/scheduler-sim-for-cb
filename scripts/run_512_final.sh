#!/usr/bin/env bash
# 本命解: LR減衰(維持で底上げ・ピーク深化) + early-stop(早期ピーク捕捉)。best_model.pth を eval。
# SCALE(512/256)・TAG・CFG・LV を env で上書き可。汎用性確認は SCALE=256 で同一 LV を使う。
# 出力: results/eval_pf/truepf_trace${SCALE}_${TAG}${SCALE}_${i}_s0.npz, /tmp/${TAG}${SCALE}.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-512}"; TAG="${TAG:-final}"; REPS="${REPS:-5}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-32}"
LV="${LV:-PCN_LR_DECAY=cosine PCN_LR_DECAY_FINAL=0.1}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"
GRP="${TAG}${SCALE}"
TIMES=/tmp/${GRP}_times.txt; MARK=/tmp/${GRP}.marker; rm -f "$MARK" "$TIMES"
echo "[$GRP] START $(date +%H:%M:%S) SCALE=$SCALE LEVERS=[$LV] EARLYSTOP=1 CFG=$CFG"
train_one(){ local i="$1"; local t0 t1; t0=$(date +%s)
  env $LV DISTRIBUTED_PCN_EARLYSTOP=1 DISTRIBUTED_PCN_CONFIG=$CFG \
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
  ck="$EXEC/best_model.pth"   # early-stop ON → 学習中に自動選択した best_model
  [ -f "$ck" ] || ck=$(find "$EXEC"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ] || [ ! -f "$ck" ]; then echo "[$GRP] MISSING ckpt rep=$i"; continue; fi
  bestit=$(grep -h "best_model更新" "$OUT/train.log" 2>/dev/null | tail -1 | grep -oE "iter[0-9]+" | head -1)
  echo "[$GRP] rep=$i use=best_model ($bestit)"
  t0=$(date +%s)
  env $LV CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_${GRP}_${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/${GRP}_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s); echo "eval rep=$i exit=$ex sec=$((t1-t0))" >> "$TIMES"
  [ "$ex" = 0 ] && echo "[$GRP] eval rep=$i saved" || echo "[$GRP] eval rep=$i FAIL"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "[$GRP] ALL DONE $(date +%H:%M:%S)"; echo "===== TIMES ====="; cat "$TIMES"
