#!/usr/bin/env bash
# 高速スクリーニング: 仮説の方向を 128ジョブ・3seed・50iter・8actors で素早く当てる（1仮説≈15-20分）。
# 勝った仮説だけ 512×5seed×100iter で確認する2段構え。512結果との差はスケール依存に注意
# （Fourier逆U字など）—方向の当たりだけを見る。
# 使い方: TAG=burst08 LV="SYNTH_TAIL_LEVEL=0.8 SYNTH_TAIL_BURST=4 <recipe>" bash scripts/run_screen.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-128}"; TAG="${TAG:-scr}"; REPS="${REPS:-3}"; NITER="${NITER:-50}"
MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-12}"; ESTOP="${ESTOP:-0}"; ACTORS="${ACTORS:-8}"
LV="${LV:-}"
CFG="${CFG:-experiments/distributed_pcn/job_synthetic_pcn.yml}"
GRP="${TAG}${SCALE}"
MARK=/tmp/${GRP}.marker; rm -f "$MARK"
echo "[$GRP] SCREEN START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER ACTORS=$ACTORS LV=[$LV]"
train_one(){ local i="$1"; local t0 t1; t0=$(date +%s)
  env $LV DISTRIBUTED_PCN_EARLYSTOP=$ESTOP DISTRIBUTED_PCN_CONFIG=$CFG DISTRIBUTED_PCN_N_ACTORS=$ACTORS \
    bash scripts/run_synthetic_urgency.sh "${GRP}_${i}" "$SCALE" "$NITER" > /tmp/${GRP}_train_${i}.out 2>&1
  t1=$(date +%s); echo "[$GRP] train rep=$i DONE exit=$? sec=$((t1-t0)) $(date +%H:%M:%S)"; }
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  train_one "$i" &
done
wait
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_${GRP}_${i}
  EXEC=$(find "$OUT"/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  ck="$EXEC/final_model.pth"
  [ -f "$ck" ] || ck=$(find "$EXEC"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  [ -f "$ck" ] || { echo "[$GRP] MISSING ckpt rep=$i"; continue; }
  env $LV CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=24 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_${GRP}_${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/${GRP}_eval_${i}.out 2>&1 && echo "[$GRP] eval rep=$i saved" || echo "[$GRP] eval rep=$i FAIL"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "[$GRP] SCREEN DONE $(date +%H:%M:%S)"
