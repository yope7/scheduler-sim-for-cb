#!/usr/bin/env bash
# Phase2 A/B: 劣化を学習で防ぐ3レバーを early-stop OFF・最終ckpt(iteration_NITER)で検証。
# 主指標: 最終ckptの HV平均/効率run/std が baseline(a0) を上回るか(=劣化を学習で防げたか)。
# usage: EXP=a1_frozen bash scripts/run_512_levers.sh
#   EXP: a0_base / a1_frozen / a2_ema99 / a3_ema999 / a4_lrcos / a5_combo
#   SCALE(512), REPS(5), NITER(100), MAXJOBS(3), NPROC(32) は env 上書き可。a5_combo は A5_LV で中身指定可。
set -u
cd /home/noguchi/scheduler-sim-for-cb
EXP="${EXP:?set EXP}"; SCALE="${SCALE:-512}"; REPS="${REPS:-5}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-32}"
CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml
case "$EXP" in
  a0_base)   LV="" ;;
  a1_frozen) LV="PCN_FROZEN_PF_CLONE=1 PCN_FROZEN_PF_MAX=128" ;;
  a2_ema99)  LV="PCN_EMA_DECAY=0.99" ;;
  a3_ema999) LV="PCN_EMA_DECAY=0.999" ;;
  a4_lrcos)  LV="PCN_LR_DECAY=cosine PCN_LR_DECAY_FINAL=0.1" ;;
  a5_combo)  LV="${A5_LV:-PCN_EMA_DECAY=0.999 PCN_LR_DECAY=cosine PCN_LR_DECAY_FINAL=0.1}" ;;   # EMA(強)+LR減衰
  a6_lrlow)  LV="PCN_LR_DECAY=cosine PCN_LR_DECAY_FINAL=0.03" ;;                                  # LR減衰を強める
  a7_lrema)  LV="PCN_EMA_DECAY=0.999 PCN_LR_DECAY=cosine PCN_LR_DECAY_FINAL=0.03" ;;             # EMA(強)+LR(強)
  *) echo "unknown EXP=$EXP"; exit 1 ;;
esac
ITERZ=$(printf %03d "$NITER")
TIMES=/tmp/lev_${EXP}_times.txt; MARK=/tmp/lev_${EXP}.marker; rm -f "$MARK" "$TIMES"
echo "[lev:$EXP] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER LEVERS=[$LV]"
train_one(){ local i="$1"; local t0 t1; t0=$(date +%s)
  env $LV DISTRIBUTED_PCN_EARLYSTOP=0 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "${EXP}${i}" "$SCALE" "$NITER" > /tmp/lev_${EXP}_train_${i}.out 2>&1
  local ex=$?; t1=$(date +%s); echo "train rep=$i exit=$ex sec=$((t1-t0))" >> "$TIMES"
  echo "[lev:$EXP] train rep=$i DONE exit=$ex sec=$((t1-t0)) $(date +%H:%M:%S)"; }
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[lev:$EXP] launch rep=$i $(date +%H:%M:%S)"; train_one "$i" &
done
wait
echo "[lev:$EXP] ALL TRAIN DONE $(date +%H:%M:%S)"
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_${EXP}${i}
  EXEC=$(find "$OUT"/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  ck="$EXEC/iteration_${ITERZ}/model_iter_${ITERZ}.pth"        # early-stop OFF → 最終ckpt を評価
  [ -f "$ck" ] || ck=$(find "$EXEC"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ] || [ ! -f "$ck" ]; then echo "[lev:$EXP] MISSING ckpt rep=$i"; continue; fi
  t0=$(date +%s)
  CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_${EXP}${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/lev_${EXP}_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s); echo "eval rep=$i exit=$ex sec=$((t1-t0))" >> "$TIMES"
  [ "$ex" = 0 ] && echo "[lev:$EXP] eval rep=$i saved" || echo "[lev:$EXP] eval rep=$i FAIL"
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "[lev:$EXP] ALL DONE $(date +%H:%M:%S)"; echo "===== TIMES ====="; cat "$TIMES"
