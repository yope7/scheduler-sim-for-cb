#!/usr/bin/env bash
# /goal 実用版early-stop: 学習中に達成front HV(固定参照点)を測り最良ckptを best_model.pth に保存。
# 続学習が効率方策を壊す(検証済: HVは中盤ピーク→劣化)ため、最終iterでなく達成HV最良を採る。
# baseline と同一レシピ + DISTRIBUTED_PCN_EARLYSTOP=1 のみ追加。eval は best_model.pth を使う。
# 出力: results/eval_pf/truepf_trace512_es{i}_s0.npz, /tmp/es512_times.txt, /tmp/es512.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE=512; REPS="${REPS:-5}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-32}"
CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml
TIMES=/tmp/es512_times.txt; MARK=/tmp/es512.marker
rm -f "$MARK" "$TIMES"
echo "[es] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS EARLYSTOP=1"
train_one () {
  local i="$1"; local t0 t1; t0=$(date +%s)
  DISTRIBUTED_PCN_EARLYSTOP=1 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "es${i}" "$SCALE" "$NITER" > /tmp/es512_train_${i}.out 2>&1
  local ex=$?; t1=$(date +%s)
  echo "train rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  echo "[es] train rep=${i} DONE exit=${ex} sec=$((t1-t0)) $(date +%H:%M:%S)"
}
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[es] launch train rep=${i} $(date +%H:%M:%S)"; train_one "$i" &
done
wait
echo "[es] ALL TRAIN DONE $(date +%H:%M:%S)"
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_es${i}
  EXEC=$(find "$OUT"/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  ck="$EXEC/best_model.pth"
  if [ ! -f "$ck" ]; then
    echo "[es] WARN best_model欠落 rep=${i}, final ckptにフォールバック"
    ck=$(find "$EXEC"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  fi
  if [ -z "$ck" ] || [ ! -f "$ck" ]; then echo "[es] MISSING ckpt rep=${i}"; continue; fi
  # どのiterがbestか記録
  bestit=$(grep -h "best_model更新" "$OUT/train.log" 2>/dev/null | tail -1)
  echo "[es] rep=${i} use=$ck  ($bestit)"
  t0=$(date +%s)
  CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_es${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/es512_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s); echo "eval rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  [ "$ex" = 0 ] && echo "[es] eval rep=${i} saved" || echo "[es] eval rep=${i} FAIL"
done
echo "[es] ALL EVAL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "===== TIMES ====="; cat "$TIMES"
