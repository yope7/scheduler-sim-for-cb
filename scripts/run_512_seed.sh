#!/usr/bin/env bash
# /goal 実験B: 探索/種を「効率の膝」へ誘導して効率の再現性を上げる。
# 発見: 既定の Phase1 閾値(0,50,150,500,999999)は効率の膝(cost0.1-0.5)を全く張らない。
#   膝は閾値 180k-700k(=ごく一部の本当に詰まるジョブだけburst)に在る。これを種に足す。
# 単一レバー: DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS をデータ由来の膝カバー集合に置換。他は baseline 同一。
# 出力: results/eval_pf/truepf_trace512_seed{i}_s0.npz, /tmp/seed512_times.txt, /tmp/seed512.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE=512; REPS="${REPS:-5}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-32}"
CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml
# 達成フロント全域(cost 0.07→1.0)を均等に張る閾値(eval_threshold_seed.py で実測した膝位置から)
THR="0,20000,80000,180000,260000,360000,500000,700000,999999"
TIMES=/tmp/seed512_times.txt; MARK=/tmp/seed512.marker
rm -f "$MARK" "$TIMES"
echo "[seed] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS THRESHOLDS=$THR"
train_one () {
  local i="$1"; local t0 t1; t0=$(date +%s)
  DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS="$THR" DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "seed${i}" "$SCALE" "$NITER" > /tmp/seed512_train_${i}.out 2>&1
  local ex=$?; t1=$(date +%s)
  echo "train rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  echo "[seed] train rep=${i} DONE exit=${ex} sec=$((t1-t0)) $(date +%H:%M:%S)"
}
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[seed] launch train rep=${i} $(date +%H:%M:%S)"; train_one "$i" &
done
wait
echo "[seed] ALL TRAIN DONE $(date +%H:%M:%S)"
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_seed${i}
  ck=$(find "$OUT"/20*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ]; then echo "[seed] MISSING ckpt rep=${i}"; continue; fi
  t0=$(date +%s)
  CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_seed${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/seed512_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s); echo "eval rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  [ "$ex" = 0 ] && echo "[seed] eval rep=${i} saved" || echo "[seed] eval rep=${i} FAIL"
done
echo "[seed] ALL EVAL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "===== TIMES ====="; cat "$TIMES"
