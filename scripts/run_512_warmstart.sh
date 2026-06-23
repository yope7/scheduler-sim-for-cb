#!/usr/bin/env bash
# /goal 実験A′(warm-start蒸留): 唯一の効率run(rep4 HV96%)の重みを Phase3 開始直前に全runへ注入し、
#   5本を異なる学習乱数で続学習する。「効率basinは安定な引力圏か(発見さえすれば保てるか)」を直接判定する。
# 仮説: rep4の重みから始めれば、初期iterでrep4が膝デモを生成→archive/frozen-PF に膝が入り、
#   以降の全学習が膝デモ付きで進む(=①②で不在だった膝デモを直接供給)。
# 単一レバー: DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3 をrep4 ckptに。他は baseline(repro) 完全同一。
# 出力: results/eval_pf/truepf_trace512_warm{i}_s0.npz, /tmp/warm512_times.txt, /tmp/warm512.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE=512; REPS="${REPS:-5}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-32}"
CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml
CKPT="${CKPT:-experiments/distributed_pcn/run_synth512_repro4/20260608_151556/iteration_100/model_iter_100.pth}"
if [ ! -f "$CKPT" ]; then echo "[warm] FATAL ckpt missing: $CKPT"; exit 1; fi
TIMES=/tmp/warm512_times.txt; MARK=/tmp/warm512.marker
rm -f "$MARK" "$TIMES"
echo "[warm] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS CKPT=$CKPT"
train_one () {
  local i="$1"; local t0 t1; t0=$(date +%s)
  DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3="$CKPT" DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "warm${i}" "$SCALE" "$NITER" > /tmp/warm512_train_${i}.out 2>&1
  local ex=$?; t1=$(date +%s)
  echo "train rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  echo "[warm] train rep=${i} DONE exit=${ex} sec=$((t1-t0)) $(date +%H:%M:%S)"
}
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[warm] launch train rep=${i} $(date +%H:%M:%S)"; train_one "$i" &
done
wait
echo "[warm] ALL TRAIN DONE $(date +%H:%M:%S)"
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_warm${i}
  ck=$(find "$OUT"/20*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ]; then echo "[warm] MISSING ckpt rep=${i}"; continue; fi
  t0=$(date +%s)
  CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_warm${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/warm512_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s); echo "eval rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  [ "$ex" = 0 ] && echo "[warm] eval rep=${i} saved" || echo "[warm] eval rep=${i} FAIL"
done
echo "[warm] ALL EVAL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "===== TIMES ====="; cat "$TIMES"
