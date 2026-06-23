#!/usr/bin/env bash
# /goal 実験: 512 trace の「効率の再現性」を上げられるか。
# 診断= frozen-PF 教師に効率デモ(677件)は全run同一で入るのに、追従できるのは1/5。
#   → 効率(low-wait/knee)デモの損失比を上げ「効率basinを深く」してSGDが毎回そこへ落ちるか試す。
# 単一レバー: PCN_TRAIN_LOW_WAIT_PF_WEIGHT 10→25, PCN_TRAIN_KNEE_PF_WEIGHT 8→18。他は baseline と同一。
# 出力: results/eval_pf/truepf_trace512_anchor{i}_s0.npz, /tmp/anchor512_times.txt, /tmp/anchor512.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE=512; REPS="${REPS:-5}"; NITER="${NITER:-100}"; MAXJOBS="${MAXJOBS:-3}"; NPROC="${NPROC:-32}"
CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml
TIMES=/tmp/anchor512_times.txt; MARK=/tmp/anchor512.marker
rm -f "$MARK" "$TIMES"
echo "[anchor] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS LOW_WAIT=25 KNEE=18"
train_one () {
  local i="$1"; local t0 t1; t0=$(date +%s)
  PCN_TRAIN_LOW_WAIT_PF_WEIGHT=25 PCN_TRAIN_KNEE_PF_WEIGHT=18 DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "anchor${i}" "$SCALE" "$NITER" > /tmp/anchor512_train_${i}.out 2>&1
  local ex=$?; t1=$(date +%s)
  echo "train rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  echo "[anchor] train rep=${i} DONE exit=${ex} sec=$((t1-t0)) $(date +%H:%M:%S)"
}
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[anchor] launch train rep=${i} $(date +%H:%M:%S)"; train_one "$i" &
done
wait
echo "[anchor] ALL TRAIN DONE $(date +%H:%M:%S)"
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_anchor${i}
  ck=$(find "$OUT"/20*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ]; then echo "[anchor] MISSING ckpt rep=${i}"; continue; fi
  t0=$(date +%s)
  CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/truepf_trace${SCALE}_anchor${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/anchor512_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s); echo "eval rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  [ "$ex" = 0 ] && echo "[anchor] eval rep=${i} saved" || echo "[anchor] eval rep=${i} FAIL"
done
echo "[anchor] ALL EVAL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > "$MARK"; echo "===== TIMES ====="; cat "$TIMES"
