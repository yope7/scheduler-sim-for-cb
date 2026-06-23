#!/usr/bin/env bash
# 512ジョブ(trace)再現性テスト: 最新レシピ(run_synthetic_urgency.sh の既定=Phase2 50 /
# 件数正規化 REF32 α=auto(512→1.0) / 全オンプレ種 / urgency obs)で REPS 回 学習し、
# 各回 eval して greedy PF を保存。学習は非決定的(Ray async)なので、同一設定でも回ごとに
# 揺れる → 「PFが広く出る」ことが再現するかを見る。各回の実行時間(秒)も記録。
# 出力: truepf_trace512_repro{i}_s0.npz, /tmp/repro512_times.txt, /tmp/repro512.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-512}"
REPS="${REPS:-5}"
NITER="${NITER:-100}"
MAXJOBS="${MAXJOBS:-3}"        # 同時学習数 (3*16=48 actors <= 64 cores, 余裕を残しタイミングを汚さない)
NPROC="${NPROC:-32}"           # eval 並列
CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml
TIMES=/tmp/repro512_times.txt
MARK=/tmp/repro512.marker
rm -f "$MARK" "$TIMES"
echo "[repro] START $(date +%H:%M:%S) SCALE=$SCALE REPS=$REPS NITER=$NITER MAXJOBS=$MAXJOBS CFG=$CFG"

# --- 学習(時間計測つき, 並列) ---
train_one () {
  local i="$1"; local t0 t1
  t0=$(date +%s)
  DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "repro${i}" "$SCALE" "$NITER" > /tmp/repro512_train_${i}.out 2>&1
  local ex=$?
  t1=$(date +%s)
  echo "train rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  echo "[repro] train rep=${i} DONE exit=${ex} sec=$((t1-t0)) $(date +%H:%M:%S)"
}
for i in $(seq 1 "$REPS"); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[repro] launch train rep=${i} $(date +%H:%M:%S)"
  train_one "$i" &
done
wait
echo "[repro] ALL TRAIN DONE $(date +%H:%M:%S)"

# --- eval(時間計測つき, 逐次; 各 eval は内部で NPROC 並列) ---
for i in $(seq 1 "$REPS"); do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_repro${i}
  ck=$(find "$OUT"/20*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ]; then echo "[repro] MISSING ckpt rep=${i} (see /tmp/repro512_train_${i}.out)"; continue; fi
  t0=$(date +%s)
  CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="truepf_trace${SCALE}_repro${i}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/repro512_eval_${i}.out 2>&1
  ex=$?; t1=$(date +%s)
  echo "eval rep=${i} exit=${ex} sec=$((t1-t0))" >> "$TIMES"
  if [ "$ex" = 0 ]; then echo "[repro] eval rep=${i} saved (sec=$((t1-t0)))"; else echo "[repro] eval rep=${i} FAIL (see /tmp/repro512_eval_${i}.out)"; fi
done
echo "[repro] ALL EVAL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "===== TIMES ====="; cat "$TIMES"
