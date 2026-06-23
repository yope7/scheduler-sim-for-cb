#!/usr/bin/env bash
# 汎用 trajectory eval: 指定prefixの各runの中間ckptをiter別に評価。
# 用途: 実用版early-stopが選んだiter vs 真の達成HVピークiter を比較し「選択精度」を測る。
# usage: PREFIX=es RUNS="1 2 3 4 5" bash scripts/eval_traj.sh
# 出力: results/eval_pf/truepf_trace512_{PREFIX}{i}_iter{X}_s0.npz, /tmp/{PREFIX}traj.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE=512; CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml; NPROC="${NPROC:-32}"
PREFIX="${PREFIX:?set PREFIX}"; RUNS="${RUNS:-1 2 3 4 5}"; ITERS="${ITERS:-010 030 050 070 090 100}"
MARK=/tmp/${PREFIX}traj.marker; rm -f "$MARK"
echo "[${PREFIX}traj] START $(date +%H:%M:%S) RUNS=$RUNS ITERS=$ITERS"
for i in $RUNS; do
  base=$(find experiments/distributed_pcn/run_synth${SCALE}_${PREFIX}${i}/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  for X in $ITERS; do
    ck="$base/iteration_${X}/model_iter_${X}.pth"
    if [ ! -f "$ck" ]; then echo "[${PREFIX}traj] miss ${PREFIX}${i} iter${X}"; continue; fi
    out="results/eval_pf/truepf_trace${SCALE}_${PREFIX}${i}_iter${X}_s0.npz"
    [ -f "$out" ] && { echo "[${PREFIX}traj] skip(exists) ${PREFIX}${i} iter${X}"; continue; }
    CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
      OUT="$out" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/${PREFIX}traj_${i}_${X}.out 2>&1
    echo "[${PREFIX}traj] ${PREFIX}${i} iter${X} exit=$? $(date +%H:%M:%S)"
  done
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[${PREFIX}traj] ALL DONE $(date +%H:%M:%S)"
