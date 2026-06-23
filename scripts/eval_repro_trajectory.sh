#!/usr/bin/env bash
# early-stop本命検証: baseline(repro, コールドスタート)各runの中間ckptを評価し
# 「効率的だった瞬間が早期iterに隠れていないか」を見る。warm2は iter10=103%→iter30=46% と
# 続学習が効率を壊した。baselineでも同様なら early-stop(ベストckpt選択)で効率再現が可能=本質的。
# 出力: results/eval_pf/truepf_trace512_repro{i}_iter{X}_s0.npz, /tmp/reprotraj.marker
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE=512; CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml; NPROC="${NPROC:-32}"
RUNS="${RUNS:-1 2 3 4 5}"; ITERS="${ITERS:-010 030 050 070 090 100}"
MARK=/tmp/reprotraj.marker; rm -f "$MARK"
echo "[rtraj] START $(date +%H:%M:%S) RUNS=$RUNS ITERS=$ITERS"
for i in $RUNS; do
  base=$(find experiments/distributed_pcn/run_synth${SCALE}_repro${i}/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  for X in $ITERS; do
    ck="$base/iteration_${X}/model_iter_${X}.pth"
    if [ ! -f "$ck" ]; then echo "[rtraj] miss repro${i} iter${X}"; continue; fi
    out="results/eval_pf/truepf_trace${SCALE}_repro${i}_iter${X}_s0.npz"
    [ -f "$out" ] && { echo "[rtraj] skip(exists) repro${i} iter${X}"; continue; }
    CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
      OUT="$out" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/rtraj_${i}_${X}.out 2>&1
    echo "[rtraj] repro${i} iter${X} exit=$? $(date +%H:%M:%S)"
  done
done
echo "DONE $(date +%H:%M:%S)" > "$MARK"
echo "[rtraj] ALL DONE $(date +%H:%M:%S)"
