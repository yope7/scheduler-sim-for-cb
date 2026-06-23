#!/usr/bin/env bash
# trace の before(戦略OFF)/after(戦略ON) チェックポイントを eval_b2_compare で評価し npz 化。
# 使い方: bash scripts/eval_plot_trace_compare.sh 512   (or 256)
# greedy(=決定的)中心なので KSAMP=1 で高速。env/seed は trace config で before/after 同一インスタンス。
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${1:?usage: eval_plot_trace_compare.sh SCALE(512|256)}"
CFG="experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml"
BEF_DIR="experiments/distributed_pcn/run_synth${SCALE}_tr${SCALE}before"
AFT_DIR="experiments/distributed_pcn/run_synth${SCALE}_tr${SCALE}after"
ckpt_of(){ find "$1"/20*/iteration_* -name "model_iter_*.pth" 2>/dev/null | sort -V | tail -1; }
BEF_CKPT=$(ckpt_of "$BEF_DIR"); AFT_CKPT=$(ckpt_of "$AFT_DIR")
echo "[eval] SCALE=$SCALE CFG=$CFG"
echo "[eval] before ckpt: $BEF_CKPT"
echo "[eval] after  ckpt: $AFT_CKPT"
[ -z "$BEF_CKPT" ] && { echo "ERROR missing before ckpt"; exit 1; }
[ -z "$AFT_CKPT" ] && { echo "ERROR missing after ckpt"; exit 1; }
NCMD="${NCMD:-40}"; KSAMP="${KSAMP:-1}"; NPROC="${NPROC:-32}"
echo "[eval] running BEFORE eval ($(date +%H:%M:%S)) ..."
CKPT="$BEF_CKPT" CFG="$CFG" NJ="$SCALE" SEEDS=0 NCMD="$NCMD" KSAMP="$KSAMP" NPROC="$NPROC" \
  OUT="truepf_trace${SCALE}_before_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py || exit 2
echo "[eval] running AFTER eval ($(date +%H:%M:%S)) ..."
CKPT="$AFT_CKPT" CFG="$CFG" NJ="$SCALE" SEEDS=0 NCMD="$NCMD" KSAMP="$KSAMP" NPROC="$NPROC" \
  OUT="truepf_trace${SCALE}_after_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py || exit 3
echo "[eval] DONE both evals ($(date +%H:%M:%S))"
