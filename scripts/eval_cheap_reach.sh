#!/usr/bin/env bash
# 修正版の「安い角への到達」を横断評価する。
# 各 EXEC（学習ディレクトリ）について eval_pf_trace を回し、
#   achieved_min / cost_max （= 0 に近いほど安い角に届く＝良い）と span/F を出す。
# usage: GPU=1 NCMD=40 ./scripts/eval_cheap_reach.sh NJ:EXEC [NJ:EXEC ...]
#   例: ./scripts/eval_cheap_reach.sh 512:experiments/.../run_synth512_film512p2s/2026...
set -u
cd /home/noguchi/scheduler-sim-for-cb
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml
GPU="${GPU:-1}"; NCMD="${NCMD:-40}"
printf "%-26s %-8s %-12s %-10s %-8s\n" "label" "NJ" "ach_min/max" "F" "span"
for spec in "$@"; do
  NJ="${spec%%:*}"; EXEC="${spec#*:}"; EXEC="${EXEC%/}"
  CK="$EXEC/iteration_100/model_iter_100.pth"
  [ -f "$CK" ] || { printf "%-26s %-8s %s\n" "$(basename "$EXEC")" "$NJ" "NO_CKPT"; continue; }
  OUT="pf_trace_$(basename "$EXEC")_${NJ}.png"
  line=$(CUDA_VISIBLE_DEVICES=$GPU CKPT="$CK" CFG=$CFG EXEC="$EXEC" NJ=$NJ NCMD=$NCMD OBS_URGENCY=1 \
    OUT="$OUT" PYTHONPATH=. .venv/bin/python -u scripts/eval_pf_trace.py 2>&1 \
    | grep -aoE "F=[+0-9.eE-]+ span=[0-9.eE-]+ achieved_cost\[[0-9,]+\]")
  F=$(sed -nE 's/.*F=([+0-9.eE-]+).*/\1/p' <<<"$line")
  span=$(sed -nE 's/.*span=([0-9.eE-]+).*/\1/p' <<<"$line")
  ac=$(sed -nE 's/.*achieved_cost\[([0-9]+),([0-9]+)\].*/\1 \2/p' <<<"$line")
  amin=$(awk '{print $1}' <<<"$ac"); amax=$(awk '{print $2}' <<<"$ac")
  ratio=$(awk -v a="$amin" -v b="$amax" 'BEGIN{ if(b>0) printf "%.3f", a/b; else print "0" }')
  printf "%-26s %-8s %-12s %-10s %-8s\n" "$(basename "$EXEC")" "$NJ" "${amin:-?}/${amax:-?}" "${F:-?}" "${span:-?}"
  printf "    -> ach_min/cost_max(=reach, 低いほど安い角に到達)=%s\n" "$ratio"
done
