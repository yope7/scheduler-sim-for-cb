#!/usr/bin/env bash
# レジーム混合モデル vs 単一レジームベースライン を、学習内/未知の複数レジームで評価。
# 各条件: eval_b2_compare(greedy) → 追従corr/フロント幅 を後段Pythonで集計。
set -u
cd /home/noguchi/scheduler-sim-for-cb
mkdir -p results/eval_pf/regmix
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"
NPROC="${NPROC:-16}"
SCFG=experiments/distributed_pcn/job_synthetic_pcn.yml
TCFG=experiments/distributed_pcn/job_trace_256_pcn.yml
REGMIX_CK=$(find experiments/distributed_pcn/run_synth256_regmix/*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
BASE_CK=$(find experiments/distributed_pcn/run_synth256_tv05_4/*/iteration_* -name 'model_iter_*.pth' | sort -V | tail -1)
echo "regmix=$REGMIX_CK"
echo "base=$BASE_CK"

# 条件: tag | cfg | nj | extra-env | 種別
CONDS=(
 "syn_rho175|$SCFG|256|SYNTH_TAIL_LEVEL=0.5|学習内(ρ175)"
 "syn_rho1|$SCFG|256|SYNTH_TAIL_LEVEL=0.5 SCHEDULER_ARRIVAL_SCALE=175|学習内(ρ1,trace的)"
 "syn_rho50|$SCFG|256|SYNTH_TAIL_LEVEL=0.5 SCHEDULER_ARRIVAL_SCALE=3.5|未知内挿(ρ50)"
 "trace|$TCFG|256||未知実trace(ρ1.1)"
 "syn_n128|$SCFG|128|SYNTH_TAIL_LEVEL=0.5|未知下scale(128)"
)
for model in regmix base; do
  CK=$([ "$model" = regmix ] && echo "$REGMIX_CK" || echo "$BASE_CK")
  for c in "${CONDS[@]}"; do
    IFS='|' read -r tag cfg nj extra kind <<< "$c"
    env $RECIPE $extra SCHEDULER_OBS_URGENCY=1 CKPT="$CK" CFG="$cfg" NJ="$nj" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
      OUT="results/eval_pf/regmix/${model}_${tag}.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/rmev_${model}_${tag}.out 2>&1
    echo "[$model] $tag ($kind) exit=$?"
  done
done
echo "BATTERY DONE $(date +%H:%M:%S)"
