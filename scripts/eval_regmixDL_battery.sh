#!/usr/bin/env bash
# 無次元版regmix(regmixDL)を PCN_DIMLESS_NORM=1 付きで5条件eval。envが報酬/目的値を無次元化するので
# 学習と整合。追従corr/フロント幅で税(幅の甘さ)が消えたか見る。
set -u
cd /home/noguchi/scheduler-sim-for-cb
mkdir -p results/eval_pf/regmix
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"
SCFG=experiments/distributed_pcn/job_synthetic_pcn.yml; TCFG=experiments/distributed_pcn/job_trace_256_pcn.yml
CK=$(find experiments/distributed_pcn/run_synth256_regmixDL/*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
echo "regmixDL=$CK"
CONDS=(
 "syn_rho175|$SCFG|256|SYNTH_TAIL_LEVEL=0.5"
 "syn_rho1|$SCFG|256|SYNTH_TAIL_LEVEL=0.5 SCHEDULER_ARRIVAL_SCALE=175"
 "syn_rho50|$SCFG|256|SYNTH_TAIL_LEVEL=0.5 SCHEDULER_ARRIVAL_SCALE=3.5"
 "trace|$TCFG|256|"
 "syn_n128|$SCFG|128|SYNTH_TAIL_LEVEL=0.5"
)
for c in "${CONDS[@]}"; do
  IFS='|' read -r tag cfg nj extra <<< "$c"
  env $RECIPE $extra PCN_DIMLESS_NORM=1 SCHEDULER_OBS_URGENCY=1 CKPT="$CK" CFG="$cfg" NJ="$nj" SEEDS=0 NCMD=40 KSAMP=1 NPROC=16 \
    OUT="results/eval_pf/regmix/regmixDL_${tag}.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/rmdl_${tag}.out 2>&1
  echo "[regmixDL] $tag exit=$?"
done
echo "BATTERY DONE"
