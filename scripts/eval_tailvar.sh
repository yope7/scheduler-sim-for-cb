#!/usr/bin/env bash
# 裾分散実験の評価: 各 tv{L}_{rep} の final_model を、同一レベル・同一eval instance(seed0)で
# eval_b2_compare(greedy, NCMD=40) して達成フロントを保存。eval workload は学習と一致(tail/scale/obs)。
# 同一Lの全repを同一instanceで測る→run間HV差は純粋な方策分散。
# 使い方: LEVELS="0.0 0.5 1.0" REPS=4 bash scripts/eval_tailvar.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
LEVELS="${LEVELS:-0.0 0.5 1.0}"; REPS="${REPS:-4}"; SCALE="${SCALE:-256}"; NPROC="${NPROC:-24}"
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4"
mkdir -p results/eval_pf/tailvar
echo "[eval_tailvar] START $(date +%H:%M:%S)"
for L in $LEVELS; do
  Ltag=$(echo "$L"|tr -d '.')
  for rep in $(seq 1 "$REPS"); do
    TAG="tv${Ltag}_${rep}"
    OUTd=experiments/distributed_pcn/run_synth${SCALE}_${TAG}
    EXEC=$(find "$OUTd"/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
    ck="$EXEC/final_model.pth"
    [ -f "$ck" ] || ck=$(find "$EXEC"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
    OUT="results/eval_pf/tailvar/${TAG}_s0.npz"
    if [ -z "$ck" ] || [ ! -f "$ck" ]; then echo "[eval_tailvar] MISSING $TAG"; continue; fi
    env $RECIPE SYNTH_TAIL_LEVEL="$L" CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
      OUT="$OUT" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/evaltv_${TAG}.out 2>&1
    echo "[eval_tailvar] $TAG exit=$? $(date +%H:%M:%S)"
  done
done
echo "[eval_tailvar] DONE $(date +%H:%M:%S)"
