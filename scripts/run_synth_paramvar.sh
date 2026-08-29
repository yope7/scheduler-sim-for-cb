#!/usr/bin/env bash
# 合成ジョブのパラメータOFAT探索: 安定ベース(synth L=0.5, scale256)から1因子ずつ変え、
# どのジョブパラメータが run間分散/崩壊を生むかを切り分ける。指標=追従corr(後段eval)。
# 各セル REPS本、2本/GPU。学習後にeval_b2_compareも同param環境で回す(workload一致)。
# セル定義: "tag|scale|param-env"。param-env は run_synthetic_urgency.sh に env で渡す。
set -u
cd /home/noguchi/scheduler-sim-for-cb
REPS="${REPS:-3}"; NITER="${NITER:-100}"; NPROC="${NPROC:-24}"
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_FAST_UPDATE=1 PCN_EVAL_ACTOR_POOL=8"
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml
mkdir -p results/eval_pf/paramvar
MARK=/tmp/synth_paramvar.marker; rm -f "$MARK"

# --- セル定義(tag|scale|param-env) --- 全て SYNTH_TAIL_LEVEL=0.5 ベース
CELLS=(
  "pvBurst4|256|SYNTH_TAIL_LEVEL=0.5 SYNTH_TAIL_BURST=4"
  "pvNodes3|256|SYNTH_TAIL_LEVEL=0.5 SYNTH_MAX_NODES=3"
  "pvN128|128|SYNTH_TAIL_LEVEL=0.5"
  "pvN512|512|SYNTH_TAIL_LEVEL=0.5"
)
echo "[paramvar] START $(date +%H:%M:%S) REPS=$REPS cells=${#CELLS[@]}"

# (cell,rep) の全組を rep優先順で
jobs=()
for rep in $(seq 1 "$REPS"); do for c in "${CELLS[@]}"; do jobs+=("$c#$rep"); done; done

train_one(){ # spec(tag|scale|penv#rep) gpu
  local spec="$1" gpu="$2"
  local body="${spec%#*}" rep="${spec##*#}"
  local tag="${body%%|*}" rest="${body#*|}"; local scale="${rest%%|*}" penv="${rest#*|}"
  env $RECIPE $penv CUDA_VISIBLE_DEVICES="$gpu" \
    bash scripts/run_synthetic_urgency.sh "${tag}_${rep}" "$scale" "$NITER" \
    > /tmp/pv_${tag}_${rep}.out 2>&1
  echo "[paramvar] TRAINED ${tag}_${rep} (scale=$scale gpu=$gpu) $(date +%H:%M:%S)"
}
i=0
while [ $i -lt ${#jobs[@]} ]; do
  a="${jobs[$i]}"; b="${jobs[$((i+1))]:-}"
  train_one "$a" 0 & pA=$!
  [ -n "$b" ] && { train_one "$b" 1 & pB=$!; }
  wait $pA; [ -n "$b" ] && wait ${pB:-}
  i=$((i+2))
done
echo "[paramvar] ALL TRAIN DONE $(date +%H:%M:%S)"

# --- eval(同param環境, seed0共通instance) ---
for rep in $(seq 1 "$REPS"); do for c in "${CELLS[@]}"; do
  tag="${c%%|*}"; rest="${c#*|}"; scale="${rest%%|*}"; penv="${rest#*|}"
  OUTd=experiments/distributed_pcn/run_synth${scale}_${tag}_${rep}
  EXEC=$(find "$OUTd"/20* -maxdepth 0 -type d 2>/dev/null | tail -1)
  ck="$EXEC/final_model.pth"; [ -f "$ck" ] || ck=$(find "$EXEC"/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  [ -f "$ck" ] || { echo "[paramvar] MISSING ${tag}_${rep}"; continue; }
  env $RECIPE $penv CKPT="$ck" CFG=$CFG NJ="$scale" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="results/eval_pf/paramvar/${tag}_${rep}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/pveval_${tag}_${rep}.out 2>&1
  echo "[paramvar] EVAL ${tag}_${rep} exit=$? $(date +%H:%M:%S)"
done; done
echo "ALL DONE $(date +%H:%M:%S)" > "$MARK"
echo "[paramvar] ALL DONE $(date +%H:%M:%S)"
