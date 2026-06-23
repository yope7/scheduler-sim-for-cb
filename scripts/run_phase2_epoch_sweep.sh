#!/usr/bin/env bash
# Phase2 の学習量(SUPERVISED_EPOCHS)だけを振るクリーン sweep。他は固定:
#   REF=0(件数正規化OFF), 全オンプレ種ON(recipe既定 thresholds 999999含む), urgency ON。
# = epoch数の純粋効果を見る(前回 off↔p2 にあった「種の有無」交絡を排除)。
# 出力: truepf_trace{SCALE}_p2e{E}_s0.npz
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-64}"
EPOCHS="${EPOCHS:-0 5 15 50 150}"
NITER="${NITER:-100}"
MAXJOBS="${MAXJOBS:-4}"
NPROC="${NPROC:-32}"
CFG=experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml
MARK=/tmp/phase2_epoch_sweep.marker; rm -f "$MARK"
echo "[p2sweep] START $(date +%H:%M:%S) SCALE=$SCALE EPOCHS=[$EPOCHS] NITER=$NITER"
for e in $EPOCHS; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[p2sweep] launch epochs=$e $(date +%H:%M:%S)"
  PCN_TRAIN_PF_BALANCE_REF=0 DISTRIBUTED_PCN_SUPERVISED_EPOCHS="$e" DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "p2e${e}" "$SCALE" "$NITER" > /tmp/p2e_${SCALE}_${e}.out 2>&1 &
done
wait
echo "[p2sweep] ALL TRAIN DONE $(date +%H:%M:%S)"
for e in $EPOCHS; do
  OUT=experiments/distributed_pcn/run_synth${SCALE}_p2e${e}
  ck=$(find "$OUT"/20*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
  if [ -z "$ck" ]; then echo "[p2sweep] MISSING ckpt epochs=$e (see /tmp/p2e_${SCALE}_${e}.out)"; continue; fi
  echo "[p2sweep] eval epochs=$e ck=${ck##*/} $(date +%H:%M:%S)"
  CKPT="$ck" CFG=$CFG NJ="$SCALE" SEEDS=0 NCMD=40 KSAMP=1 NPROC="$NPROC" \
    OUT="truepf_trace${SCALE}_p2e${e}_s0.npz" PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py \
    > /tmp/eval_p2e_${SCALE}_${e}.out 2>&1 \
    && echo "[p2sweep]   saved truepf_trace${SCALE}_p2e${e}_s0.npz" || echo "[p2sweep]   EVAL FAIL epochs=$e"
done
echo "[p2sweep] ALL EVAL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > "$MARK"
