#!/usr/bin/env bash
# trace 小規模(64/128)アブレーション: off / p2 / norm05 / norm10 を各スケールで学習→eval。
#   off    = Phase2 OFF, 件数正規化 OFF, 全オンプレ種なし  (256/512 の before と同条件)
#   p2     = Phase2 ON,  件数正規化 OFF                     (Phase2 単独の効果を分離)
#   norm05 = Phase2 ON,  REF=32 α=0.5                       (recipe 既定の小規模α)
#   norm10 = Phase2 ON,  REF=32 α=1.0                       (過正規化の害を測る)
# off→p2 で Phase2 寄与、p2→norm05→norm10 で件数正規化寄与を分解。
# 出力: truepf_trace{SCALE}_abl_{COND}_s0.npz （eval_b2_compare greedy/samp/rp）。
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALES="${SCALES:-64 128}"
CONDS="${CONDS:-off p2 norm05 norm10}"
NITER="${NITER:-100}"
MAXJOBS="${MAXJOBS:-4}"
NPROC="${NPROC:-32}"
MARK=/tmp/trace_abl_sweep.marker
rm -f "$MARK"
echo "[abl] START $(date +%H:%M:%S) SCALES=[$SCALES] CONDS=[$CONDS] NITER=$NITER"

envfor(){
  case "$1" in
    off)    echo "DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0 PCN_TRAIN_PF_BALANCE_REF=0 DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS=0,50,150,500";;
    p2)     echo "PCN_TRAIN_PF_BALANCE_REF=0";;
    norm05) echo "PCN_TRAIN_PF_BALANCE_REF=32 PCN_TRAIN_PF_BALANCE_ALPHA=0.5";;
    norm10) echo "PCN_TRAIN_PF_BALANCE_REF=32 PCN_TRAIN_PF_BALANCE_ALPHA=1.0";;
  esac
}

# ---- train (並列キャップ) ----
for s in $SCALES; do
  for c in $CONDS; do
    CFG=experiments/distributed_pcn/job_trace_${s}_pcn.yml
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "[abl] launch train s=$s c=$c $(date +%H:%M:%S)"
    env $(envfor "$c") DISTRIBUTED_PCN_CONFIG=$CFG \
      bash scripts/run_synthetic_urgency.sh "abl_${c}" "$s" "$NITER" \
      > /tmp/abl_${s}_${c}.out 2>&1 &
  done
done
wait
echo "[abl] ALL TRAIN DONE $(date +%H:%M:%S)"

# ---- eval (逐次, 各 NPROC) ----
for s in $SCALES; do
  for c in $CONDS; do
    OUT=experiments/distributed_pcn/run_synth${s}_abl_${c}
    ck=$(find "$OUT"/20*/iteration_* -name 'model_iter_*.pth' 2>/dev/null | sort -V | tail -1)
    if [ -z "$ck" ]; then echo "[abl] MISSING ckpt s=$s c=$c (train failed? see /tmp/abl_${s}_${c}.out)"; continue; fi
    echo "[abl] eval s=$s c=$c ck=${ck##*/} $(date +%H:%M:%S)"
    CKPT="$ck" CFG=experiments/distributed_pcn/job_trace_${s}_pcn.yml NJ="$s" SEEDS=0 \
      NCMD=40 KSAMP=1 NPROC="$NPROC" OUT="truepf_trace${s}_abl_${c}_s0.npz" \
      PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py > /tmp/eval_abl_${s}_${c}.out 2>&1 \
      && echo "[abl]   saved truepf_trace${s}_abl_${c}_s0.npz" || echo "[abl]   EVAL FAIL s=$s c=$c"
  done
done
echo "[abl] ALL EVAL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > "$MARK"
