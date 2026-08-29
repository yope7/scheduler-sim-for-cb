#!/usr/bin/env bash
# 裾ダイヤルの因果実験: synth256 の SYNTH_TAIL_LEVEL を L∈{LEVELS} で振り、各Lで REPS本学習。
# 目的=「run間ばらつき(分散)が裾の重さと共に増えるか」を測る(崩壊の崖でなく分散トレンド)。
# scale/recipe を固定し裾だけ変える。学習シードは時刻ベース非決定=各repは独立サンプル。
# 2本/GPU(GPU0,GPU1)で並列。学習のみ(evalは後段Pythonでeval_b2_compare)。
# 使い方: LEVELS="0.0 0.5 1.0" REPS=4 bash scripts/run_synth256_tailvar.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
LEVELS="${LEVELS:-0.0 0.5 1.0}"
REPS="${REPS:-4}"
SCALE="${SCALE:-256}"
NITER="${NITER:-100}"
RECIPE="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_FAST_UPDATE=1 PCN_EVAL_ACTOR_POOL=8"
MARK=/tmp/synth256_tailvar.marker; rm -f "$MARK"
echo "[tailvar] START $(date +%H:%M:%S) LEVELS=[$LEVELS] REPS=$REPS SCALE=$SCALE"

launch(){ # L rep gpu
  local L="$1" rep="$2" gpu="$3"
  local TAG="tv$(echo "$L"|tr -d '.')_${rep}"
  env $RECIPE SYNTH_TAIL_LEVEL="$L" CUDA_VISIBLE_DEVICES="$gpu" \
    bash scripts/run_synthetic_urgency.sh "$TAG" "$SCALE" "$NITER" \
    > /tmp/tailvar_${TAG}.out 2>&1
  echo "[tailvar] DONE L=$L rep=$rep gpu=$gpu $(date +%H:%M:%S)"
}

# (L,rep) の全組を作り、2本ずつ(GPU0/GPU1)で回す。
# rep優先順(各repで全レベルを1本ずつ)=部分結果でも裾トレンドが見える。
combos=()
for rep in $(seq 1 "$REPS"); do for L in $LEVELS; do combos+=("$L:$rep"); done; done
i=0
while [ $i -lt ${#combos[@]} ]; do
  a="${combos[$i]}"; b="${combos[$((i+1))]:-}"
  La="${a%:*}"; ra="${a#*:}"
  echo "[tailvar] launch $a (gpu0)${b:+ + $b (gpu1)} $(date +%H:%M:%S)"
  launch "$La" "$ra" 0 &
  pidA=$!
  if [ -n "$b" ]; then Lb="${b%:*}"; rb="${b#*:}"; launch "$Lb" "$rb" 1 & pidB=$!; fi
  wait $pidA; [ -n "$b" ] && wait ${pidB:-}
  i=$((i+2))
done
echo "ALL DONE $(date +%H:%M:%S)" > "$MARK"
echo "[tailvar] ALL DONE $(date +%H:%M:%S)"
