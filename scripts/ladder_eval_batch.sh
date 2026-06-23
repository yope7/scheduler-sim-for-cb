#!/usr/bin/env bash
# 段階的(ladder)アブレーションの rich eval。合成/trace 両対応(EXTRA_ENVで job_type/CFG/HV参照を渡す)。
# 各構成の iteration_100/model_iter_100.pth を rich_eval_cell.py で評価し /tmp/ladrich_{tag}_{i}.json へ。
# 使い方:
#   TAGS="lad_s_FWD lad_s_FD lad_s_F lad_s_base" NJ=256 PAR=6 \
#   EXTRA_ENV="JOB_TYPE=1 CFG=experiments/distributed_pcn/job_synthetic_pcn.yml HV_COST_REF=1.685e6 HV_WAIT_REF=3183 COST_SCALE=1.46e6" \
#   bash scripts/ladder_eval_batch.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAGS="${TAGS:?need TAGS}"; PAR="${PAR:-4}"; NJ="${NJ:-256}"
EXTRA_ENV="${EXTRA_ENV:-}"
run_eval(){ local tag="$1" i="$2" d ck
  d=$(ls -dt experiments/distributed_pcn/run_synth${NJ}_${tag}_${i}/2026* 2>/dev/null | head -1)
  ck="$d/iteration_100/model_iter_100.pth"
  if [ ! -f "$ck" ]; then echo "{\"tag\":\"$tag\",\"seed\":$i,\"err\":\"no ckpt\"}" > /tmp/ladrich_${tag}_${i}.json; return; fi
  env CKPT="$ck" NJ="$NJ" $EXTRA_ENV PYTHONPATH=. uv run python scripts/rich_eval_cell.py > /tmp/ladrich_${tag}_${i}.json 2>/tmp/ladrich_${tag}_${i}.err
  echo "[ladeval] $tag _$i done $(date +%H:%M:%S)"
}
echo "[ladeval] START $(date +%H:%M:%S) TAGS='$TAGS' NJ=$NJ PAR=$PAR EXTRA='$EXTRA_ENV'"
for tag in $TAGS; do for i in 1 2 3; do
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do sleep 3; done
  run_eval "$tag" "$i" &
done; done
wait
echo "[ladeval] ALL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > /tmp/ladeval_${TAGS%% *}.marker