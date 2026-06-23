#!/usr/bin/env bash
# Ver.2(fdタグ)の完了セルを rich eval(offset=1自動・密度Wは学習時のみ影響)。
# 学習と競合しないよう低並列(PAR)。各seedの iteration_100/model_iter_100.pth を評価し /tmp/v2rich_{tag}_{i}.json へ。
# 使い方: TAGS="fd0000 fd0001 fd0010 fd0011" PAR=2 bash scripts/v2_rich_eval_batch.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAGS="${TAGS:-fd0000 fd0001 fd0010 fd0011}"
PAR="${PAR:-2}"
run_eval(){ local tag="$1" i="$2"
  local d ck
  d=$(ls -dt experiments/distributed_pcn/run_synth256_${tag}_${i}/2026* 2>/dev/null | head -1)
  ck="$d/iteration_100/model_iter_100.pth"
  if [ ! -f "$ck" ]; then echo "{\"tag\":\"$tag\",\"seed\":$i,\"err\":\"no ckpt\"}" > /tmp/v2rich_${tag}_${i}.json; return; fi
  CKPT="$ck" NJ=256 PYTHONPATH=. uv run python scripts/rich_eval_cell.py > /tmp/v2rich_${tag}_${i}.json 2>/tmp/v2rich_${tag}_${i}.err
  echo "[v2eval] $tag _$i done $(date +%H:%M:%S)"
}
echo "[v2eval] START $(date +%H:%M:%S) TAGS='$TAGS' PAR=$PAR"
for tag in $TAGS; do for i in 1 2 3; do
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do sleep 3; done
  run_eval "$tag" "$i" &
done; done
wait
echo "[v2eval] ALL DONE $(date +%H:%M:%S)"
echo "DONE $(date +%H:%M:%S)" > /tmp/v2rich_batch.marker