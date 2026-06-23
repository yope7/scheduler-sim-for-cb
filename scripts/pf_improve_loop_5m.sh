#!/usr/bin/env bash
# PF改善ループ — 5分おき
PROMPT='PF改善ループ(5分): pf_improve_run3.log または最新 experiments/distributed_pcn/wa_trace24_*/**/pf_score.json を確認。学習中なら進捗のみ、完了なら passed 判定。未達かつ未起動なら Run4 調整して run_workload_pcn_from_scratch.sh 実行。passed ならループ停止。'
while true; do
  sleep 300
  echo "AGENT_LOOP_TICK_pf_improve {\"prompt\":\"${PROMPT}\"}"
done
