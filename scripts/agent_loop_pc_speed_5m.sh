#!/usr/bin/env bash
# Cursor 監視シェル用: 5分ごとに progress_check を実行しログへ追記、エージェント wake 用 sentinel を出す。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${PCN_SPEED_LOOP_LOG:-$ROOT/experiments/distributed_pcn/pcn_speed_loop.log}"
INTERVAL_SEC="${PCN_SPEED_LOOP_INTERVAL_SEC:-300}"
mkdir -p "$(dirname "$LOG")"

run_tick() {
  local tick_id="$1"
  {
    echo "======== tick ${tick_id} @ $(date -Iseconds) ========"
    "$ROOT/scripts/progress_check_pcn_speed.sh"
    echo "======== tick ${tick_id} done ========"
  } >>"$LOG" 2>&1
  # 監視シェル notify 用（stdout のみ）
  echo "AGENT_LOOP_TICK_pc_speed {\"prompt\":\"5分進捗: experiments/distributed_pcn/pcn_speed_loop.log と走行中 prod_lt_*/pcn_run.log を確認。高速化達成度・本番ベンチ進捗・次の実装を短く報告し、未達ならコード改善を継続。\",\"tick\":\"${tick_id}\",\"log\":\"${LOG}\"}"
}

tick=0
run_tick "$tick"
while true; do
  sleep "$INTERVAL_SEC"
  tick=$((tick + 1))
  run_tick "$tick"
done
