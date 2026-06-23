#!/usr/bin/env bash
# /loop 用: HTML 網羅性レビューを 3 回（間隔 PCN_HTML_REVIEW_INTERVAL_SEC、既定 120秒）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${PCN_HTML_REVIEW_INTERVAL_SEC:-120}"
HTML="$ROOT/docs/pcn_speedup_report_20260602.html"
PROMPT='HTML網羅性レビュー pass={pass}/3: scripts/review_pcn_speedup_html.sh を実行。docs/pcn_speedup_report_20260602.html を読み、欠落があれば追記し #review-entries に pass 記録。'

for pass in 1 2 3; do
  sleep "$INTERVAL"
  if ! "$ROOT/scripts/review_pcn_speedup_html.sh" "$pass"; then
    echo "AGENT_LOOP_TICK_html_review {\"prompt\":\"${PROMPT}\",\"pass\":${pass},\"status\":\"missing\",\"html\":\"${HTML}\"}"
  else
    echo "AGENT_LOOP_TICK_html_review {\"prompt\":\"${PROMPT}\",\"pass\":${pass},\"status\":\"ok\",\"html\":\"${HTML}\"}"
  fi
done
echo 'AGENT_LOOP_DONE_html_review {"prompt":"3回レビュー完了。HTML最終確認をユーザーに報告。"}'
