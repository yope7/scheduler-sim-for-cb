#!/usr/bin/env bash
# pcn_speedup_report HTML の網羅性チェック（必須セクション・実験パス・画像）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HTML="${PCN_SPEEDUP_HTML:-$ROOT/docs/pcn_speedup_report_20260602.html}"
LOG="${PCN_SPEEDUP_REVIEW_LOG:-$ROOT/experiments/distributed_pcn/pcn_speedup_review.log}"
PASS="${1:-?}"
STAMP="$(date -Iseconds)"

missing=()
grep -q '2.1 Conditioning' "$HTML" || missing+=('sec_conditioning')
grep -q '4.1 壁時計' "$HTML" || missing+=('sec_timing')
grep -q 'prod_lt_20260602_070829' "$HTML" || missing+=('prod_path')
grep -q 'left_tail_p2e100_j24_20260602_023001' "$HTML" || missing+=('baseline_path')
grep -q '却下' "$HTML" || missing+=('rejected_ideas')
grep -q 'デッドロック' "$HTML" || missing+=('bugs')

IMG1="$ROOT/experiments/distributed_pcn/prod_lt_20260602_070829/20260602_070832/uniform_cmd_pf_iter_100.png"
IMG2="$ROOT/experiments/distributed_pcn/left_tail_p2e100_j24_20260602_023001/20260602_023004/uniform_cmd_pf_iter_100.png"
[[ -f "$IMG1" ]] || missing+=('img_optimized')
[[ -f "$IMG2" ]] || missing+=('img_baseline')

{
  echo "=== review pass=$PASS @ $STAMP ==="
  if ((${#missing[@]} == 0)); then
    echo "OK: all required sections and assets present"
  else
    echo "MISSING: ${missing[*]}"
  fi
  echo "html=$HTML"
} | tee -a "$LOG"

if ((${#missing[@]} > 0)); then
  exit 1
fi
