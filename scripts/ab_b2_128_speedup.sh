#!/usr/bin/env bash
# 同条件A/B: fourier128 を「最適化OFF」→「最適化ON」で連続実行し壁時計を比較。
# 同一config/seed/recipe（run_synthetic_urgency）。直列（他学習を起動しない=OOM回避）。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${1:-100}"
getsec() { grep -aE "総経過時間" "$1" 2>/dev/null | grep -aoE "[0-9.]+秒" | head -1 | tr -d '秒'; }
ph() { grep -aE "フェーズ[123]完了|総経過時間|経過時間:" "$1" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -8; }

echo "[AB] === OFF (PCN_FAST_UPDATE=0 PCN_FAST_ENV=0) === START=$(date +%H:%M:%S)"
PCN_FAST_UPDATE=0 PCN_FAST_ENV=0 PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 \
  bash scripts/run_synthetic_urgency.sh fourier128_off 128 "$NITER"
LOFF=experiments/distributed_pcn/run_synth128_fourier128_off/train.log
echo "[AB] OFF phases:"; ph "$LOFF"

echo "[AB] === ON (all defaults: update+env+sweep) === START=$(date +%H:%M:%S)"
PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 \
  bash scripts/run_synthetic_urgency.sh fourier128_on 128 "$NITER"
LON=experiments/distributed_pcn/run_synth128_fourier128_on/train.log
echo "[AB] ON phases:"; ph "$LON"

OFF=$(getsec "$LOFF"); ON=$(getsec "$LON")
echo "[AB] ====================================="
echo "[AB] OFF total = ${OFF}s   ON total = ${ON}s"
awk -v o="$OFF" -v n="$ON" 'BEGIN{ if(n>0) printf "[AB] SPEEDUP = %.2fx (%.0fs saved)\n", o/n, o-n }'
echo "[AB] DONE=$(date +%H:%M:%S)"