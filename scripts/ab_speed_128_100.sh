#!/usr/bin/env bash
# 128job/100iter の A/B 学習: 高速化ON(新既定) vs OFF(legacy)。総経過時間と Phase1/2/3 内訳を比較。
# OOM回避で直列実行（学習中に別学習を起動しない）。出力は両 train.log + 本サマリ。
set -u
cd /home/noguchi/scheduler-sim-for-cb
SUM=experiments/distributed_pcn/ab_speed_128_100_summary.txt
: > "$SUM"

extract() {  # $1=run dir name prefix label, $2=OUT dir
  local label="$1" out="$2"
  local log="$out/train.log"
  echo "==== $label ====" | tee -a "$SUM"
  grep -E "総経過時間|フェーズ[123]" "$log" 2>/dev/null | tee -a "$SUM"
  cat "$out/done.txt" 2>/dev/null | tee -a "$SUM"
  echo "" | tee -a "$SUM"
}

echo "[AB] START $(date +%H:%M:%S)" | tee -a "$SUM"

# --- 1) 高速化 ON (新既定: 何も設定しない=全部ON) ---
echo "[AB] === FAST (新既定 ALL ON) 128/100 ===" | tee -a "$SUM"
bash scripts/run_synthetic_urgency.sh fast 128 100
extract "FAST (ALL ON)" experiments/distributed_pcn/run_synth128_fast

# --- 2) 高速化 OFF (legacy) ---
echo "[AB] === LEGACY (ALL OFF) 128/100 ===" | tee -a "$SUM"
PCN_FAST_ENV=0 \
PCN_FAST_QUEUE_ROLL=0 \
PCN_FAST_UPDATE=0 \
PCN_ACTOR_WEIGHTS_REF=0 \
PCN_REUSE_URGENCY_ALLOC=0 \
PCN_FWD_NANCHECK=1 \
PCN_UPDATE_NANCHECK=1 \
bash scripts/run_synthetic_urgency.sh legacy 128 100
extract "LEGACY (ALL OFF)" experiments/distributed_pcn/run_synth128_legacy

echo "[AB] DONE $(date +%H:%M:%S)" | tee -a "$SUM"
echo "[AB] summary -> $SUM" | tee -a "$SUM"
