#!/bin/bash
# スケール学習の自走チェーン: 1024(実行中)→4096→40960。
# ワークロード統一: 小ジョブ(SYNTH_MAX_NODES=16)・オンプレ1520・クラウド65536。
# 40960 のみ eval 削減(格子6=~156cmd, 間隔25, samples32, sup20)で~2h に収める。
# 各段の done.txt(TRAIN_EXIT=0)を確認して次へ。失敗したら停止。
set -u
cd /home/noguchi/scheduler-sim-for-cb
export CUDA_VISIBLE_DEVICES=0
export DISTRIBUTED_PCN_ONPREM=1520 DISTRIBUTED_PCN_CLOUD=65536 SYNTH_MAX_NODES=16

wait_done() {  # $1=run dir
  local d="$1" log="$1/train.log"
  while true; do
    if [ -f "$d/done.txt" ]; then
      grep -q "TRAIN_EXIT=0" "$d/done.txt" && { echo "OK $d"; return 0; }
      echo "FAIL $d ($(cat $d/done.txt))"; return 1
    fi
    # 崩壊/プロセス消滅の検知
    if [ -f "$log" ] && grep -qE "Traceback|CUDA out of memory|Killed" "$log"; then
      echo "ERROR-in-log $d"; return 1
    fi
    sleep 30
  done
}

echo "=== [1/3] 1024 完了待ち(実行中) $(date +%H:%M) ==="
wait_done experiments/distributed_pcn/run_synth1024_scale1024 || { echo CHAIN_STOP_1024; exit 1; }

echo "=== [2/3] 4096 開始 $(date +%H:%M) ==="
bash scripts/run_synthetic_urgency.sh scale4096 4096 100
wait_done experiments/distributed_pcn/run_synth4096_scale4096 || { echo CHAIN_STOP_4096; exit 1; }

echo "=== [3/3] 40960 開始(eval削減) $(date +%H:%M) ==="
DISTRIBUTED_PCN_EVAL_INTERVAL=25 DISTRIBUTED_PCN_EVAL_SAMPLES=32 \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS=20 PCN_EVAL_GAP_FEEDBACK_GRID=6 \
  bash scripts/run_synthetic_urgency.sh scale40960 40960 100
wait_done experiments/distributed_pcn/run_synth40960_scale40960 || { echo CHAIN_STOP_40960; exit 1; }

echo "=== SCALE_CHAIN_DONE $(date +%H:%M) ==="
