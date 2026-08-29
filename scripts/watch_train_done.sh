#!/usr/bin/env bash
# 学習完了の番人: wrapperの生死に依存せず「成果物」だけで完了を判定し、done.txt を自分で書く。
# 背景: setsid起動時にwrapperが先に死ぬと done.txt が永遠に書かれない事故が頻発したための恒久対策。
# usage: watch_train_done.sh EXEC_DIR NITER [INTERVAL_SEC]
#   EXEC_DIR = experiments/.../run_*/20YYMMDD_HHMMSS (タイムスタンプ実行ディレクトリ)
#   完了条件: distributed_pcn_event プロセス不在 かつ iteration_{NITER}/ 存在 → EXEC_DIR/done.txt を書いて exit 0
#   死亡条件: プロセス不在 かつ iteration_{NITER}/ 不在 → EXEC_DIR/dead.txt を書いて exit 1
set -u
EXEC="${1:?usage: watch_train_done.sh EXEC_DIR NITER [INTERVAL]}"
NITER="${2:?usage: watch_train_done.sh EXEC_DIR NITER [INTERVAL]}"
INTERVAL="${3:-30}"
ITER_DIR=$(printf "iteration_%03d" "$NITER")
while true; do
  ALIVE=$(pgrep -f "distributed_pcn_event" | wc -l)
  if [ "$ALIVE" -eq 0 ]; then
    sleep 5  # 起動直後の空白と誤検知しないよう二度見
    ALIVE=$(pgrep -f "distributed_pcn_event" | wc -l)
    if [ "$ALIVE" -eq 0 ]; then
      if [ -d "$EXEC/$ITER_DIR" ]; then
        echo "DONE(watcher) EXEC=$EXEC NITER=$NITER END=$(date +%H:%M:%S)" | tee "$EXEC/done.txt"
        exit 0
      else
        echo "DEAD(watcher) EXEC=$EXEC 最終iter未到達 END=$(date +%H:%M:%S)" | tee "$EXEC/dead.txt"
        exit 1
      fi
    fi
  fi
  sleep "$INTERVAL"
done
