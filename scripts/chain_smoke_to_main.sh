#!/usr/bin/env bash
# スモークの完了を待ち、合格なら本番100iterを投入する。落ちたら投入せず通知だけ出す。
# 作法: 完走判定は done.txt のみ(10) / 本番は setsid で起動(11) / 番人を同時起動(12) / 通知(7)
set -u
cd /home/noguchi/scheduler-sim-for-cb

SMOKE_LOG=/tmp/smoke_v10_fast.runlog
SMOKE_OUT=/tmp/smoke_v10_fast
TAG=fast100
NITER=100
OUT="experiments/distributed_pcn/run_j50000_v10_${TAG}"
NTFY="ntfy.sh/claudeyope"

notify() {
  curl -s -H "Title: $1" -d "$2" "$NTFY" > /dev/null 2>&1 || true
  echo "[notify] $1 / $2"
}

echo "[chain] スモーク完了を待機 START=$(date +%H:%M:%S)"
DEADLINE=$(( $(date +%s) + 3600 ))
while ! grep -q '\[smoke\] exit=' "$SMOKE_LOG" 2>/dev/null; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    notify "スモーク タイムアウト" "1時間経っても終わらないので本番は投入していません。$SMOKE_OUT/train.log を確認してください。"
    exit 1
  fi
  sleep 20
done

SMOKE_EXIT=$(grep -o '\[smoke\] exit=[0-9]*' "$SMOKE_LOG" | tail -1 | cut -d= -f2)
ERRS=$(grep -acE 'Traceback|CUDA_ERROR|FAILED|Killed' "$SMOKE_OUT/train.log" 2>/dev/null || echo 0)
GETLINE=$(grep -a 'PROFILE Phase3' "$SMOKE_OUT/train.log" 2>/dev/null | tail -1)
ITER2=$(ls -d "$SMOKE_OUT"/*/iteration_002 2>/dev/null | head -1)

echo "[chain] smoke_exit=$SMOKE_EXIT errs=$ERRS iter2=$ITER2"
echo "[chain] $GETLINE"

if [ "$SMOKE_EXIT" != "0" ] || [ "$ERRS" -gt 0 ] || [ -z "$ITER2" ]; then
  notify "スモーク失敗 本番は未投入" "exit=$SMOKE_EXIT errs=$ERRS iter2=${ITER2:-なし}
$GETLINE
$SMOKE_OUT/train.log を確認してください。"
  exit 1
fi

echo "[chain] スモーク合格。本番を投入する $(date +%H:%M:%S)"
notify "スモーク合格 本番100iterを投入" "$GETLINE
出力: $OUT"

NITER=$NITER PCN_ACTOR_ARRAY_EPISODE=1 \
  setsid nohup bash scripts/run_j50000_v10.sh "$TAG" > /tmp/v10_${TAG}.runlog 2>&1 < /dev/null &
disown

echo "[chain] 実行ディレクトリの出現を待つ"
for _ in $(seq 1 60); do
  EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' 2>/dev/null | tail -1)
  [ -n "$EXEC" ] && break
  sleep 10
done

if [ -z "$EXEC" ]; then
  notify "本番 起動失敗の疑い" "10分待っても実行ディレクトリが出来ません。/tmp/v10_${TAG}.runlog を確認してください。"
  exit 1
fi

echo "[chain] EXEC=$EXEC 番人を起動"
setsid nohup bash scripts/watch_train_done.sh "$EXEC" "$NITER" 60 \
  > /tmp/watch_${TAG}.log 2>&1 < /dev/null &
disown

notify "本番100iter 開始" "EXEC=$EXEC
番人とntfyを設定済み。見込み4.5〜5時間。
完了は $EXEC/done.txt、失敗は dead.txt で判定します。"

echo "[chain] 完了を待機"
while [ ! -f "$EXEC/done.txt" ] && [ ! -f "$EXEC/dead.txt" ]; do sleep 60; done

if [ -f "$EXEC/done.txt" ]; then
  TOTAL=$(grep -a '総経過時間' "$EXEC/pcn_run.log" 2>/dev/null | tail -1)
  P3=$(grep -a 'PROFILE Phase3' "$EXEC/pcn_run.log" 2>/dev/null | tail -1)
  notify "本番100iter 完走" "$TOTAL
$P3
比較対象 v9=33,128秒 / v10_main100=33,959秒
EXEC=$EXEC"
else
  notify "本番100iter 途中で停止" "最終iterに到達せず。EXEC=$EXEC
$OUT/train.log の末尾を確認してください。"
fi
