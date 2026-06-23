#!/usr/bin/env bash
# 長時間ジョブ完了時の通知音（利用可能な方法を順に試す）
msg="${1:-Done}"

if command -v notify-send >/dev/null 2>&1; then
  notify-send "scheduler-sim" "$msg" 2>/dev/null || true
fi

if command -v paplay >/dev/null 2>&1; then
  for f in /usr/share/sounds/freedesktop/stereo/complete.oga \
           /usr/share/sounds/freedesktop/stereo/message.oga; do
    if [[ -f "$f" ]]; then
      paplay "$f" 2>/dev/null && exit 0
    fi
  done
fi

if command -v aplay >/dev/null 2>&1 && [[ -f /usr/share/sounds/alsa/Front_Center.wav ]]; then
  aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null && exit 0
fi

# ターミナルベル
printf '\a'
echo "[notify] $msg"
