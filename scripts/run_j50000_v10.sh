#!/usr/bin/env bash
# v10: オンライン学習に両側の距離罰 (2026-08-27)
#   v9の実効環境(train.logのenvダンプ=唯一の真実)を機械写しし、v10差分だけ上書きする。
#   差分: ①PCN_CMD_TRACK_WAIT_WEIGHT(wait側の片側罰) ②PCN_COND_WAIT_ROBUST=logexpand
#        (wait指令入力の低域対数展開, z0=1e-3) ③PCN_WAIT_SENS_PROBE(毎iterのwait感受性TV出力)
#   評価時も ②の env を一致させること(モデルクラス共有なので env さえ合えば自動で一致)。
# usage: [NITER=20 WAIT_TRACK=0.3] run_j50000_v10.sh TAG
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAG="${1:?usage: run_j50000_v10.sh TAG}"
NITER="${NITER:-20}"
OUT="experiments/distributed_pcn/run_j50000_v10_${TAG}"
# [2026-08-30 R1] rm -rf は廃止。TAG再利用で既存run(復元不能なcheckpoint)を消す事故の遮断。
if [ -e "$OUT" ]; then echo "[ABORT] 既存のrunがあります: $OUT (消すなら手で)"; exit 1; fi
mkdir -p "$OUT"

source tools/cuda_env.sh

V9LOG="experiments/distributed_pcn/run_j50000_gpu_v9_100iter/train.log"
ENVFILE="$OUT/v9_env_export.sh"
.venv/bin/python - "$V9LOG" > "$ENVFILE" <<'PY'
import re
import sys

skip = {"DISTRIBUTED_PCN_OUTPUT_DIR", "DISTRIBUTED_PCN_N_ITERATIONS"}
envs = {}
for line in open(sys.argv[1], errors="ignore"):
    m = re.match(r"^ {4}([A-Z][A-Z0-9_]*)=(\S*)$", line)
    if m:
        envs[m.group(1)] = m.group(2)
n = 0
for k, v in sorted(envs.items()):
    if k in skip:
        continue
    if k.startswith(("PCN_", "SCHEDULER_", "DISTRIBUTED_PCN_", "XLA_")):
        print(f'export {k}="{v}"')
        n += 1
print(f'echo "[v10] v9環境 {n} 変数を機械写し"')
PY
source "$ENVFILE"

export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_N_ITERATIONS="$NITER"
export PCN_CMD_TRACK_WAIT_WEIGHT="${WAIT_TRACK:-0.3}"
export PCN_COND_WAIT_ROBUST="${COND_WAIT_ROBUST:-logexpand}"
export PCN_COND_WAIT_Z0="${COND_WAIT_Z0:-3e-2}"
export PCN_WAIT_SENS_PROBE=1
# [2026-08-30] efficiency 観測(3次元)を外す。観測は 224 → 221 次元。
#   理由1: 3次元目の log(b)=待ち削減 は urgency と 1bit も違わない(実測 6000step で完全一致)。
#          クラウドは弾性で s_cl=到着時刻 が常に成立し b = s_on-arrival = urgency の中身。
#          EFF_GAIN_K=16 を urgency と同じ較正に揃えてあるので正規化後も同値。
#   理由2: 3次元を無効化しても行動が変わらない(iter020/iter100 × 6指令 × 5万step の反実仮想で
#          0で潰す 0.011%/0.001%、時間軸シャッフル 0.006%/0.001%。cloud率も不変)。
#   効果: 観測のための追加の配置探索が1回減る。現行コード(CSRキャッシュ入り)の実測は
#          env 単独 5万step で 22.4→18.2秒 = -18.9%(rollout 全体では約-8%、壁時計で約-4%)。
#          docs の -30%(69.1→48.5) は CSR 修正前の旧コードの値なので使わないこと。
#   検証: 同一行動列 5万step で obs 先頭221次元のハッシュ・待ち・コストが全件一致(最大差0)。
export SCHEDULER_OBS_EFFICIENCY=0

git rev-parse HEAD > "$OUT/git_head.txt" 2>/dev/null
git diff > "$OUT/git_diff.patch" 2>/dev/null

# [2026-08-30 R3] このrunの実効env(機械写し+上書きの合成結果)を保存する。
# <run>/v9_env_export.sh は「v9のenv」であって本runの実効値ではない(OBS_EFFICIENCY等の上書きを含まない)。
# 以後の評価・週末適応ハーネスは、これではなく effective_env.sh を source すること。
env | grep -E '^(PCN_|SCHEDULER_|DISTRIBUTED_PCN_|XLA_)' | sort | sed 's/^/export /' > "$OUT/effective_env.sh"

echo "[v10] TAG=$TAG NITER=$NITER WAIT_TRACK=$PCN_CMD_TRACK_WAIT_WEIGHT \
COND_WAIT=$PCN_COND_WAIT_ROBUST/$PCN_COND_WAIT_Z0 OUT=$OUT START=$(date +%H:%M:%S)"
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
TRAIN_EXIT=$?
EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "DONE TAG=$TAG EXEC=$EXEC TRAIN_EXIT=$TRAIN_EXIT END=$(date +%H:%M:%S)" | tee "$OUT/done.txt"
