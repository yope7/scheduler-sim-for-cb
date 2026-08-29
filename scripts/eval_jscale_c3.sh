#!/usr/bin/env bash
# ジョブ数粒度実験の外部評価ランナー。
#
# ★重要(2026-08-12に実証): eval_b2_compare.py は学習側が --conditioning/--mid-core 経由で
# 自動設定する条件付けフラグを設定しないため、素で回すと「学習と別のネットワーク」を走らせる。
# 実測(18J run_2x2_c3_s1 iter100, NCMD=16):
#   未設定           L2med=0.4298 出せる点 4/16   ← 崩壊
#   COND_ADDのみ     L2med=0.0661 出せる点12/16
#   +COMMAND_BALANCE L2med=0.1430 出せる点14/16  ← 学習と一致(下記の再現テストで確定)
# 「未設定」と「明示的にOFFを指定」は greedy 出力がビット一致 → 未設定=OFF を厳密に確認済み。
# 根本: src/agents/pcn_agent.py:1276 `if _COMMAND_BALANCE: r = r * self.command_balance`
#   → フラグOFFだと ckpt に保存済みの command_balance([2.802,0.357]) が適用されないまま推論。
#   PCN_COND_ADD_SCALE も同様に条件付け経路を変える。2026-08-10の箱HV 31.5%→88.3% と同型のバグ。
#
# どの設定が学習時のものかは「学習中に実際に使われた指令(cmd_outcomes.jsonl)を外部評価に
# CG_LIST/WG_LIST で与え、学習中の達成点を再現できるか」で決定した(再現L2中央/完全一致数):
#   BALANCE=0,OBS_LOG=0 → 0.2659 / 0件      BALANCE=1,OBS_LOG=0 → 0.2896 / 0件
#   BALANCE=0,OBS_LOG=1 → 0.1344 / 2件      BALANCE=1,OBS_LOG=1 → 0.0634 / 3件  ★最良
# よって学習時設定 = COND_ADD_SCALE=0.25 + COMMAND_BALANCE=1 + OBS_LOG=1 と確定。
# (PCN_OBS_LOG=1 はリポジトリ内の学習スクリプト群でも事実上の標準で、この結論と整合する)
#
# usage: eval_jscale_c3.sh CKPT CFG NJ OUT_NPZ [NCMD] [NPROC]
set -u
cd /home/noguchi/scheduler-sim-for-cb
CKPT="${1:?usage: eval_jscale_c3.sh CKPT CFG NJ OUT_NPZ [NCMD] [NPROC]}"
CFG="${2:?}"; NJ="${3:?}"; OUT_NPZ="${4:?}"
NCMD="${5:-40}"; NPROC="${6:-1}"
mkdir -p "$(dirname "$OUT_NPZ")"
echo "[eval_jscale] CKPT=$CKPT NJ=$NJ NCMD=$NCMD NPROC=$NPROC START=$(date +%H:%M:%S)"
env \
  SCHEDULER_OBS_URGENCY=1 SCHEDULER_OBS_EFFICIENCY=1 \
  PCN_FOURIER_CMD=1 PCN_FC_DEPTH=4 \
  PCN_COND_ADD_SCALE=0.25 PCN_COMMAND_BALANCE=1 PCN_OBS_LOG=1 \
  CKPT="$CKPT" CFG="$CFG" NJ="$NJ" SEEDS=0 NCMD="$NCMD" KSAMP=0 NPROC="$NPROC" \
  OUT="$OUT_NPZ" PYTHONPATH=. .venv/bin/python -u scripts/eval_b2_compare.py
echo "[eval_jscale] DONE exit=$? $(date +%H:%M:%S)"
