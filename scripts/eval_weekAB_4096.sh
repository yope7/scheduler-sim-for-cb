#!/usr/bin/env bash
# 4096窓レシピ(FILM=0/Fourier4band/FC_DEPTH=4/defer/urgency)の checkpoint を
# 学習週weekA と 未知週weekB で評価して npz を出す。
# usage: bash scripts/eval_weekAB_4096.sh CKPT TAG [WEEK(A|B|AB)]
set -eu
cd /home/noguchi/scheduler-sim-for-cb
CKPT="${1:?usage: eval_weekAB_4096.sh CKPT TAG [WEEK]}"
TAG="${2:?usage: eval_weekAB_4096.sh CKPT TAG [WEEK]}"
WEEK="${3:-AB}"
NPROC="${NPROC:-32}"
NCMD="${NCMD:-20}"
mkdir -p results/eval_pf/regmix

run_one() {
  local cfg="$1" cgmax="$2" out="$3"
  echo "[EVAL] $out (cfg=$cfg CG_MAX=$cgmax)"
  # [重要] 学習は distributed_pcn_event --conditioning --mid-core 経由で
  #   PCN_COND_ADD_SCALE=0.25 (distributed_pcn_cli.py:47 / workload_pcn_profile.py:42)
  #   PCN_COMMAND_BALANCE=1   (distributed_pcn_cli.py:64)
  # を pcn_agent import 前に立てている。これらは forward() を変える（pcn_agent.py:761,815）:
  #   fc(s*c + 0.25*c) vs fc(s*c) / desired_return * command_balance の有無。
  # ここで揃えないと「学習したのと別のネットワーク」を評価することになる。
  # 実測(evregBest weekA, 真PF箱HV): 未設定 31.5% → 揃えると 88.3% (+56.7pt)。
  env PCN_FILM=0 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_FC_DEPTH=4 \
      PCN_OBS_LOG=1 SCHEDULER_OBS_URGENCY=1 \
      PCN_COND_ADD_SCALE="${PCN_COND_ADD_SCALE:-0.25}" \
      PCN_COMMAND_BALANCE="${PCN_COMMAND_BALANCE:-1}" \
      SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 \
      CKPT="$CKPT" CFG="$cfg" NJ=4096 SEEDS=0 NCMD="$NCMD" KSAMP=0 NPROC="$NPROC" \
      CG_MAX="$cgmax" OUT="$out" \
      PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py
}

case "$WEEK" in
  *A*) run_one experiments/distributed_pcn/job_trace_weekAfull_win4096_pcn.yml 1050418097 \
               "results/eval_pf/regmix/${TAG}_weekA.npz" ;;
esac
case "$WEEK" in
  *B*) run_one experiments/distributed_pcn/job_trace_weekBfull_win4096_pcn.yml 596196119 \
               "results/eval_pf/regmix/${TAG}_weekB.npz" ;;
esac
echo "[EVAL] done TAG=$TAG"
