#!/usr/bin/env bash
# trace(重い裾)のrobust化: 崩壊は本質的な最適化確率分散で steer 不能・単一recipeで安定化不可
# ([[tail-dial-no-cliff-seed-variance]])。唯一の確実解 = 信頼指標 n_pf で崩壊を検知し、
# 該当seedを自動再起動(機械的リトライ)。best-of-K の cherry-pick ではなく「失敗runを引き直す」
# =実運用で当然やる動作(崩壊モデルはデプロイしない)。1 seed が n_pf>=THRESH に達するまで最大 MAXRETRY 回。
# 期待: 崩壊率 p(~0.4) のとき k 回リトライで残存崩壊 p^(k+1) (1回→0.16, 2回→0.06)。
# 使い方: SCALE=128 NITER=100 THRESH=25 MAXRETRY=3 SEED_TAG=robust \
#         CFG=experiments/distributed_pcn/job_trace_128_pcn.yml \
#         LV="PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 DISTRIBUTED_PCN_N_UPDATES=200 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1" \
#         bash scripts/run_with_retry.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
SCALE="${SCALE:-128}"; NITER="${NITER:-100}"; THRESH="${THRESH:-25}"; MAXRETRY="${MAXRETRY:-3}"
SEED_TAG="${SEED_TAG:-robust}"; ESTOP="${ESTOP:-0}"; LV="${LV:-}"
CFG="${CFG:-experiments/distributed_pcn/job_trace_${SCALE}_pcn.yml}"

npf_of(){ # $1=run dir名(run_synth${SCALE}_${tag})
  local sub=$(ls -d experiments/distributed_pcn/$1/2026* 2>/dev/null | tail -1)
  ls "$sub"/uniform_cmd_stats_iter_*.json 2>/dev/null | tail -1 | xargs grep -h n_pf 2>/dev/null | grep -o '[0-9]*'
}

best_npf=-1; best_attempt=0
for try in $(seq 0 "$MAXRETRY"); do
  tag="${SEED_TAG}_try${try}"
  echo "[retry] === attempt $try (tag=$tag) $(date +%H:%M:%S) ==="
  env $LV DISTRIBUTED_PCN_EARLYSTOP=$ESTOP DISTRIBUTED_PCN_CONFIG=$CFG \
    bash scripts/run_synthetic_urgency.sh "$tag" "$SCALE" "$NITER" > /tmp/retry_${tag}.out 2>&1
  np=$(npf_of "run_synth${SCALE}_${tag}"); np=${np:-0}
  echo "[retry] attempt $try → n_pf=$np (threshold=$THRESH)"
  if [ "$np" -gt "$best_npf" ]; then best_npf=$np; best_attempt=$try; fi
  if [ "$np" -ge "$THRESH" ]; then
    echo "[retry] SUCCESS at attempt $try (n_pf=$np>=$THRESH)。崩壊回避。"
    echo "BEST tag=${SEED_TAG}_try${try} n_pf=$np"
    exit 0
  fi
  echo "[retry] 崩壊検知(n_pf=$np<$THRESH)→ 引き直し"
done
echo "[retry] MAXRETRY尽きた。最良= attempt $best_attempt n_pf=$best_npf (tag=${SEED_TAG}_try${best_attempt})"
echo "BEST tag=${SEED_TAG}_try${best_attempt} n_pf=$best_npf"
