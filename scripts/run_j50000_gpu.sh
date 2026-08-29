#!/usr/bin/env bash
# 5万ジョブ・正容量(ρ=0.675)の C3 学習 — GPU工場混成版 (2026-08-19)。
# 8/17 の敗因と対策:
#   - CPU経路: Actor→ReplayBuffer 転送(オブジェクト列)が 8本/74秒 × 48回で iter0 のまま
#     → Phase1 を GPU 工場の一括生産に置換(配列フロー=転送壁なし)
#   - GPU工場を 2万で捨てた原因の frag/ev_on/amb 溢れ
#     → 5万実測(bench_50k_smoke)に基づき E_ALLOC=8192 / NAMB=2561 / KPICK は引数で
#   - Phase3(学習中rollout)は scan 長=T が逐次律速で 1チャンク〜26分 → PCN_GPU_FACTORY_P3=0
#     で CPU Actor(16並列・1本〜4分)に委譲する混成
# usage: [NITER=30 INITEP=24 EVALINT=30 EVALSAMP=16 NACTORS=16 KPICK=128 CHUNK=128] \
#          run_j50000_gpu.sh TAG
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAG="${1:?usage: run_j50000_gpu.sh TAG}"
CFG="experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml"
JOBS=50000
NITER="${NITER:-30}"; INITEP="${INITEP:-24}"; EVALINT="${EVALINT:-20}"
EVALSAMP="${EVALSAMP:-16}"; NACTORS="${NACTORS:-16}"
# CHUNK: B方向並列はスループットに効かない実測(24秒/ep固定)なので、小さくしても損なし。
# 96 は工場の観測バッファ等と合算で 28.11GiB 要求→OOM したため 64 に(v1 の教訓)。
KPICK="${KPICK:-128}"; CHUNK="${CHUNK:-64}"
OUT="${OUT:-experiments/distributed_pcn/run_j50000_gpu_${TAG}}"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[j50000_gpu] TAG=$TAG NITER=$NITER INITEP=$INITEP(×$NACTORS actors) EVALINT=$EVALINT KPICK=$KPICK CHUNK=$CHUNK OUT=$OUT START=$(date +%H:%M:%S)"
# --- [2026-08-25 フラグ監査による修正] ------------------------------------------
#  * SUP_EPOCHS: 未設定だと workload プロファイルが 0 を入れ Phase2(教師あり)が全無効
#  * S_EMB_DROPOUT=0.08: hindsight冗長性対策(学習時のみ)。Actor側は_load_policy_weightsで
#    eval()に固定したため(2026-08-26)、rollout/evalは決定的なまま学習の対策だけが効く
#  * USE_AMP: 未設定だと AMP=ON。GradScaler の step スキップは検知しづらい
#  * TEACH_FRONT_ONLY: PCN_FROZEN_PF_CLONE=1 が前提で、未設定のため一度も効いていなかった。
#    原論文 §4.4 は「非支配のみに絞ると性能劣化を招く」と警告 → OFF を明示(教師=replay全件)
#  * XLA_MEM_FRACTION は PREALLOCATE=true でないと JAX が参照しない(従来は死にフラグ)
# ------------------------------------------------------------------------------
DISTRIBUTED_PCN_CONFIG="$CFG" \
DISTRIBUTED_PCN_OUTPUT_DIR="$OUT" \
DISTRIBUTED_PCN_JOBS="$JOBS" \
DISTRIBUTED_PCN_N_ITERATIONS="$NITER" \
DISTRIBUTED_PCN_INITIAL_EPISODES="$INITEP" \
DISTRIBUTED_PCN_N_ACTORS="$NACTORS" \
DISTRIBUTED_PCN_EVAL_INTERVAL="$EVALINT" \
DISTRIBUTED_PCN_EVAL_SAMPLES="$EVALSAMP" \
DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${SUP_EPOCHS:-50}" \
PCN_S_EMB_DROPOUT="${S_EMB_DROPOUT:-0.08}" \
PCN_USE_AMP="${USE_AMP:-0}" \
PCN_GPU_FACTORY=1 PCN_GPU_FACTORY_NUM_GPUS=1 \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PREALLOC:-false}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.92}" \
PCN_GPU_FACTORY_P3="${FACTORY_P3:-0}" \
PCN_GPU_FACTORY_CHEAP_TO_ACTOR="${CHEAP_TO_ACTOR:-1}" \
PCN_GPU_FACTORY_CHUNK="$CHUNK" \
PCN_GPU_DEFER_E_ALLOC=8192 \
PCN_GPU_DEFER_NAMB=2561 PCN_GPU_DEFER_NAMB_CL=2561 \
PCN_GPU_DEFER_KPICK="$KPICK" \
PCN_EVAL_ACTOR_POOL="${EVALPOOL:-0}" \
SCHEDULER_OBS_URGENCY=1 SCHEDULER_OBS_EFFICIENCY=1 PCN_FOURIER_CMD=1 PCN_FC_DEPTH=4 \
PCN_OBS_LOG=1 \
PCN_TEACH_FRONT_ONLY="${TEACH_FRONT_ONLY:-0}" PCN_REFIT_EVERY="${REFIT_EVERY:-25}" PCN_REFIT_EPOCHS=2000 PCN_REFIT_COLD=1 \
DISTRIBUTED_PCN_REPLAY_TX_BUDGET="${TX_BUDGET:-1200000}" \
PCN_LABEL_G=1 PCN_DEDUP_TRAIN_WEIGHT=1 \
DISTRIBUTED_PCN_LEARNING_RATE="${LR:-1e-4}" \
PCN_TRAIN_HEAD_STEP_WEIGHT=20 PCN_TRAIN_HEAD_STEP_FRAC=0.15 \
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
TRAIN_EXIT=$?
EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "DONE TAG=$TAG EXEC=$EXEC TRAIN_EXIT=$TRAIN_EXIT END=$(date +%H:%M:%S)" | tee "$OUT/done.txt"
