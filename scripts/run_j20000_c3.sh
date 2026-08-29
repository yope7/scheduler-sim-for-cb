#!/usr/bin/env bash
# 2万ジョブ・正容量(ρ=0.70)のC3学習。scripts/run_jscale_c3.sh と同一レシピ(18J/256J/1024Jと
# 比較可能にするため条件付け・観測フラグは一字一句そのまま)に、大規模で必要な2点だけ足す:
#   1. DISTRIBUTED_PCN_JOBS を明示 (config の n_jobs が別値にクロバーされる既知バグの回避
#      = pcn-1024-njobs-clobber-bug。config 側 nb_jobs=20000 と二重で担保)
#   2. PCN_GPU_FACTORY=1 (GPU rollout/eval 工場。CPU env の O(N^2) 評価を O(N) へ置換)
# EVAL_SAMPLES(学習中evalの指令格子)は run_jscale_c3.sh と同じ 64 が既定。
# usage: [NITER=100 INITEP=100 EVALINT=10 EVALSAMP=64 NACTORS=16] run_j20000_c3.sh TAG CFG JOBS
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAG="${1:?usage: run_j20000_c3.sh TAG CFG JOBS}"
CFG="${2:?}"; JOBS="${3:?}"
NITER="${NITER:-100}"; INITEP="${INITEP:-100}"; EVALINT="${EVALINT:-10}"
EVALSAMP="${EVALSAMP:-64}"; NACTORS="${NACTORS:-16}"
# FACTORY=1: GPU rollout工場 / FACTORY=0: 従来のCPU Actor並列。
# 2万ジョブでは工場カーネルが 'frag' 溢れで再実行を繰り返し CPU より遅い実測があるため切替可能にした。
FACTORY="${FACTORY:-1}"
EVALPOOL="${EVALPOOL:-0}"   # PCN_EVAL_ACTOR_POOL: 学習中evalを評価専用Actorプールで回す(CPU経路の高速化)
OUT="${OUT:-experiments/distributed_pcn/run_j20000_c3_${TAG}}"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[j20000_c3] TAG=$TAG JOBS=$JOBS NITER=$NITER INITEP=$INITEP EVALINT=$EVALINT EVALSAMP=$EVALSAMP NACTORS=$NACTORS CFG=$CFG OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG="$CFG" \
DISTRIBUTED_PCN_OUTPUT_DIR="$OUT" \
DISTRIBUTED_PCN_JOBS="$JOBS" \
DISTRIBUTED_PCN_N_ITERATIONS="$NITER" \
DISTRIBUTED_PCN_INITIAL_EPISODES="$INITEP" \
DISTRIBUTED_PCN_N_ACTORS="$NACTORS" \
DISTRIBUTED_PCN_EVAL_INTERVAL="$EVALINT" \
DISTRIBUTED_PCN_EVAL_SAMPLES="$EVALSAMP" \
DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
PCN_GPU_FACTORY="$FACTORY" PCN_GPU_FACTORY_NUM_GPUS=1 \
PCN_EVAL_ACTOR_POOL="$EVALPOOL" \
SCHEDULER_OBS_URGENCY=1 SCHEDULER_OBS_EFFICIENCY=1 PCN_FOURIER_CMD=1 PCN_FC_DEPTH=4 \
PCN_OBS_LOG=1 \
PCN_TEACH_FRONT_ONLY=1 PCN_REFIT_EVERY=25 PCN_REFIT_EPOCHS=2000 PCN_REFIT_COLD=1 \
PCN_LABEL_G=1 PCN_DEDUP_TRAIN_WEIGHT=1 \
DISTRIBUTED_PCN_LEARNING_RATE=1e-4 \
PCN_TRAIN_HEAD_STEP_WEIGHT=20 PCN_TRAIN_HEAD_STEP_FRAC=0.15 \
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -u -m src.distributed.distributed_pcn_event \
  --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
TRAIN_EXIT=$?
EXEC=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "DONE TAG=$TAG JOBS=$JOBS EXEC=$EXEC TRAIN_EXIT=$TRAIN_EXIT END=$(date +%H:%M:%S)" | tee "$OUT/done.txt"
