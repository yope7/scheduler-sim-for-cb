#!/usr/bin/env bash
# ジョブ数粒度実験(J256/J1024/J4096): 18Jの最良設定C3を固定し、ジョブ数だけを変えて学習する。
# C3 = PCN_TEACH_FRONT_ONLY=1 PCN_REFIT_EVERY=25 PCN_REFIT_EPOCHS=2000 PCN_REFIT_COLD=1
#      PCN_LABEL_G=1 PCN_DEDUP_TRAIN_WEIGHT=1 DISTRIBUTED_PCN_LEARNING_RATE=1e-4
#      PCN_TRAIN_HEAD_STEP_WEIGHT=20 PCN_TRAIN_HEAD_STEP_FRAC=0.15
# (既存run_2x2_c3_s1/s2/s3(18J)と同一レシピ。基盤変数はそのpcn_run.logの実測値
#  (EVAL_GAP iter=10刻み, EVAL/VIS samples=64)に合わせてある。configのn_iterations/n_actors/
#  initial_episodes/conditioning/mid_core/workload_adaptive/use_event_obsはyml側で完結)。
# 共通基盤(2x2グリッド実験全セルc0-c3で共通、run_2x2_c3budgetとのcheckpoint次元比較(224 vs 227)
# から確定): SCHEDULER_OBS_URGENCY=1 SCHEDULER_OBS_EFFICIENCY=1 PCN_FOURIER_CMD=1 PCN_FC_DEPTH=4
# (SCHEDULER_OBS_BUDGET_RATIOはc3budgetセルで試して効果なしと確認済みのため既定OFFのまま)。
set -u
cd /home/noguchi/scheduler-sim-for-cb
TAG="${1:?usage: run_jscale_c3.sh TAG CFG JOBS}"
CFG="${2:?usage: run_jscale_c3.sh TAG CFG JOBS}"
JOBS="${3:?usage: run_jscale_c3.sh TAG CFG JOBS}"
OUT="experiments/distributed_pcn/run_jscale_c3_${TAG}"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[jscale_c3] TAG=$TAG JOBS=$JOBS CFG=$CFG OUT=$OUT START=$(date +%H:%M:%S)"
DISTRIBUTED_PCN_CONFIG="$CFG" \
DISTRIBUTED_PCN_OUTPUT_DIR="$OUT" \
DISTRIBUTED_PCN_EVAL_INTERVAL=10 \
DISTRIBUTED_PCN_EVAL_SAMPLES=64 \
DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
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
