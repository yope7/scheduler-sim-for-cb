#!/usr/bin/env bash
# アブレーション harness: 24ジョブ urgency レシピを GPU1 で回し、完走後に崩壊カーブを自動測定。
# 介入ノブは「外側の env」で渡す（例: PCN_FROZEN_PF_CLONE=1 bash run_ablation.sh frozen 24）。
# run_synthetic_urgency.sh の inline env に無いノブはそのまま通過する。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NAME="${1:?usage: [KNOBS...] run_ablation.sh NAME [JOBS] [NITER]}"
JOBS="${2:-24}"
NITER="${3:-100}"
# 注: CUDA_VISIBLE_DEVICES によるGPU固定はこのコードの ray.get_gpu_ids() ベースの
# デバイス選択と非互換（文字列を torch.cuda.set_device に渡して落ちる）。
# 24ジョブは軽量なので GPU0 を synth1024 と共有して回す（16GB空き・低負荷）。
echo "[ablation] NAME=$NAME JOBS=$JOBS START=$(date +%H:%M:%S)"
echo "[ablation] knobs: FROZEN=${PCN_FROZEN_PF_CLONE:-} BALANCE=${PCN_COMMAND_BALANCE:-} ANCHORS=${PCN_PF_COMMAND_ANCHORS:-} LR=${DISTRIBUTED_PCN_LEARNING_RATE:-} SENS=${PCN_CONDITIONING_SENS_WEIGHT:-}"
bash scripts/run_synthetic_urgency.sh "$NAME" "$JOBS" "$NITER"
# --- 完走後: 崩壊カーブを自動測定 ---
EX=$(find experiments/distributed_pcn/run_synth${JOBS}_${NAME} -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
echo "[ablation] measuring collapse curve for $EX"
EXEC="$EX" CFG=experiments/distributed_pcn/job_synthetic_pcn.yml NJOBS=$JOBS OBS_URGENCY=1 \
  LABEL="$NAME (synth$JOBS)" OUT=collapse_${NAME}_synth${JOBS}.png \
  PYTHONPATH=. .venv/bin/python -u scripts/collapse_curve.py
echo "[ablation] DONE NAME=$NAME  curve=collapse_${NAME}_synth${JOBS}.png END=$(date +%H:%M:%S)"
