#!/usr/bin/env bash
# トレース24 PF — 安定再現版。
# run_trace24_pcn_scratch.sh (Run9 固定設定) をベースに、
#  (1) 学習後に最終 checkpoint だけでなくアーカイブが豊かだった checkpoint を
#      best-checkpoint 評価で選ぶ (Phase3 mode-collapse 救済)。
#  (2) 安定化/低wait拡張ノブを env で上書き可能にする。
# 主要ノブはすべて env で上書き可 (workflow から sweep する想定)。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

export DISTRIBUTED_PCN_CONFIG="${DISTRIBUTED_PCN_CONFIG:-experiments/distributed_pcn/job_trace_24_scratch_pass.yml}"
export REF_NPZ="${REF_NPZ:-experiments/distributed_pcn/trace24_no_outlier_sampled_exhaustive_pf.npz}"
export PCN_REF_PF_NPZ="${REF_NPZ}"
export EVAL_GRID="${EVAL_GRID:-32}"
export PCN_FLAGS="${PCN_FLAGS:---conditioning --mid-core --no-viz}"

# --- 学習フェーズ設定 (scratch と同じ既定、ただし上書き可) ---
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS="${DISTRIBUTED_PCN_SUPERVISED_EPOCHS:-0}"
export DISTRIBUTED_PCN_N_ITERATIONS="${DISTRIBUTED_PCN_N_ITERATIONS:-100}"
export PCN_TRAIN_KNEE_PF_WEIGHT="${PCN_TRAIN_KNEE_PF_WEIGHT:-8}"
export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT="${PCN_TRAIN_LOW_SLOPE_PF_WEIGHT:-6}"
export PCN_TRAIN_LOW_WAIT_PF_WEIGHT="${PCN_TRAIN_LOW_WAIT_PF_WEIGHT:-10}"
export PCN_TRAIN_LOW_WAIT_MAX="${PCN_TRAIN_LOW_WAIT_MAX:-600}"
export PCN_EVAL_GAP_BOOST_MAX="${PCN_EVAL_GAP_BOOST_MAX:-5.0}"

# --- 安定化ノブ (mode-collapse 抑制: command 感度) ---
# 未指定なら conditioning プロファイル既定 (0.03/0.08/0.25/0.08) を使う。
[ -n "${PCN_CONDITIONING_SENS_WEIGHT:-}" ] && export PCN_CONDITIONING_SENS_WEIGHT
[ -n "${PCN_CONDITIONING_KL_MARGIN:-}" ] && export PCN_CONDITIONING_KL_MARGIN
[ -n "${PCN_COND_ADD_SCALE:-}" ] && export PCN_COND_ADD_SCALE
[ -n "${PCN_S_EMB_DROPOUT:-}" ] && export PCN_S_EMB_DROPOUT

# --- command / 評価モード ---
export PCN_CHOOSE_COMMANDS_MODE="${PCN_CHOOSE_COMMANDS_MODE:-pf_archive}"
export PCN_EVAL_COMMAND_MODE="${PCN_EVAL_COMMAND_MODE:-pf_ref}"
export PCN_PF_COMMAND_JITTER="${PCN_PF_COMMAND_JITTER:-0}"
export PCN_PF_COMMAND_LOW_WAIT_MAX="${PCN_PF_COMMAND_LOW_WAIT_MAX:-600}"
export PCN_PF_COMMAND_LOW_WAIT_FRAC="${PCN_PF_COMMAND_LOW_WAIT_FRAC:-0.35}"
export PCN_PF_COMMAND_LOW_WAIT_QUOTA="${PCN_PF_COMMAND_LOW_WAIT_QUOTA:-8}"
export DISTRIBUTED_PCN_CMD_OUTCOMES="${DISTRIBUTED_PCN_CMD_OUTCOMES:-1}"
export PCN_SCORE_LOW_WAIT_MAX="${PCN_SCORE_LOW_WAIT_MAX:-600}"
export PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC="${PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC:-0.05}"

# --- best-checkpoint 選抜（既定: 全 checkpoint 評価。eval は安価 ~18s/ckpt）---
export BEST_CKPT_TOPK="${BEST_CKPT_TOPK:-99}"
export BEST_CKPT_MIN_ITER_FRAC="${BEST_CKPT_MIN_ITER_FRAC:-0.0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
export DISTRIBUTED_PCN_OUTPUT_DIR="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/trace24_stable_${STAMP}}"
LOG="$DISTRIBUTED_PCN_OUTPUT_DIR/pcn_run.log"
mkdir -p "$DISTRIBUTED_PCN_OUTPUT_DIR"

ray stop --force 2>/dev/null || true
echo "[trace24_stable] CONFIG=$DISTRIBUTED_PCN_CONFIG OUT=$DISTRIBUTED_PCN_OUTPUT_DIR GRID=$EVAL_GRID iter=$DISTRIBUTED_PCN_N_ITERATIONS SENS=${PCN_CONDITIONING_SENS_WEIGHT:-default}"
PYTHONUNBUFFERED=1 "$PYTHON" -u -m src.distributed.distributed_pcn_event $PCN_FLAGS 2>&1 | tee "$LOG"

EXEC="$(find "$DISTRIBUTED_PCN_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
PF_DIR="$ROOT/experiments/distributed_pcn"
echo "[trace24_stable] best-checkpoint eval over $EXEC"

EXEC_DIR="$EXEC" REF_NPZ="$REF_NPZ" DISTRIBUTED_PCN_CONFIG="$DISTRIBUTED_PCN_CONFIG" \
  DISTRIBUTED_PCN_USE_EVENT_NATIVE=1 PYTHONPATH=. "$PYTHON" -m scripts.eval_best_checkpoint
RC=$?

if [ -f "$EXEC/eval_pf_best.png" ]; then
  cp "$EXEC/eval_pf_best.png" "$PF_DIR/trace24_no_outlier_pcn_eval_pf_stable_latest.png" || true
fi
echo "[trace24_stable] exec=$EXEC score=$EXEC/pf_score.json selection=$EXEC/best_checkpoint_selection.json rc=$RC"
exit $RC
