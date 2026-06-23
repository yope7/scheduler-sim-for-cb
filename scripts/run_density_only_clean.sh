#!/usr/bin/env bash
# density単独 再実行: 手動帯(episode PF重み)を全OFF明示 → cli/profile の setdefault 汚染を排除し
# 密度逆数版"一本"で戦わせる。baseline(現行フル手動帯)は density_ab/baseline を再利用。
# step重み(MID_STEP/KNEE_STEP/EVALIKE_STEP)は両条件とも自動設定のまま=共通基盤(交絡なし)。
set -uo pipefail
ROOT=/home/noguchi/scheduler-sim-for-cb
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
out="$ROOT/experiments/distributed_pcn/density_ab/density"
rm -rf "$out"; mkdir -p "$out"
echo "[density] START $(date +%H:%M:%S) out=$out"
env \
  DISTRIBUTED_PCN_OUTPUT_DIR="$out" \
  DISTRIBUTED_PCN_JOBS=128 \
  DISTRIBUTED_PCN_USE_EVENT_OBS=1 \
  SCHEDULER_LEARNER_BITMAP=0 \
  DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0 \
  DISTRIBUTED_PCN_EVAL_DIAG=1 \
  DISTRIBUTED_PCN_PHASE2_IMPORTANCE=0 \
  DISTRIBUTED_PCN_QUICK=0 \
  DISTRIBUTED_PCN_N_ITERATIONS=50 \
  DISTRIBUTED_PCN_EVAL_INTERVAL=25 \
  DISTRIBUTED_PCN_SUPERVISED_EPOCHS=100 \
  DISTRIBUTED_PCN_EVAL_SAMPLES=200 \
  DISTRIBUTED_PCN_SKIP_FINAL_EVAL=0 \
  DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1 \
  PCN_ADAPTIVE_RETURN_NORMALIZATION=1 \
  PCN_CONDITIONING_SENS_WEIGHT=0.03 \
  PCN_CONDITIONING_KL_MARGIN=0.08 \
  PCN_COND_ADD_SCALE=0.25 \
  PCN_S_EMB_DROPOUT=0.08 \
  PCN_VALUE_REPRO_WEIGHT=0 \
  PCN_EVAL_PF_GRID=64 \
  PCN_EVAL_STOCHASTIC=0 \
  PCN_TRAIN_COST_ENDPOINT_WEIGHT=0 \
  PCN_TRAIN_LOW_WAIT_PF_WEIGHT=0 \
  PCN_TRAIN_KNEE_PF_WEIGHT=0 \
  PCN_TRAIN_MID_PF_WEIGHT=0 \
  PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=0 \
  PCN_TRAIN_PF_DENSITY_WEIGHT=8 \
  PCN_TRAIN_PF_DENSITY_K=2 \
  PCN_TRAIN_PF_DENSITY_ALPHA=1.0 \
  "$PYTHON" -u -m src.distributed.distributed_pcn_event --conditioning --no-viz > "$out/train.log" 2>&1
rc=$?
echo "[density] DONE $(date +%H:%M:%S) exit=$rc"
grep -aE '総経過時間|MO_HV|LowWaitPF重み対象|Cost端重み対象|MidPF重み対象' "$out/train.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -5
