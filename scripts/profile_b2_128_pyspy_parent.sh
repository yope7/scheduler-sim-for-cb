#!/usr/bin/env bash
# py-spy を「親」として run を起動し、子(Ray Learner/Actor)ごと flamegraph を採取する。
# ptrace_scope=1 でも py-spy は自分の子孫をプロファイルできる。
# Learner の update ループに集中させるため EVAL は実質OFF・Actorは4本に削減（学習計算自体は不変）。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${1:-14}"
FMT="${2:-flamegraph}"        # flamegraph(SVG) or speedscope(JSON)
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml
OUT="experiments/distributed_pcn/profile_b2_128_pyspy"
ART=docs/figures/pyspy
mkdir -p "$OUT" "$ART"
PYSPY=/home/noguchi/.local/bin/py-spy
if [ "$FMT" = "speedscope" ]; then OUTFILE="$ART/learner.speedscope.json"; else OUTFILE="$ART/learner_flame.svg"; fi

echo "[pyspy-parent] NITER=$NITER FMT=$FMT OUT=$OUTFILE START=$(date +%H:%M:%S)"
"$PYSPY" record --subprocesses --rate 200 --format "$FMT" --output "$OUTFILE" -- \
  env DISTRIBUTED_PCN_PROFILE=1 \
    PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 PCN_FILM=1 \
    DISTRIBUTED_PCN_CONFIG=$CFG DISTRIBUTED_PCN_JOBS=128 DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
    DISTRIBUTED_PCN_N_ACTORS=4 \
    SCHEDULER_OBS_URGENCY=1 \
    DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS=0,50,150,500 DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES=8 \
    DISTRIBUTED_PCN_SUPERVISED_EPOCHS=50 DISTRIBUTED_PCN_N_ITERATIONS=$NITER \
    DISTRIBUTED_PCN_INITIAL_EPISODES=32 DISTRIBUTED_PCN_EVAL_INTERVAL=9999 DISTRIBUTED_PCN_EVAL_SAMPLES=64 \
    DISTRIBUTED_PCN_REPLAY_TX_BUDGET=1200000 \
    PCN_TRAIN_KNEE_PF_WEIGHT=8 PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=6 \
    PCN_TRAIN_LOW_WAIT_PF_WEIGHT=10 PCN_TRAIN_LOW_WAIT_MAX=0 PCN_TRAIN_LOW_WAIT_FRAC=0.30 \
    PCN_USE_AMP=0 PCN_OBS_LOG=1 \
    PCN_PHASE1_SWEEP_TRAIN_WEIGHT=10 PCN_PF_COMMAND_ANCHORS=16 \
    PCN_CHOOSE_COMMANDS_MODE=pf_archive DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
    PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.distributed.distributed_pcn_event \
      --conditioning --mid-core --no-viz > "$OUT/train.log" 2>&1
echo "[pyspy-parent] DONE=$(date +%H:%M:%S) exit=$?  artifact=$OUTFILE"
ls -la "$OUTFILE" 2>/dev/null
echo "=== PROFILE per-iter ==="; grep -aE 'PROFILE Learner|総経過|フェーズ[123]完了|経過時間' "$OUT/train.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -20