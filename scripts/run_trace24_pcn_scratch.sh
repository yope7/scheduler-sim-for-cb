#!/usr/bin/env bash
# トレース24 PF — scratch 一発（ckpt 継続なし）。Run9 相当の固定設定。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

export DISTRIBUTED_PCN_CONFIG="${DISTRIBUTED_PCN_CONFIG:-experiments/distributed_pcn/job_trace_24_scratch_pass.yml}"
export REF_NPZ="${REF_NPZ:-experiments/distributed_pcn/trace24_no_outlier_sampled_exhaustive_pf.npz}"
export EVAL_GRID="${EVAL_GRID:-32}"
export PCN_FLAGS="${PCN_FLAGS:---conditioning --mid-core --no-viz}"

# import 前に固定（pcn_agent モジュール定数 + Phase2 スキップ）
export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0
export PCN_TRAIN_KNEE_PF_WEIGHT=8
export PCN_TRAIN_LOW_SLOPE_PF_WEIGHT=6
export PCN_TRAIN_LOW_WAIT_PF_WEIGHT="${PCN_TRAIN_LOW_WAIT_PF_WEIGHT:-10}"
export PCN_TRAIN_LOW_WAIT_MAX="${PCN_TRAIN_LOW_WAIT_MAX:-600}"
export PCN_EVAL_GAP_BOOST_MAX=5.0
export PCN_CHOOSE_COMMANDS_MODE=pf_archive
export PCN_EVAL_COMMAND_MODE=pf_ref
export PCN_REF_PF_NPZ="${REF_NPZ:-experiments/distributed_pcn/trace24_no_outlier_sampled_exhaustive_pf.npz}"
export PCN_PF_COMMAND_JITTER=0
export PCN_PF_COMMAND_LOW_WAIT_MAX="${PCN_PF_COMMAND_LOW_WAIT_MAX:-600}"
export PCN_PF_COMMAND_LOW_WAIT_FRAC="${PCN_PF_COMMAND_LOW_WAIT_FRAC:-0.35}"
export PCN_PF_COMMAND_LOW_WAIT_QUOTA="${PCN_PF_COMMAND_LOW_WAIT_QUOTA:-8}"
export DISTRIBUTED_PCN_CMD_OUTCOMES="${DISTRIBUTED_PCN_CMD_OUTCOMES:-1}"
export PCN_SCORE_LOW_WAIT_MAX="${PCN_SCORE_LOW_WAIT_MAX:-600}"
export PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC="${PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC:-0.05}"
export DISTRIBUTED_PCN_N_ITERATIONS=100

STAMP="$(date +%Y%m%d_%H%M%S)"
export DISTRIBUTED_PCN_OUTPUT_DIR="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/trace24_scratch_${STAMP}}"
LOG="$DISTRIBUTED_PCN_OUTPUT_DIR/pcn_run.log"
mkdir -p "$DISTRIBUTED_PCN_OUTPUT_DIR"

ray stop --force 2>/dev/null || true
echo "[trace24_scratch] CONFIG=$DISTRIBUTED_PCN_CONFIG OUT=$DISTRIBUTED_PCN_OUTPUT_DIR GRID=$EVAL_GRID iter=100"
PYTHONUNBUFFERED=1 "$PYTHON" -u -m src.distributed.distributed_pcn_event $PCN_FLAGS 2>&1 | tee "$LOG"

EXEC="$(find "$DISTRIBUTED_PCN_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -path '*/model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
PF_DIR="$ROOT/experiments/distributed_pcn"

DISTRIBUTED_PCN_CONFIG="$DISTRIBUTED_PCN_CONFIG" DISTRIBUTED_PCN_USE_EVENT_NATIVE=1 PYTHONPATH=. "$PYTHON" - <<PY
import json, os, shutil
from pathlib import Path
import numpy as np
from scripts.eval_uniform_command_pf import run
from scripts.score_eval_pf_vs_reference import score_eval_vs_reference, load_npz_pf

ckpt = Path("$CKPT")
snap = Path("$SNAP")
exec_dir = Path("$EXEC")
grid = int("$EVAL_GRID")
stats = run(ckpt, snap, exec_dir, "scratch_final", grid=grid, device="cpu",
            config_path=os.environ.get("DISTRIBUTED_PCN_CONFIG"),
            command_mode=os.environ.get("PCN_EVAL_COMMAND_MODE", "pf_ref"),
            ref_pf_npz=Path("$REF_NPZ"))
pf_pts = np.asarray(stats.get("score_points") or stats["achieved_points"], dtype=np.float64)
all_pts = np.asarray(stats["achieved_points"], dtype=np.float64)
cmd_targets = np.asarray(stats.get("command_targets") or [], dtype=np.float64)
np.savez(exec_dir / "eval_pf_points.npz", points=all_pts, pareto_front=pf_pts,
         command_targets=cmd_targets)
ref = load_npz_pf(Path("$REF_NPZ"))
score_pts = all_pts if stats.get("command_mode") in ("pf_ref", "pf_archive") else pf_pts
score = score_eval_vs_reference(
    score_pts, ref,
    eval_is_pf_commands=(stats.get("command_mode") in ("pf_ref", "pf_archive")),
    command_targets=cmd_targets if len(cmd_targets) else None,
    low_wait_max=float(os.environ.get("PCN_SCORE_LOW_WAIT_MAX", "0")),
    min_low_wait_covered_frac=float(os.environ.get("PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC", "0")),
)
(exec_dir / "pf_score.json").write_text(json.dumps(score, indent=2) + "\\n")
shutil.copy(stats["plot"], "$PF_DIR/trace24_no_outlier_pcn_eval_pf_wa_latest.png")
print(json.dumps(score, indent=2))
raise SystemExit(0 if score["passed"] else 1)
PY

echo "[trace24_scratch] exec=$EXEC score=$EXEC/pf_score.json"
