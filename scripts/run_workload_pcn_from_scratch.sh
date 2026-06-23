#!/usr/bin/env bash
# ワークロード適応 PCN を 0 から学習 → Eval PF → 参照 PF とのスコア
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${DISTRIBUTED_PCN_CONFIG:-experiments/distributed_pcn/job_trace_24_no_outlier_pcn.yml}"
REF_NPZ="${REF_NPZ:-experiments/distributed_pcn/trace24_no_outlier_sampled_exhaustive_pf.npz}"
EVAL_GRID="${EVAL_GRID:-24}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${DISTRIBUTED_PCN_OUTPUT_DIR:-$ROOT/experiments/distributed_pcn/wa_trace24_${STAMP}}"
LOG="$OUT/pcn_run.log"
PCN_FLAGS="${PCN_FLAGS:---conditioning --mid-core --no-viz}"

mkdir -p "$OUT"
export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
export DISTRIBUTED_PCN_CONFIG="$CONFIG"
ray stop --force 2>/dev/null || true

echo "[wa_pcn] CONFIG=$CONFIG OUT=$OUT GRID=$EVAL_GRID FLAGS=$PCN_FLAGS"
PYTHONUNBUFFERED=1 "$PYTHON" -u -m src.distributed.distributed_pcn_event $PCN_FLAGS 2>&1 | tee "$LOG"

EXEC="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
CKPT="$(find "$EXEC" -path '*/model_iter_*.pth' | sort -V | tail -1)"
SNAP="$EXEC/learner_replay_snapshot.pkl.gz"
PF_DIR="$ROOT/experiments/distributed_pcn"

DISTRIBUTED_PCN_CONFIG="$CONFIG" DISTRIBUTED_PCN_USE_EVENT_NATIVE=1 PYTHONPATH=. "$PYTHON" - <<PY
import json, shutil
from pathlib import Path
import numpy as np
from scripts.eval_uniform_command_pf import run
from scripts.score_eval_pf_vs_reference import score_eval_vs_reference, load_npz_pf

ckpt = Path("$CKPT")
snap = Path("$SNAP")
out = Path("$OUT")
exec_dir = Path("$EXEC")
grid = int("$EVAL_GRID")
stats = run(ckpt, snap, exec_dir, "wa_final", grid=grid, device="cpu")
pf_pts = np.asarray(stats["pareto_front"], dtype=np.float64)
all_pts = np.asarray(stats["achieved_points"], dtype=np.float64)
np.savez(exec_dir / "eval_pf_points.npz", points=all_pts, pareto_front=pf_pts)
ref = load_npz_pf(Path("$REF_NPZ"))
score = score_eval_vs_reference(pf_pts, ref)
(exec_dir / "pf_score.json").write_text(json.dumps(score, indent=2) + "\\n")
shutil.copy(stats["plot"], "$PF_DIR/trace24_no_outlier_pcn_eval_pf_wa_latest.png")
print(json.dumps(score, indent=2))
raise SystemExit(0 if score["passed"] else 1)
PY

echo "[wa_pcn] exec=$EXEC score=$EXEC/pf_score.json"
