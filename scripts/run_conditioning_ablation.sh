#!/usr/bin/env bash
# 方策つぶれ（conditioning collapse）の診断 + アブレーション学習
# 完了時に notify_done.sh で音を鳴らす
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ABL_ROOT="${ABL_ROOT:-$ROOT/experiments/conditioning_ablation/$STAMP}"
mkdir -p "$ABL_ROOT"

# ユーザ実験に合わせた既定: イベント生観測・ビットマップ復元OFF
export DISTRIBUTED_PCN_USE_EVENT_OBS=1
export SCHEDULER_LEARNER_BITMAP=0
export DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP=1
export DISTRIBUTED_PCN_ABLATION=1
export DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0
export DISTRIBUTED_PCN_EVAL_DIAG=0
export DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1
export DISTRIBUTED_PCN_EVAL_SAMPLES=64
export DISTRIBUTED_PCN_OUTPUT_DIR="$ABL_ROOT"

PHASE1_CACHE="$ABL_ROOT/shared_phase1/initial_episodes_cache.pkl.gz"
PHASE1_DIR="$ABL_ROOT/shared_phase1"
mkdir -p "$PHASE1_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_train() {
  local name="$1"
  shift
  local run_dir="$ABL_ROOT/$name"
  mkdir -p "$run_dir"
  log "=== TRAIN $name ==="
  (
    export DISTRIBUTED_PCN_RUN_DIR="$run_dir"
    env "$@" "$PYTHON" -m src.distributed.distributed_pcn_event \
      --output-dir "$ABL_ROOT" \
      --load-phase1 \
      --initial-episode-cache-path "$PHASE1_CACHE"
  ) 2>&1 | tee "$run_dir/pcn_run.log"
  # 最新 checkpoint
  local ckpt
  ckpt="$(find "$run_dir" -name 'model_iter_*.pth' | sort | tail -1)"
  if [[ -z "$ckpt" ]]; then
    ckpt="$(find "$run_dir" -name 'final_model.pth' | head -1)"
  fi
  if [[ -n "$ckpt" ]]; then
  "$PYTHON" scripts/measure_conditioning_metrics.py \
    --checkpoint "$ckpt" \
    --output "$run_dir/conditioning_metrics.json"
  else
    log "WARN: no checkpoint for $name"
  fi
}

# ---------- Phase 0: 共有 Phase1 キャッシュ（全アブレーションで同一データ） ----------
if [[ ! -f "$PHASE1_CACHE" ]]; then
  log "Collecting shared Phase1 cache (STOP_AFTER_PHASE1)..."
  export DISTRIBUTED_PCN_RUN_DIR="$PHASE1_DIR"
  export DISTRIBUTED_PCN_STOP_AFTER_PHASE1=1
  env \
    DISTRIBUTED_PCN_ABLATION_INITIAL_EP=40 \
    DISTRIBUTED_PCN_ABLATION_N_ACTORS=8 \
    "$PYTHON" -m src.distributed.distributed_pcn_event \
    --output-dir "$ABL_ROOT" \
    --save-phase1 \
    2>&1 | tee "$PHASE1_DIR/phase1_collect.log"
  unset DISTRIBUTED_PCN_STOP_AFTER_PHASE1
  if [[ -f "$PHASE1_DIR/initial_episodes_cache.pkl.gz" && "$PHASE1_DIR/initial_episodes_cache.pkl.gz" != "$PHASE1_CACHE" ]]; then
    cp "$PHASE1_DIR/initial_episodes_cache.pkl.gz" "$PHASE1_CACHE"
  fi
fi

if [[ -f "$PHASE1_CACHE" ]]; then
  log "=== DIAGNOSE replay (shared Phase1) ==="
  "$PYTHON" scripts/diagnose_replay_conditioning.py \
    --cache "$PHASE1_CACHE" \
    --output "$ABL_ROOT/replay_conditioning_diagnosis.json"
else
  log "WARN: Phase1 cache not found; skip replay diagnosis"
fi

# ---------- アブレーション（単一要因） ----------
# A0: ベースライン（現行既定）
run_train "A0_baseline" \
  PCN_TRAIN_COST_ENDPOINT_WEIGHT=32 \
  PCN_PHASE1_SWEEP_TRAIN_WEIGHT=1.0 \
  PCN_S_EMB_DROPOUT=0 \
  PCN_CONDITIONING_SENS_WEIGHT=0

# A1: Cost 端の過剰重みを下げる（データ偏り仮説）
run_train "A1_low_cost_endpoint" \
  PCN_TRAIN_COST_ENDPOINT_WEIGHT=4 \
  PCN_PHASE1_SWEEP_TRAIN_WEIGHT=1.0 \
  PCN_S_EMB_DROPOUT=0 \
  PCN_CONDITIONING_SENS_WEIGHT=0

# A2: Phase1 スイープの学習重みを下げる
run_train "A2_downweight_phase1" \
  PCN_TRAIN_COST_ENDPOINT_WEIGHT=32 \
  PCN_PHASE1_SWEEP_TRAIN_WEIGHT=0.2 \
  PCN_S_EMB_DROPOUT=0 \
  PCN_CONDITIONING_SENS_WEIGHT=0

# A3: A1+A2
run_train "A3_low_cost_and_phase1" \
  PCN_TRAIN_COST_ENDPOINT_WEIGHT=4 \
  PCN_PHASE1_SWEEP_TRAIN_WEIGHT=0.2 \
  PCN_S_EMB_DROPOUT=0 \
  PCN_CONDITIONING_SENS_WEIGHT=0

# A4: s_emb dropout（観測ショートカット抑制）
run_train "A4_s_emb_dropout" \
  PCN_TRAIN_COST_ENDPOINT_WEIGHT=32 \
  PCN_PHASE1_SWEEP_TRAIN_WEIGHT=1.0 \
  PCN_S_EMB_DROPOUT=0.3 \
  PCN_CONDITIONING_SENS_WEIGHT=0

# A5: conditioning 感度 KL 損失
run_train "A5_cond_sens_loss" \
  PCN_TRAIN_COST_ENDPOINT_WEIGHT=32 \
  PCN_PHASE1_SWEEP_TRAIN_WEIGHT=1.0 \
  PCN_S_EMB_DROPOUT=0 \
  PCN_CONDITIONING_SENS_WEIGHT=0.1

# ---------- 結果集約 ----------
"$PYTHON" - <<'PY' "$ABL_ROOT"
import json, sys
from pathlib import Path
abl = Path(sys.argv[1])
rows = []
for run in sorted(abl.iterdir()):
    m = run / "conditioning_metrics.json"
    if not m.exists():
        continue
    data = json.loads(m.read_text())
    sw = data.get("sweep", {})
    rows.append({
        "run": run.name,
        "n_unique_logits": sw.get("n_unique_logits"),
        "prob_action1_std": sw.get("prob_action1_std"),
        "c_emb_std_mean": data.get("c_emb", {}).get("c_emb_std_mean"),
    })
rows.sort(key=lambda r: (-(r["n_unique_logits"] or 0), -(r["prob_action1_std"] or 0)))
note = "Higher n_unique_logits and prob_action1_std indicate better conditioning"
(abl / "ablation_summary.json").write_text(json.dumps({"runs": rows, "ranking_note": note}, indent=2))
print(json.dumps(rows, indent=2))
PY

log "=== DONE ablation root: $ABL_ROOT ==="
bash "$ROOT/scripts/notify_done.sh" "Conditioning ablation finished: $ABL_ROOT"
