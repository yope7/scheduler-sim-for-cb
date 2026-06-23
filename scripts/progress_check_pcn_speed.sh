#!/usr/bin/env bash
# 高速化タスク用: 1〜3分で終わる進捗チェック（本番フルランは走らせない）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH=.

STAMP="$(date +%Y%m%d_%H%M%S)"
echo "=== PCN speed progress @ ${STAMP} ==="

# 目標: ベースライン left_tail_p2e100_j24 総 4306s → 半分以下 2153s
BASELINE_TOTAL_SEC=4306
TARGET_TOTAL_SEC=2153

echo "--- 実装フラグ (import 前 env) ---"
"$PYTHON" - <<'PY'
import os
from src.distributed.distributed_pcn_cli import apply_distributed_pcn_cli_env, apply_left_tail_training_env
apply_distributed_pcn_cli_env()
apply_left_tail_training_env()
import src.distributed.distributed_pcn as d
print(f"  DISTRIBUTED_EVAL={d.USE_DISTRIBUTED_EVAL}")
print(f"  LOG_RAY_TRANSFER={d._LOG_RAY_TRANSFER}")
print(f"  GC_EACH_ITER={d._GC_EACH_ITER}")
print(f"  EVAL_QUIET={d._EVAL_QUIET}")
print(f"  UNIFORM_PF_DIST={os.environ.get('DISTRIBUTED_PCN_UNIFORM_PF_DISTRIBUTED')}")
PY

echo "--- マイクロベンチ (教師cacheサンプリング) ---"
"$PYTHON" - <<'PY'
import os, time
import numpy as np
os.environ["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1"
from src.distributed.distributed_pcn_cli import apply_left_tail_training_env
apply_left_tail_training_env()

# 軽量 PCN: replay/cache のみ計測
from src.agents.pcn_agent import PCN

n_ep, ep_len, obs_dim = 400, 24, 220
batch_size = 2048
n_draws = 200

rng = np.random.default_rng(0)
observations = rng.standard_normal((n_ep * ep_len, obs_dim), dtype=np.float32)
actions = rng.integers(0, 2, size=n_ep * ep_len, dtype=np.int64)
desired_returns = rng.standard_normal((n_ep * ep_len, 2), dtype=np.float32)
desired_horizons = rng.random(n_ep * ep_len, dtype=np.float32) * 40
lengths = np.full(n_ep, ep_len, dtype=np.int64)
offsets = np.arange(0, n_ep * ep_len, ep_len, dtype=np.int64)
flat = np.ones(n_ep * ep_len, dtype=np.float64)
padded = PCN._build_padded_step_cdf(lengths, offsets, flat)
cache = {
    "observations": observations,
    "actions": actions,
    "desired_returns": desired_returns,
    "desired_horizons": desired_horizons,
    "episode_lengths": lengths,
    "episode_offsets": offsets,
    "episode_probs": None,
    "flat_step_probs": flat,
    "padded_step_cdf": padded,
    "on_device": False,
}
sampler = type("S", (), {"np_random": np.random.default_rng(0)})()

t0 = time.perf_counter()
for _ in range(n_draws):
    PCN._sample_training_flat_indices(sampler, cache, batch_size)
t1 = time.perf_counter()
sec = t1 - t0
per_update_est = sec / n_draws
# 本番: phase2 100*100 + phase3 100*100 ≈ 20000 update 相当（粗い）
est_train_sample_sec = per_update_est * 20000
print(f"  sample_batch x{n_draws}: {sec:.2f}s ({per_update_est*1000:.2f} ms/draw)")
print(f"  粗算 サンプリングのみ ~{est_train_sample_sec/60:.1f} min (2万draw; GPU学習は別)")
PY

echo "--- 走行中プロセス ---"
pgrep -af "distributed_pcn_event|bench_left_tail|prod_lt_" || echo "  (なし)"

echo "--- 本番ベンチ (prod_lt_*) ---"
PROD="$(ls -td experiments/distributed_pcn/prod_lt_* 2>/dev/null | head -1 || true)"
if [[ -n "${PROD}" && -f "${PROD}/pcn_run.log" ]]; then
  echo "  prod=${PROD}"
  grep -E 'フェーズ[123]|Iter [0-9]+/|総経過時間' "${PROD}/pcn_run.log" 2>/dev/null | tail -6 | sed 's/\x1b\[[0-9;]*m//g' || true
fi

echo "--- 直近ベンチ / 実験ログ ---"
LATEST="$(ls -td experiments/distributed_pcn/prod_lt_* experiments/distributed_pcn/bench_lt_* experiments/distributed_pcn/left_tail_* 2>/dev/null | head -1 || true)"
BLOG=""
if [[ -n "${LATEST}" && -f "${LATEST}/bench.log" ]]; then
  BLOG="${LATEST}/bench.log"
elif [[ -n "${LATEST}" && -f "${LATEST}/pcn_run.log" ]]; then
  BLOG="${LATEST}/pcn_run.log"
fi
if [[ -n "${BLOG}" ]]; then
  echo "  log=${BLOG}"
  grep -E 'フェーズ[123]完了|総経過時間|PROFILE Iter.*Learner|per-iter mean' "${BLOG}" 2>/dev/null | tail -8 | sed 's/\x1b\[[0-9;]*m//g' || true
  tail -3 "${BLOG}" | sed 's/\x1b\[[0-9;]*m//g' || true
else
  echo "  (bench_lt / left_tail ログなし)"
fi

echo "--- git 変更ファイル (高速化関連) ---"
git -C "$ROOT" status -s -- src/agents/pcn_agent.py src/distributed/distributed_pcn.py src/utils/pf_eval_gap.py src/distributed/distributed_pcn_cli.py scripts/ 2>/dev/null | head -20 || true

echo "--- 目標 ---"
echo "  baseline_total=${BASELINE_TOTAL_SEC}s → target<=${TARGET_TOTAL_SEC}s (2x)"
echo "  次: bench_left_tail_quick.sh または本番コマンドの A/B (未計測なら要実行)"
echo "=== done ==="
