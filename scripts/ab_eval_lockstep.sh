#!/usr/bin/env bash
# 学習中evalを「JAX一括生成(従来)」と「lockstepブロックカーネル」で A/B する。
# 測るのは2つ: ①eval 1回の秒数 ②評価値(PF)がどれだけズレるか。
#   ②が要る理由: この経路は torch NN を使うため JAX 版と fp32 の演算順序が違い、
#   greedy の argmax が僅差で割れうる(gpu_factory._use_lockstep_eval の docstring)。
#   完全一致は設計上諦めている経路なので、ズレ幅を数字で出してから採否を決める。
# Phase1 を 1本に縮めて eval だけを2回踏ませる。学習の質は見ない(速度と一致のみ)。
# usage: ab_eval_lockstep.sh
set -u
cd /home/noguchi/scheduler-sim-for-cb
source tools/cuda_env.sh

for MODE in 0 1; do
  OUT="/tmp/ab_eval_ls_$MODE"
  rm -rf "$OUT"; mkdir -p "$OUT"

  set -a
  source experiments/distributed_pcn/run_j50000_v10_main100/v9_env_export.sh >/dev/null
  set +a

  export PCN_CMD_TRACK_WAIT_WEIGHT="0.3"
  export PCN_COND_WAIT_ROBUST="logexpand"
  export PCN_COND_WAIT_Z0="3e-2"
  export PCN_WAIT_SENS_PROBE=1
  export PCN_ACTOR_ARRAY_EPISODE=1

  export DISTRIBUTED_PCN_OUTPUT_DIR="$OUT"
  export DISTRIBUTED_PCN_N_ITERATIONS=2
  export DISTRIBUTED_PCN_INITIAL_EPISODES=1
  export DISTRIBUTED_PCN_EVAL_INTERVAL=1
  export DISTRIBUTED_PCN_SUPERVISED_EPOCHS=2
  export DISTRIBUTED_PCN_PROFILE=1
  export PCN_GPU_EVAL_LOCKSTEP=$MODE

  echo "[ab_eval] MODE=$MODE (0=JAX一括生成 / 1=lockstep) START=$(date +%H:%M:%S)"
  PYTHONUNBUFFERED=1 PYTHONPATH=. timeout 5400 .venv/bin/python -u \
    -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz \
    > "$OUT/train.log" 2>&1
  echo "[ab_eval] MODE=$MODE exit=$? END=$(date +%H:%M:%S)"

  grep -a 'EVAL_LOCKSTEP' "$OUT/train.log" | head -2
  grep -ao '分散評価(16ep): [0-9.]*s' "$OUT/train.log"
done

echo
echo "=== 評価値のズレ (PF が何点ずれたか) ==="
.venv/bin/python - <<'PY'
import glob, json
import numpy as np
def pf(m):
    f = glob.glob(f"/tmp/ab_eval_ls_{m}/*/pcn_mo_hv.json")
    if not f:
        print(f"  MODE={m}: pcn_mo_hv.json なし"); return None
    return json.load(open(f[0])).get("pareto_fronts_per_eval", [])
a, b = pf(0), pf(1)
if not a or not b:
    raise SystemExit(0)
for i, (x, y) in enumerate(zip(a, b)):
    X = np.unique(np.array(x, dtype=float), axis=0)
    Y = np.unique(np.array(y, dtype=float), axis=0)
    print(f"  eval{i+1}: 従来 {len(X)}点 / lockstep {len(Y)}点")
    if X.shape == Y.shape:
        d = np.abs(X - Y)
        rel = d / np.maximum(np.abs(X), 1e-12)
        print(f"    最大絶対差 cost={d[:,0].max():.4g} wait={d[:,1].max():.4g} / 最大相対差={rel.max():.3e}")
        print(f"    完全一致: {'はい' if d.max()==0 else 'いいえ'}")
    else:
        print("    点数が違う=PFの構成が変わった(要確認)")
PY
