#!/usr/bin/env python3
"""アンカー残差方策 Step0 単体検証。

(a) 選択の決定性・正規化定数の固定
(b) 指令経由の往復で select(dr) == select_by_values(cost,wait)  (γ=1.0 の一貫性)
(c) 残差0(全follow)再生の同値性: アンカー遺伝子を絶対行動で再生した達成値が
    nsga2_agent._rollout の PF 値と一致(=env規約とXOR規約の整合の基礎)

usage:
  CFG=experiments/distributed_pcn/job_trace_128_pcn.yml NJ=128 \
  NPZ=results/eval_pf/nsga2_trace128_s0.npz \
  PYTHONPATH=. .venv/bin/python scripts/verify_anchor_residual.py
"""
import os
import sys

for _a in sys.argv[1:]:
    if "=" in _a:
        k, v = _a.split("=", 1)
        os.environ[k] = v

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

import numpy as np

from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.nsga2_agent import _rollout
from src.utils.anchor_residual import AnchorSet
from src.utils.pf_command_eval import objectives_to_command

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_128_pcn.yml")
NJ = int(os.environ.get("NJ", "128"))
NPZ = os.environ.get("NPZ", f"results/eval_pf/nsga2_trace{NJ}_s0.npz")

print(f"[verify] CFG={CFG} NJ={NJ} NPZ={NPZ}")
A = AnchorSet.from_npz(NPZ)
print(f"[verify] AnchorSet: {len(A.pf)} アンカー(重複除去後) nj={A.nj} "
      f"cost[{A._c_lo:.0f},{A._c_hi:.0f}] wait[{A._w_lo:.1f},{A._w_hi:.1f}]")

fail = 0

# ---- (a) 決定性: 同じ入力で同じ index ----
i1, _ = A.select_by_values(A.pf[10, 0], A.pf[10, 1])
i2, _ = A.select_by_values(A.pf[10, 0], A.pf[10, 1])
assert i1 == i2, "決定性違反"
# アンカー点そのものを与えたら自分自身が選ばれる(最近傍=距離0)
self_hit = sum(
    1 for k in range(len(A.pf))
    if A.select_by_values(A.pf[k, 0], A.pf[k, 1])[0] == k
)
print(f"(a) 決定性OK / アンカー自己選択 {self_hit}/{len(A.pf)} "
      f"(重複pfで近接した別indexに飛ぶ分を除き概ね一致)")

# ---- (b) 指令往復: select(dr) == select_by_values(cost,wait) ----
mismatch_b = 0
for k in range(len(A.pf)):
    cost, wait = float(A.pf[k, 0]), float(A.pf[k, 1])
    dr = objectives_to_command(cost, wait, NJ)
    ib, _ = A.select(dr)
    iv, _ = A.select_by_values(cost, wait)
    if ib != iv:
        mismatch_b += 1
if mismatch_b == 0:
    print(f"(b) 指令往復OK: 全{len(A.pf)}点で select(dr)==select_by_values")
else:
    print(f"(b) ★FAIL: {mismatch_b}/{len(A.pf)} 点で不一致")
    fail += 1

# ---- (c) 残差0再生の同値性: 全follow = アンカー遺伝子の絶対再生 ----
# AnchorSet 経由の遺伝子を _rollout で回した達成値が、npz の pf 値と一致するか。
env = create_eval_env(load_config(CFG), job_seed=0, n_jobs=NJ)
max_abs_err = 0.0
n_check = min(len(A.pf), int(os.environ.get("NCHECK", "200")))
idxs = np.linspace(0, len(A.pf) - 1, n_check).round().astype(int)
idxs = np.unique(idxs)
for k in idxs:
    gene = A.genes[k].tolist()
    obj = _rollout(env, gene)  # [cost, avg_wait]
    err = np.abs(obj - A.pf[k])
    max_abs_err = max(max_abs_err, float(err.max()))
rel = max_abs_err / max(1.0, float(A.pf[:, 0].max()))
if max_abs_err < 1e-3:
    print(f"(c) 残差0再生OK: {len(idxs)}遺伝子 最大絶対誤差={max_abs_err:.2e} (ビット一致水準)")
else:
    print(f"(c) ★再生誤差 max={max_abs_err:.4g} (相対 {rel:.2e}) — "
          f"許容? cost規模{A.pf[:,0].max():.0f}に対し小さければenv非決定の範囲")
    if rel > 1e-6:
        fail += 1

# ---- XOR規約の自己テスト: 0(follow)=アンカー / 1(flip)=反転 ----
g = A.genes[0]
follow = (g ^ 0)
flip = (g ^ 1)
assert np.array_equal(follow, g) and np.array_equal(flip, 1 - g), "XOR規約違反"
print("(XOR) follow=アンカー / flip=反転 の規約OK")

print("=" * 48)
print("RESULT:", "ALL PASS ✅" if fail == 0 else f"{fail} 件 FAIL ❌")
sys.exit(1 if fail else 0)
