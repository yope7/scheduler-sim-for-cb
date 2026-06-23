#!/usr/bin/env python
"""NSGA-II の評価系が PCN eval (eval_b2_compare) と同じ目的値 (cost, avg_wait) を返すかを
同一ワークロード(trace512 job_seed=0)で照合する。

背景: 既存 NSGA-II の評価は bitmap env (スライドウィンドウ n_window=100, 1刻みスクロール) だったが、
trace ワークロードは処理時間が最大 43万時刻単位 ≫ window=100 で構造的に評価不能（実測: 600s でも
1エピソード終わらず）。worker を event native env（PCN eval と同一）に切り替えたので、その正しさを
ここで担保する。

照合:
 1) _evaluate_worker(event backend, calc_objective_values 経路) ≡ 報酬累積(_rp_one と同型)
 2) 同一 env インスタンスの reset() 再利用で2回評価が一致（persistent pool の安全性）
 3) rp 掃引 npz の端点 (p=0 → all-onprem, p=1 → all-cloud) と一致（既報 PF 図の座標系と同一）
 4) 1評価の実行時間（NSGA-II の pop/gen 規模決めの材料）

usage: PYTHONPATH=. .venv/bin/python -u scripts/verify_nsga2_eval_equiv.py
"""
import os
import time

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

import numpy as np

from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.nsga2_agent import _build_env_params, _evaluate_worker, _make_env, _rollout

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_512_pcn.yml")
NJ = int(os.environ.get("NJ", "512"))
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
REF_NPZ = os.environ.get("REF_NPZ", "results/eval_pf/truepf_trace512_schedlr512_1_s0.npz")

config = load_config(CFG)
base_env = create_eval_env(config, job_seed=JOB_SEED, n_jobs=NJ)
env_params = _build_env_params(base_env)
print(f"CFG={CFG} NJ={NJ} job_seed={JOB_SEED} worker_backend={env_params['env_backend']}")


def eval_reward_accum(env, chrom):
    """PCN eval (eval_b2_compare._rp_one) と同型: step 報酬の累積から (cost, avg_wait)。"""
    env.reset()
    tw = tc = 0.0
    n = 0
    done = False
    idx = 0
    while not done:
        a = int(chrom[idx]) if idx < len(chrom) else 0
        r = env.step(a)
        tw += -float(r[1][0])
        tc += -float(r[1][1])
        if r[2]:
            idx += 1
            n += 1
        done = r[-1]
    return np.array([tc, tw / max(1, n)])


cases = {
    "all-onprem(0)": np.zeros(NJ, dtype=int),
    "all-cloud(1)": np.ones(NJ, dtype=int),
    "bernoulli0.5": (np.random.default_rng(42).random(NJ) < 0.5).astype(int),
}

shared_env = _make_env(env_params)  # reset 再利用の検証用に1個だけ作る
n_fail = 0
results = {}
print(f"\n{'case':>16} {'worker [cost, avg_wait]':>32} {'sec':>6}  checks")
for name, chrom in cases.items():
    t0 = time.time()
    ob_w = _evaluate_worker((list(map(int, chrom)), env_params))
    tw_ = time.time() - t0
    ob_a = eval_reward_accum(create_eval_env(config, job_seed=JOB_SEED, n_jobs=NJ), chrom)
    ob_r1 = _rollout(shared_env, list(map(int, chrom)))
    ob_r2 = _rollout(shared_env, list(map(int, chrom)))  # 同一 env 2回目 (reset 再利用)
    eq_accum = np.allclose(ob_w, ob_a, rtol=1e-9, atol=1e-9)
    eq_reset = np.allclose(ob_r1, ob_r2, rtol=0, atol=0) and np.allclose(ob_w, ob_r1, rtol=1e-9, atol=1e-9)
    if not (eq_accum and eq_reset):
        n_fail += 1
    results[name] = ob_w
    print(f"{name:>16} [{ob_w[0]:>14.1f}, {ob_w[1]:>12.4f}] {tw_:>6.2f}  "
          f"報酬累積一致={'OK' if eq_accum else 'NG'} reset再利用一致={'OK' if eq_reset else 'NG'}")

if os.path.exists(REF_NPZ):
    d = np.load(REF_NPZ)
    rp = d["rp_0"]
    print(f"\nrp 掃引 npz ({os.path.basename(REF_NPZ)}) 端点との照合:")
    for name, idx in (("all-onprem(0)", 0), ("all-cloud(1)", -1)):
        ob = results[name]
        ref = rp[idx]
        rel = np.abs(ob - ref) / np.maximum(np.abs(ref), 1e-9)
        ok = np.all(rel < 1e-6)
        if not ok:
            n_fail += 1
        print(f"  {name:>16}: npz=[{ref[0]:.1f}, {ref[1]:.4f}] worker=[{ob[0]:.1f}, {ob[1]:.4f}] "
              f"-> {'OK' if ok else 'MISMATCH'}")
else:
    print(f"\n(ref npz {REF_NPZ} なし: rp 照合スキップ)")

print(f"\n=> {'ALL OK: NSGA-II 評価系は PCN eval と同一座標系' if n_fail == 0 else f'{n_fail} 件不一致あり'}")
