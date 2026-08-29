"""5万ジョブ(weekBフル週・正容量cap48000)を GPU 工場カーネル(run_fused_defer_ev)で
回すスモーク。目的: (1) 1チャンクの実時間 (2) ovf(溢れ)発火の実態 (3) 5万を GPU で
回す時の推奨バッファ(KPICK/NAMB/POOL)の決定。

usage:
  CUDA_VISIBLE_DEVICES=0 B=64 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/bench_50k_smoke.py
  env: B(バッチ数, 既定64) / KPICK / NAMB / TSCAN_FRAC(既定1.0)
"""
import os
import sys
import time

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ["SCHEDULER_OBS_URGENCY"] = "1"

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from scripts.pcn_replay_snapshot import load_config, create_eval_env  # noqa: E402

CFG = "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml"
B = int(os.environ.get("B", "64"))

cfg = load_config(CFG)
env = create_eval_env(cfg, job_seed=0, n_jobs=50000)
env.reset()
jobs = np.asarray(env.jobs, dtype=np.float64).copy()
print(f"jobs shape={jobs.shape} n_on=48000 n_cl=192000 B={B}", flush=True)

from factory_defer_ev import run_fused_defer_ev  # noqa: E402

kw = {}
if os.environ.get("KPICK"):
    kw["k_pick"] = int(os.environ["KPICK"])
if os.environ.get("NAMB"):
    kw["n_amb"] = int(os.environ["NAMB"])
    kw["n_amb_cl"] = int(os.environ["NAMB"])

probs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
probs_full = [probs[i % len(probs)] for i in range(B)]

t0 = time.time()
res = run_fused_defer_ev(
    jobs, probs=probs_full, seed=0, episode_id0=0,
    n_on=48000, n_cl=192000, n_window=100, **kw)
dt = time.time() - t0
st = res.get("stats") or {}
print(f"[SMOKE] B={B} time={dt:.1f}s ({dt/B*1000:.0f}ms/ep) ovf={st.get('ovf')}", flush=True)
print(f"[SMOKE] stats keys: e_on={st.get('e_on')} pool={st.get('pool')} "
      f"n_amb={st.get('n_amb')} k_pick={st.get('k_pick')} all_done={st.get('all_done')}",
      flush=True)
# 目的関数の妥当性ざっくり確認(全オンプレのcost=0、全クラウドのcost=総額)
obj = res.get("objectives")
if obj is not None:
    obj = np.asarray(obj)
    print(f"[SMOKE] objectives sample: p=0 -> {obj[0]}, p=1.0 -> {obj[6]}", flush=True)
