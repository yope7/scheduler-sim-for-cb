#!/usr/bin/env python3
"""WaitTimeThreshold（reactive cloud overflow）を event-native env で閾値sweepし、
   生成されるフロントを policy の到達フロント / random-Bernoulli と比較する。
   = 「種まきが、方策の届かない低wait領域に到達するか」の検証 + 閾値選定。"""
import os, time
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import create_eval_env, load_config

CFG = "experiments/distributed_pcn/job_trace_1024_pcn.yml"
N_JOBS = 1024
config = load_config(CFG)
env = create_eval_env(config, job_seed=0, n_jobs=N_JOBS)

def run_threshold(th):
    env.reset(); done = False
    while not done:
        j = int(env.index_next_job)
        if j >= len(env.jobs): break
        raw = env.jobs[j]; job = env._to_queue_job(raw); arr = int(raw[0])
        _, onp = env._find_event_allocation(job, False, arr)
        action = 1 if (int(onp) - arr) >= th else 0
        _, _, _, _, done = env.step(action)
    env.finalize_window_history(build_maps=False) if hasattr(env, "finalize_window_history") else None
    cost, _, avg_wt = env.calc_objective_values()
    return float(cost), float(avg_wt)

def run_bernoulli(p, seed=0):
    rng = np.random.default_rng(seed)
    env.reset(); done = False
    while not done:
        if int(env.index_next_job) >= len(env.jobs): break
        action = 1 if rng.random() < p else 0
        _, _, _, _, done = env.step(action)
    env.finalize_window_history(build_maps=False) if hasattr(env, "finalize_window_history") else None
    cost, _, avg_wt = env.calc_objective_values()
    return float(cost), float(avg_wt)

t0 = time.perf_counter()
THS = [0, 1, 10, 50, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
heur = []
print("=== WaitTimeThreshold heuristic front ===")
for th in THS:
    c, w = run_threshold(th)
    heur.append((th, c, w)); print(f"  th={th:>8} -> cost={c:.3e} avg_wait={w:.3e}")
PS = [0.1, 0.25, 0.5, 0.75, 0.9]
bern = []
print("=== random Bernoulli(p) baseline ===")
for p in PS:
    c, w = run_bernoulli(p)
    bern.append((p, c, w)); print(f"  p={p:>4} -> cost={c:.3e} avg_wait={w:.3e}")
print(f"[time] {time.perf_counter()-t0:.1f}s")

H = np.array([[c, w] for _, c, w in heur]); B = np.array([[c, w] for _, c, w in bern])
fig, ax = plt.subplots(figsize=(9.5, 6.5))
ax.plot(H[:, 0], H[:, 1], "-o", color="#2ca02c", ms=7, lw=2, label="WaitTimeThreshold heuristic (reactive)")
ax.scatter(B[:, 0], B[:, 1], s=70, marker="x", color="#9467bd", label="random Bernoulli(p) (= current Phase-1 seed)")
try:
    d = np.load("pf_1024_conditioning_response.npz")
    ach = d["achieved"]
    ax.scatter(ach[:, 0], ach[:, 1], s=22, color="#d62728", alpha=0.7, label="policy achieved (amplog_b iter100)")
except Exception:
    pass
ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title("Does reactive-overflow heuristic reach the low-wait region the policy/random miss?\n(lower-left = better; trace1024)")
fig.tight_layout(); fig.savefig("pf_1024_heuristic_front.png", dpi=115, bbox_inches="tight")
print("saved pf_1024_heuristic_front.png")
