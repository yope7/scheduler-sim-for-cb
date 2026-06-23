#!/usr/bin/env python3
"""trace1024 の discovered Pareto front ＋ achieved(探索点) をリッチに再現・逐次保存。
   高速(クラウド寄り/低閾値)→低速(混雑/高閾値)の順で回し、各バッチ後に png/npz を更新するので
   途中killでも常に最新のリッチ図が残る。端点は calibration 値も併せて投入。"""
import os, time
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import get_non_dominated_inds_minimize

CFG = "experiments/distributed_pcn/job_trace_1024_pcn.yml"
N_JOBS = 1024
config = load_config(CFG)
env = create_eval_env(config, job_seed=0, n_jobs=N_JOBS)
EXTREMES = [(0.0, 1637654.1, "extreme"), (1826606492.0, 195241.3, "extreme")]
achieved = list(EXTREMES)
t0 = time.perf_counter()

def run_episode(action_fn):
    env.reset(); done = False
    while not done:
        j = int(env.index_next_job)
        if j >= len(env.jobs): break
        a = action_fn(env, j)
        _, _, _, _, done = env.step(a)
    c, _, w = env.calc_objective_values()
    return float(c), float(w)

def th_action(e, j, th):
    raw = e.jobs[j]; job = e._to_queue_job(raw); arr = int(raw[0])
    _, onp = e._find_event_allocation(job, False, arr)
    return 1 if (int(onp) - arr) >= th else 0

def save():
    A = np.array([[c, w] for c, w, _ in achieved], dtype=np.float64)
    kinds = [k for _, _, k in achieved]
    nd = get_non_dominated_inds_minimize(A)
    pf = A[nd]; pf = pf[np.argsort(pf[:, 0])]
    np.savez("pf_1024_discovered.npz", achieved=A, pareto_front=pf, kinds=np.array(kinds))
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    def sub(kind):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        return A[idx] if idx else np.empty((0, 2))
    rnd, heu, ext = sub("random"), sub("heuristic"), sub("extreme")
    if len(rnd): ax.scatter(rnd[:, 0], rnd[:, 1], s=24, c="#9aa0a6", alpha=0.55, label=f"achieved: random sweep ({len(rnd)})")
    if len(heu): ax.scatter(heu[:, 0], heu[:, 1], s=55, marker="x", c="#1f77b4", lw=1.6, label=f"achieved: heuristic seeding ({len(heu)})")
    if len(ext): ax.scatter(ext[:, 0], ext[:, 1], s=160, marker="*", c="#2ca02c", zorder=6, label="extremes: cost=0 / cost=max")
    ax.plot(pf[:, 0], pf[:, 1], "-o", color="#d62728", ms=5, lw=2, label=f"discovered Pareto front ({len(pf)})", zorder=5)
    ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title(f"trace1024 — discovered Pareto front + achieved (explored) points  [{len(A)} pts]\n"
                 "random sweep + reactive-overflow seeding; event-native (no bitmap)")
    fig.tight_layout(); fig.savefig("pf_1024_discovered.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  [save] achieved={len(A)} PF={len(pf)} cost[{pf[:,0].min():.2e},{pf[:,0].max():.2e}] "
          f"wait[{pf[:,1].min():.2e},{pf[:,1].max():.2e}] ({time.perf_counter()-t0:.0f}s)", flush=True)

# 高速→低速: クラウド寄り(高prob)・低閾値を先に、混雑(低prob)・高閾値を後に
PROBS = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.0]
THS   = [0,500,1000,5000,10000,50000,100000,300000,500000,1000000,2000000]
for p in PROBS:
    seeds = 4 if p >= 0.3 else 2  # 低prob(混雑=低速)は seed を絞る
    for s in range(seeds):
        rng = np.random.default_rng(s * 311 + int(p * 1000))
        c, w = run_episode(lambda e, j, p=p, rng=rng: 1 if rng.random() < p else 0)
        achieved.append((c, w, "random"))
    print(f"[random] p={p} ({time.perf_counter()-t0:.0f}s)", flush=True)
    save()
for th in THS:
    c, w = run_episode(lambda e, j, th=th: th_action(e, j, th))
    achieved.append((c, w, "heuristic"))
    print(f"[heuristic] th={th} -> cost={c:.2e} wait={w:.2e}", flush=True)
    save()
print("DONE", flush=True)
