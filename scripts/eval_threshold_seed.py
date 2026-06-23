#!/usr/bin/env python
"""Phase1 の WaitTimeThreshold 種ヒューリスティックを 512 trace で直接評価。
各閾値Tで「先頭ジョブのオンプレ予測待ち >= T ならクラウド(1)、未満ならオンプレ(0)」を1エピソード走らせ
(cost, avg_wait) を記録 → repro/anchor の union真PF に重ねて「種が効率の膝に届くか」を見る。
出力: docs/figures/pf_512_heuristic_seed.png, /tmp/seed_pts.npz
"""
import os
import glob
import numpy as np
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import get_non_dominated_inds_minimize

NJ = int(os.environ.get("NJ", "512"))
CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_512_pcn.yml")
THRS = [float(x) for x in os.environ.get(
    "THRS", "0,25,50,100,150,250,400,600,1000,2000,5000,20000,999999").split(",")]

env = create_eval_env(load_config(CFG), job_seed=0, n_jobs=NJ)


def predict_front_wait(e):
    try:
        j = int(getattr(e, "index_next_job", 0))
        jobs = getattr(e, "jobs", None)
        if jobs is None or j >= len(jobs):
            return 0.0
        raw = jobs[j]
        job = e._to_queue_job(raw) if hasattr(e, "_to_queue_job") else raw
        arr = int(raw[0])
        _, onp = e._find_event_allocation(job, False, arr)
        return float(max(0, int(onp) - arr))
    except Exception:
        return 0.0


pts = []
for T in THRS:
    env.reset(); tw = tc = 0.0; n = 0; done = False; st = 0
    while not done and st < NJ + 5:
        pw = predict_front_wait(env)
        a = 1 if pw >= T else 0
        r = env.step(a)
        tw += -float(r[1][0]); tc += -float(r[1][1]); n += 1 if r[2] else 0
        done = r[-1]; st += 1
    pts.append([tc, tw / max(1, n)])
seed = np.array(pts)
np.savez("/tmp/seed_pts.npz", seed=seed, thrs=np.array(THRS))

fs = glob.glob("truepf_trace512_repro*_s0.npz") + glob.glob("truepf_trace512_anchor*_s0.npz")
allp = []
cmax = 0
best = None
for f in fs:
    d = np.load(f); allp.append(np.vstack([d["greedy_0"], d["rp_0"]])); cmax = max(cmax, d["rp_0"][:, 0].max())
    if "repro4" in f:
        best = d["greedy_0"]
allp = np.vstack(allp)
pf = allp[get_non_dominated_inds_minimize(allp)]; pf = pf[np.argsort(pf[:, 0])]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#188038", lw=2.4, label="union true PF", zorder=2)
if best is not None:
    o = np.argsort(best[:, 0])
    ax.plot(best[o, 0] / cmax, best[o, 1], "--", c="#1a73e8", lw=1.4, alpha=.8, label="rep4 (best run, HV96%)")
so = np.argsort(seed[:, 0])
ax.plot(seed[so, 0] / cmax, seed[so, 1], "-o", c="#d93025", lw=1.6, ms=6, label="Phase1 threshold-heuristic seed", zorder=4)
for (c, w), T in zip(seed, THRS):
    ax.annotate(f"T={T:g}", (c / cmax, w), fontsize=6.5, color="#d93025", xytext=(2, 3), textcoords="offset points")
ax.axvspan(0, 0.12, color="#d7ecd9", alpha=.5, zorder=0)
ax.set_xlim(-0.03, 1.03)
ax.set_xlabel("Cost (fraction of all-cloud; 0=cheapest)")
ax.set_ylabel("Average wait time")
ax.set_title(f"Does the Phase1 seed reach the efficient knee? - trace {NJ}", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=.3)
fig.savefig("docs/figures/pf_512_heuristic_seed.png", dpi=125, bbox_inches="tight")
print("saved docs/figures/pf_512_heuristic_seed.png")
nd = seed[get_non_dominated_inds_minimize(seed)]; nd = nd[np.argsort(nd[:, 0])]
print(f"\nseed front (nd {len(nd)}/{len(seed)}):")
print(f"{'T':>9} {'cost_frac':>10} {'wait':>10}")
for (c, w), T in sorted(zip(seed.tolist(), THRS), key=lambda x: x[0][0]):
    print(f"{T:>9g} {c/cmax:>10.3f} {w:>10.0f}")


def wat(front, q):
    f = front[get_non_dominated_inds_minimize(front)]; f = f[np.argsort(f[:, 0])]
    return np.interp(q * cmax, f[:, 0], f[:, 1])


if best is not None:
    print("\nefficiency wait@q: seed vs rep4(best)")
    for q in [0.1, 0.25, 0.5]:
        print(f"  q={q}: seed={wat(seed,q):.0f}  rep4={wat(best,q):.0f}")
