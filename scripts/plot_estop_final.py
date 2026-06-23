#!/usr/bin/env python
"""実用版early-stop の最終判定: 4群を同一土俵で比較。
  baseline   : repro 各runの iter100 (固定100iter学習の着地)
  oracle-ES  : repro 各runの HVピークiter (後知恵=early-stopの上限)
  practical-ES: es 各runを DISTRIBUTED_PCN_EARLYSTOP=1 で新規学習し best_model.pth を採用 (実用版)
共通真PF/cmax/ref = 全群 union。実用版が oracle に届くか(平均84%/std10%/効率3/5)を見る。
出力: docs/figures/pf_512_estop_final.png
usage: PYTHONPATH=. .venv/bin/python scripts/plot_estop_final.py
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

SCALE = os.environ.get("SCALE", "512")


def hv2d(pts, ref):
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0; prev = ref[0]
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev - c) * (ref[1] - w); prev = c
    return hv


def load_group(tag):
    runs = {}
    for fn in sorted(glob.glob(f"results/eval_pf/truepf_trace{SCALE}_{tag}[0-9]_s0.npz"),
                     key=lambda s: int(re.search(rf"{tag}(\d+)_s0", s).group(1))):
        i = int(re.search(rf"{tag}(\d+)_s0", fn).group(1))
        d = np.load(fn)
        runs[i] = (d["greedy_0"], d["rp_0"])
    return runs


G = {"baseline": load_group("repro"), "oracle-ES": load_group("estop"), "practical-ES": load_group("es")}
for k, v in G.items():
    assert len(v) == 5, f"{k}: {len(v)} runs (expected 5)"
allpts = np.vstack([np.vstack([g, r]) for grp in G.values() for (g, r) in grp.values()])
cmax = max(r[:, 0].max() for grp in G.values() for (_, r) in grp.values())
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf = allpts[get_non_dominated_inds_minimize(allpts)]
pf = pf[np.argsort(pf[:, 0])]
hvt = hv2d(pf, ref)


def wat(g, q):
    f = g[get_non_dominated_inds_minimize(g)]; f = f[np.argsort(f[:, 0])]
    return float(np.interp(q * cmax, f[:, 0], f[:, 1]))


stats = {}
for name, grp in G.items():
    hv = np.array([hv2d(g, ref) / hvt for (g, _) in grp.values()])
    w25 = np.array([wat(g, 0.25) for (g, _) in grp.values()])
    stats[name] = dict(hv=hv, w25=w25, eff=int((hv >= 0.8).sum()))
    print(f"\n[{name}] HV per run: " + " ".join(f"{x:.0%}" for x in hv) +
          f"  | mean={hv.mean():.0%} std={hv.std():.0%} 効率run(>=80%)={int((hv>=0.8).sum())}/5")
    print(f"  wait@q0.25 median={np.median(w25):.0f}")

COL = {"baseline": "#c0c4cc", "oracle-ES": "#9334e6", "practical-ES": "#1a73e8"}
fig = plt.figure(figsize=(16, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.05, 0.95], wspace=0.28)

axL = fig.add_subplot(gs[0, 0])
axL.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#188038", lw=2.6, label="union true PF", zorder=4)
for name in ["baseline", "practical-ES"]:
    for k, (i, (g, _)) in enumerate(G[name].items()):
        o = np.argsort(g[:, 0])
        axL.plot(g[o, 0] / cmax, g[o, 1], "-o" if name == "practical-ES" else "-",
                 c=COL[name], lw=1.3 if name == "practical-ES" else 1.0, ms=3,
                 alpha=.85 if name == "practical-ES" else .55,
                 zorder=3 if name == "practical-ES" else 1,
                 label=name if k == 0 else None)
axL.axvspan(0, 0.10, color="#d7ecd9", alpha=.6, zorder=0)
axL.set_xlim(-0.03, 1.03)
axL.set_xlabel("Cost (fraction of all-cloud; 0=cheapest)")
axL.set_ylabel("Average wait time")
axL.set_title(f"trace {SCALE}: baseline-final(grey) vs practical early-stop(blue)", fontsize=11.5, fontweight="bold")
axL.legend(fontsize=9, loc="upper right"); axL.grid(alpha=.3)

axM = fig.add_subplot(gs[0, 1])
x = np.arange(5)
for j, name in enumerate(["baseline", "oracle-ES", "practical-ES"]):
    hv = sorted(stats[name]["hv"], reverse=True)
    axM.bar(x + (j - 1) * 0.27, hv, width=0.26, color=COL[name], edgecolor="#333",
            label=f"{name} (mean {stats[name]['hv'].mean():.0%}, eff {stats[name]['eff']}/5)")
axM.axhline(0.8, color="#d93025", ls=":", lw=1.4)
axM.set_xticks(x); axM.set_xticklabels([f"#{i+1}" for i in x])
axM.set_ylim(0, 1.09); axM.set_ylabel("greedy HV (frac of true PF)")
axM.set_title("HV per run (sorted)", fontsize=10.5, fontweight="bold")
axM.legend(fontsize=8.2, loc="lower left"); axM.grid(axis="y", alpha=.3)

axR = fig.add_subplot(gs[0, 2])
names = ["baseline", "oracle-ES", "practical-ES"]
means = [stats[n]["hv"].mean() * 100 for n in names]
stds = [stats[n]["hv"].std() * 100 for n in names]
axR.bar(range(3), means, yerr=stds, color=[COL[n] for n in names], edgecolor="#333", capsize=5)
for i, m in enumerate(means):
    axR.text(i, m + 2, f"{m:.0f}%", ha="center", fontsize=10, fontweight="bold")
axR.set_xticks(range(3)); axR.set_xticklabels(names, rotation=15, ha="right", fontsize=8.5)
axR.set_ylabel("mean HV (% true PF)"); axR.set_ylim(0, 100)
axR.set_title("mean HV", fontsize=10.5, fontweight="bold")
axR.grid(axis="y", alpha=.3)

fig.suptitle("512 trace — does the PRACTICAL early-stop (auto best_model) match the oracle?",
             fontsize=12.5, y=1.02, fontweight="bold")
fig.savefig("docs/figures/pf_512_estop_final.png", dpi=125, bbox_inches="tight")
print("\nsaved docs/figures/pf_512_estop_final.png")
