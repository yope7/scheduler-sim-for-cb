#!/usr/bin/env python
"""/goal 実験: baseline(LW10/KNEE8) vs anchor(LW25/KNEE18) の 512 trace 5本比較。
左=baseline PF束, 中=anchor PF束, 右=HV分布(箱+点)。共通真PF基準でHV正規化。
出力: docs/figures/pf_512_anchor_compare.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

OUT = os.environ.get("OUT", "docs/figures/pf_512_anchor_compare.png")


def load(tag):
    d = np.load(tag); return d["greedy_0"], d["rp_0"]


base = {i: load(f"truepf_trace512_repro{i}_s0.npz") for i in range(1, 6)}
anch = {i: load(f"truepf_trace512_anchor{i}_s0.npz") for i in range(1, 6)}
allg = []
cmax = 0
for D in (base, anch):
    for g, r in D.values():
        allg.append(np.vstack([g, r])); cmax = max(cmax, r[:, 0].max())
allpts = np.vstack(allg)
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf = allpts[get_non_dominated_inds_minimize(allpts)]
pf = pf[np.argsort(pf[:, 0])]


def hv(pts):
    nd = pts[get_non_dominated_inds_minimize(pts)]
    fr = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not fr:
        return 0.0
    h = 0.0; p = ref[0]
    for c, w in sorted(fr, key=lambda x: -x[0]):
        h += (p - c) * (ref[1] - w); p = c
    return h


hvt = hv(allpts[get_non_dominated_inds_minimize(allpts)])
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.8], wspace=0.26))
for axi, (name, D, col) in zip(ax[:2], [("baseline  (LW10/KNEE8)", base, plt.cm.Blues),
                                        ("anchor  (LW25/KNEE18)", anch, plt.cm.Oranges)]):
    axi.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#188038", lw=2.3, label="union true PF", zorder=2)
    hs = []
    for i, (g, r) in D.items():
        h = hv(g) / hvt; hs.append(h)
        o = np.argsort(g[:, 0])
        axi.plot(g[o, 0] / cmax, g[o, 1], "-o", c=col(0.35 + 0.5 * i / 5), lw=1, ms=3.5, alpha=.85,
                 label=f"run{i} (HV {h:.0%})")
    axi.axvspan(0, 0.10, color="#d7ecd9", alpha=.6, zorder=0)
    axi.set_xlim(-0.03, 1.03)
    axi.set_xlabel("Cost (fraction of all-cloud; 0=cheapest)")
    axi.set_ylabel("Average wait time")
    axi.set_title(f"{name}\nmean {np.mean(hs):.0%} ± {np.std(hs):.0%}", fontsize=11, fontweight="bold")
    axi.legend(fontsize=7.6, loc="upper right")
    axi.grid(alpha=.3)

hb = np.array([hv(g) / hvt for g, _ in base.values()])
ha = np.array([hv(g) / hvt for g, _ in anch.values()])
axS = ax[2]
for x, hs, c in [(0, hb, "#1a73e8"), (1, ha, "#e8710a")]:
    axS.scatter([x] * len(hs), hs * 100, s=70, c=c, alpha=.8, zorder=3, edgecolor="#333")
    axS.plot([x - .18, x + .18], [hs.mean() * 100] * 2, c=c, lw=3)
    axS.add_patch(plt.Rectangle((x - .14, (hs.mean() - hs.std()) * 100), .28, 2 * hs.std() * 100,
                                facecolor=c, alpha=.15, edgecolor=c))
axS.set_xticks([0, 1]); axS.set_xticklabels(["baseline", "anchor"], fontsize=10)
axS.set_ylabel("greedy HV (% of true PF)")
axS.set_ylim(0, 105)
axS.set_title(f"variance ↓ (std {hb.std():.0%}→{ha.std():.0%})\nbut mean/ceiling did NOT rise",
              fontsize=11, fontweight="bold")
axS.grid(axis="y", alpha=.3)
fig.suptitle("512 trace /goal experiment: up-weighting efficient demos homogenizes to mediocrity "
             "(variance↓, ceiling↓) — wrong lever", fontsize=12, y=1.02, fontweight="bold")
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("saved", OUT)
print(f"baseline mean={hb.mean():.0%} std={hb.std():.0%} max={hb.max():.0%} eff={int((hb>=.6).sum())}/5")
print(f"anchor   mean={ha.mean():.0%} std={ha.std():.0%} max={ha.max():.0%} eff={int((ha>=.6).sum())}/5")
