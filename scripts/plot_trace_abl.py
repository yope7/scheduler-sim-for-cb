#!/usr/bin/env python
"""trace 小規模アブレーション(off/p2/norm05/norm10)の greedy front を共通真PF上に重ねる。
存在する truepf_trace{SCALE}_abl_{COND}_s0.npz を自動検出。HV(真PF比)を凡例に表示。
出力: docs/figures/pf_trace_abl.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

CONDS = [("off", "#888888", "o", "off (P2 off, norm off)"),
         ("p2", "#ff9800", "s", "p2 (P2 on, norm off)"),
         ("norm05", "#1a73e8", "D", "norm α=0.5 (P2 on)"),
         ("norm10", "#d62728", "^", "norm α=1.0 (P2 on)")]
SCALES = [16, 32, 64, 128, 256, 512, 1024]


def load(fn):
    d = np.load(fn)
    return d["greedy_0"], d["samp_0"], d["rp_0"]


def hv2d(pts, ref):
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0
    prev_cost = ref[0]
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev_cost - c) * (ref[1] - w)
        prev_cost = c
    return hv


present = []
for nj in SCALES:
    avail = [(c, col, mk, lab) for (c, col, mk, lab) in CONDS
             if os.path.exists(f"truepf_trace{nj}_abl_{c}_s0.npz")]
    if len(avail) >= 2:
        present.append((nj, avail))
if not present:
    raise SystemExit("no truepf_trace*_abl_*_s0.npz found")

n = len(present)
fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.6), squeeze=False)
axes = axes[0]
summary = {}
for ax, (nj, avail) in zip(axes, present):
    data = {c: load(f"truepf_trace{nj}_abl_{c}_s0.npz") for (c, _, _, _) in avail}
    allpts = np.vstack([np.vstack([g, s, r]) for (g, s, r) in data.values()])
    cmax = max(r[:, 0].max() for (_, _, r) in data.values())
    nd = get_non_dominated_inds_minimize(allpts)
    pf = allpts[nd]
    pf = pf[np.argsort(pf[:, 0])]
    ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
    hvt = hv2d(pf, ref)
    ax.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#2ca02c", lw=2.2,
            label="true PF (best achievable)", zorder=2)
    summary[nj] = {}
    for (c, col, mk, lab) in avail:
        g = data[c][0]
        hv = hv2d(g, ref) / hvt if hvt > 0 else 0.0
        summary[nj][c] = hv
        order = np.argsort(g[:, 0])
        ax.plot(g[order, 0] / cmax, g[order, 1], "-", c=col, lw=0.8, alpha=0.5, zorder=3)
        ax.scatter(g[:, 0] / cmax, g[:, 1], s=30, c=col, marker=mk,
                   label=f"{lab}  HV {hv:.0%}", zorder=4, edgecolor="k", lw=.25)
    ax.axvspan(0, 0.12, color="#cfe8cf", alpha=.5, zorder=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_title(f"trace  n_jobs = {nj}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cost  (fraction of all-cloud;  0 = all on-prem / cheapest)")
    ax.set_ylabel("Average wait time")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=.95)
    ax.grid(alpha=.3)
fig.suptitle("Trace small-scale ablation: decompose the strategy.  off→p2 = Phase2 effect;  p2→norm = count-normalization effect.  "
             "HV = greedy hypervolume as % of the true PF (higher=better).",
             fontsize=10, y=1.005)
fig.tight_layout()
out = "docs/figures/pf_trace_abl.png"
fig.savefig(out, dpi=125, bbox_inches="tight")
print("saved", out)
print("\nHV (% of true PF) by scale x condition:")
hdr = "scale  " + "  ".join(f"{c:>7s}" for (c, _, _, _) in CONDS)
print(hdr)
for nj in sorted(summary):
    row = f"{nj:5d}  " + "  ".join(f"{summary[nj].get(c, float('nan')):7.0%}" if c in summary[nj] else "    ---" for (c, _, _, _) in CONDS)
    print(row)
