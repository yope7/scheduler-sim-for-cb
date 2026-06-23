#!/usr/bin/env python3
"""32〜256の各ジョブ数について eval PF を2×2で並べ、パラメータ関係を一覧化。
   各 pf_scale_{nj}.npz (commanded, achieved, archive_pf, explored) を読む。
   usage: JOBS=32,64,128,256 OUT=jobsize_summary.png python scripts/plot_jobsize_summary.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

JOBS = [int(x) for x in os.environ.get("JOBS", "32,64,128,256").split(",")]
OUT = os.environ.get("OUT", "jobsize_summary.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
rows = []
for ax, nj in zip(axes, JOBS):
    d = np.load(f"pf_scale_{nj}.npz")
    cmd, ach, arch = d["commanded"], d["achieved"], d["archive_pf"]
    explo = d["explored"] if "explored" in d else np.empty((0, 2))
    arch = arch[np.argsort(arch[:, 0])]
    nd = get_non_dominated_inds_minimize(ach); apf = ach[nd]; apf = apf[np.argsort(apf[:, 0])]
    F = float(np.corrcoef(cmd[:, 0], ach[:, 0])[0, 1])
    g = ach[:, 0].max() * 0.005
    dist = len(np.unique(np.round(apf / g, 0), axis=0))
    span = (ach[:, 0].max() - ach[:, 0].min()) / max(1e-9, cmd[:, 0].ptp())
    order = np.argsort(arch[:, 0])
    excess = np.clip(ach[:, 1] - np.interp(ach[:, 0], arch[order, 0], arch[order, 1]), 0, None) / max(1e-9, arch[:, 1].ptp())
    healthy = F >= 0.8
    if len(explo):
        ax.scatter(explo[::15, 0], explo[::15, 1], s=5, c="#dddddd", alpha=0.4, zorder=0)
    ax.plot(arch[:, 0], arch[:, 1], "-", color="#888", lw=1.3, zorder=2, label=f"discovered PF ({len(arch)})")
    ax.scatter(cmd[:, 0], cmd[:, 1], s=20, marker="x", c="#1a73e8", zorder=3, label="commanded")
    ax.scatter(ach[:, 0], ach[:, 1], s=28, c="#d62728" if healthy else "#7a0000", edgecolor="k", lw=0.3, zorder=4, label="achieved")
    ax.set_title(f"{nj} jobs:  distinct={dist}  F={F:+.2f}" + ("" if healthy else "  COLLAPSED"),
                 fontsize=12, color="black" if healthy else "red")
    ax.set_xlabel("Cost"); ax.set_ylabel("Avg Wait"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    rows.append((nj, len(arch), dist, F, span, float(np.mean(excess)), ach[:, 0].max()))
fig.suptitle("Eval PF by job count (32-256): more jobs -> lower variance -> denser, more reliable PF", fontsize=13, y=1.0)
fig.tight_layout(); fig.savefig(OUT, dpi=115, bbox_inches="tight")
print(f"saved {OUT}")
print(f"{'jobs':>5} {'discPF':>7} {'distinct':>8} {'F':>6} {'span':>5} {'裂開':>5} {'cost_max':>9}")
for nj, dp, di, F, sp, ex, cm in rows:
    print(f"{nj:>5} {dp:>7} {di:>8} {F:>+6.2f} {sp:>5.2f} {ex:>5.3f} {cm:>9.0f}")
