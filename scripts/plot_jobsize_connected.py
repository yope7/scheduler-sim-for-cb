#!/usr/bin/env python3
"""全ジョブ規模(24-256)で「指令(青✕)→達成(赤)を桃線で繋いだ」eval PF を並べる。
   短い線=指令追従、長い線=無視。pf_scale_{nj}.npz を読む。
   usage: JOBS=24,32,64,128,256 OUT=jobsize_connected.png python scripts/plot_jobsize_connected.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

JOBS = [int(x) for x in os.environ.get("JOBS", "24,32,64,128,256").split(",")]
OUT = os.environ.get("OUT", "jobsize_connected.png")

ncol = 3
nrow = (len(JOBS) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax, nj in zip(axes, JOBS):
    d = np.load(f"pf_scale_{nj}.npz")
    cmd, ach, arch = d["commanded"], d["achieved"], d["archive_pf"]
    arch = arch[np.argsort(arch[:, 0])]
    F = float(np.corrcoef(cmd[:, 0], ach[:, 0])[0, 1])
    g = ach[:, 0].max() * 0.005
    dist = len(np.unique(np.round(ach[get_non_dominated_inds_minimize(ach)] / g, 0), axis=0))
    # discovered PF reference
    ax.plot(arch[:, 0], arch[:, 1], "-", color="#bbb", lw=1.1, zorder=1, label=f"discovered PF ({len(arch)})")
    # connector lines: commanded -> achieved
    for (cc, cw), (ac, aw) in zip(cmd, ach):
        ax.plot([cc, ac], [cw, aw], "-", color="#f0a0a0", lw=0.7, alpha=0.7, zorder=2)
    ax.scatter(cmd[:, 0], cmd[:, 1], s=26, marker="x", c="#1a73e8", lw=1.0, zorder=3, label="commanded (input)")
    ax.scatter(ach[:, 0], ach[:, 1], s=24, c="#d62728", edgecolor="k", lw=0.3, zorder=4, label="achieved")
    ok = F >= 0.8
    ax.set_title(f"{nj} jobs   distinct={dist}   F={F:+.2f}" + ("" if ok else "  (collapsed)"),
                 fontsize=11.5, color="black" if ok else "#b00000")
    ax.set_xlabel("Cost"); ax.set_ylabel("Avg Wait"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
for ax in axes[len(JOBS):]:
    ax.axis("off")
fig.suptitle("Command(blue x) -> Achieved(red) connectors per job size:  short line = followed, long = ignored.  more jobs -> shorter, denser",
             fontsize=12.5, y=1.0)
fig.tight_layout(); fig.savefig(OUT, dpi=118, bbox_inches="tight")
print(f"saved {OUT}")
for nj in JOBS:
    d = np.load(f"pf_scale_{nj}.npz")
    F = np.corrcoef(d["commanded"][:, 0], d["achieved"][:, 0])[0, 1]
    print(f"  {nj} jobs: F={F:+.3f}")
