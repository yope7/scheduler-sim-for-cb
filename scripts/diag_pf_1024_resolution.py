#!/usr/bin/env python3
"""command数 vs 再現PF点数 のカーブ: 律速がcommand数かconditioningかを示す。"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# archive-command sweep on amplog_b iter100 policy (this session, offline)
cmds = [20, 50, 100, 151]
pf   = [18, 34, 46, 46]
# live 2D uniform grid datapoint (372 commands -> 52 nd), and the discoverable ceiling
GRID_2D = (372, 52)
ARCHIVE_CEIL = 151  # non-dominated points exploration actually discovered

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.plot(cmds, pf, "-o", color="#d62728", ms=9, lw=2.2, label="Archive-command sweep (1D along front)")
ax.scatter([GRID_2D[0]], [GRID_2D[1]], s=130, marker="D", color="#1f77b4", zorder=5,
           label=f"Live 2D uniform grid ({GRID_2D[0]} cmd -> {GRID_2D[1]})")
ax.axhline(ARCHIVE_CEIL, color="#2ca02c", ls="--", lw=1.8,
           label=f"Discovered (archive PF) ceiling = {ARCHIVE_CEIL}")
ax.axhspan(44, 54, color="#d62728", alpha=0.08)
ax.annotate("policy saturates at ~46-52\n(conditioning ceiling)",
            xy=(151, 46), xytext=(190, 70), fontsize=11, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.annotate("CONDITIONING GAP\n(discovered but NOT reproducible):\nthe real target for 'more points'",
            xy=(120, 151), xytext=(60, 118), fontsize=10.5, color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))

ax.set_xlabel("# commands fed to policy", fontsize=12)
ax.set_ylabel("# distinct non-dominated PF points reproduced", fontsize=12)
ax.set_title("trace1024 (amplog_b iter100): PF point count is CONDITIONING-limited, not command-limited\n"
             "more commands stop helping at ~100; the policy reproduces only ~46-52 of the 151 it discovered",
             fontsize=11.5)
ax.set_xlim(0, 400); ax.set_ylim(0, 165)
ax.grid(alpha=0.3); ax.legend(loc="center right", fontsize=10)
fig.tight_layout()
out = Path("pf_1024_resolution_curve.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
