#!/usr/bin/env python3
"""合成ジョブ run の discovered PF + achieved(探索点) を snapshot から図示・保存。
   usage: DIAG_SNAP=<snapshot> DIAG_NJOBS=24 DIAG_OUT=pf_synth24.png python scripts/diag_synth_discovered.py"""
import os
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import (load_learner_replay_snapshot,
    episode_objectives_from_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import get_non_dominated_inds_minimize

SNAP = os.environ["DIAG_SNAP"]
NJ = int(os.environ.get("DIAG_NJOBS", "24"))
OUT = os.environ.get("DIAG_OUT", "pf_synth.png")
TITLE = os.environ.get("DIAG_TITLE", f"synthetic {NJ}-job — discovered Pareto front + achieved")

snap = load_learner_replay_snapshot(SNAP)
explo = episode_objectives_from_snapshot(snap, NJ)
arch = archive_pf_from_snapshot(snap, NJ)
nd = get_non_dominated_inds_minimize(arch); pf = arch[nd] if len(nd) else arch
pf = pf[np.argsort(pf[:, 0])]
np.savez(OUT.replace(".png", ".npz"), achieved=explo, pareto_front=pf)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.scatter(explo[:, 0], explo[:, 1], s=10, c="#7fd6e0", alpha=0.5, label=f"achieved: exploration ({len(explo)})")
ax.plot(pf[:, 0], pf[:, 1], "-D", color="#d62728", ms=4, lw=1.6, label=f"discovered Pareto front ({len(pf)})", zorder=5)
ax.scatter([pf[0, 0]], [pf[0, 1]], s=150, marker="*", c="#2ca02c", zorder=6, label="cost=0 (all on-prem)")
ax.scatter([pf[-1, 0]], [pf[-1, 1]], s=150, marker="*", c="#9467bd", zorder=6, label="cost=max (heavy cloud)")
ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title(TITLE)
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved {OUT}  explo={len(explo)} PF={len(pf)} cost[{pf[:,0].min():.0f},{pf[:,0].max():.0f}] wait[{pf[:,1].min():.1f},{pf[:,1].max():.1f}]")
