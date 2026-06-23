#!/usr/bin/env python3
"""現状PF診断図: trace1024 (amplog_b, iter100) の
   探索クラウド / archive PF / 主eval PF を重ね、conditioning gap を可視化する。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.pcn_replay_snapshot import (
    archive_pf_from_snapshot,
    episode_objectives_from_snapshot,
    load_learner_replay_snapshot,
)
from src.agents.pcn_agent import get_non_dominated_inds_minimize

RUN = Path("experiments/distributed_pcn/run1024_amplog_b/20260604_122948")
N_JOBS = 1024


def nd(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    i = get_non_dominated_inds_minimize(points)
    p = points[i] if len(i) else points
    return p[np.argsort(p[:, 0])]


def main() -> None:
    snap = load_learner_replay_snapshot(str(RUN / "learner_replay_snapshot.pkl.gz"))
    explo = episode_objectives_from_snapshot(snap, N_JOBS)
    arch = nd(archive_pf_from_snapshot(snap, N_JOBS))

    hv = json.loads((RUN / "pcn_mo_hv.json").read_text())
    pfe = hv.get("pareto_fronts_per_eval", [])
    main_last = nd(np.asarray(pfe[-1], dtype=np.float64)) if pfe else np.empty((0, 2))
    # best-ever main eval PF = nd over union of all eval fronts
    union = np.concatenate([np.asarray(p, dtype=np.float64) for p in pfe], axis=0) if pfe else np.empty((0, 2))
    main_best = nd(union)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    for ax in axes:
        ax.scatter(explo[:, 0], explo[:, 1], s=6, c="0.78", alpha=0.5,
                   label=f"Exploration cloud ({len(explo)})", zorder=1)
        ax.plot(arch[:, 0], arch[:, 1], "-D", color="#1f77b4", ms=4, lw=1.0,
                label=f"Archive PF — discovered ({len(arch)})", zorder=3)
        ax.plot(main_best[:, 0], main_best[:, 1], "-s", color="#2ca02c", ms=6, lw=1.2,
                label=f"Main-eval PF, best-of-all-evals ({len(main_best)})", zorder=4)
        ax.plot(main_last[:, 0], main_last[:, 1], "-o", color="#d62728", ms=7, lw=1.6,
                label=f"Main-eval PF, final iter100 ({len(main_last)})", zorder=5)
        ax.set_xlabel("Cost")
        ax.set_ylabel("Average Waiting Time")
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("trace1024 - full view\nArchive PF (discovered) is DENSE & full-range; eval PF (reproduced) is SPARSE")

    # zoom into knee/low-wait region
    axes[1].set_xlim(0.4e9, 1.85e9)
    axes[1].set_ylim(1.0e5, 7.0e5)
    axes[1].set_title("trace1024 — knee/low-wait zoom")
    axes[1].legend(loc="upper right", fontsize=9)

    gap = ""
    if len(arch) and len(main_last):
        gap = f"  |  conditioning gap: archive {len(arch)} → main-eval {len(main_last)}"
    fig.suptitle(f"Current PF state (amplog_b iter100){gap}", fontsize=13, y=1.02)
    fig.tight_layout()
    out = Path("pf_1024_current_diagnosis.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved: {out}")
    print(f"explo={len(explo)} archive_pf={len(arch)} main_last={len(main_last)} main_best={len(main_best)}")
    print(f"archive cost[{arch[:,0].min():.3e},{arch[:,0].max():.3e}] wait[{arch[:,1].min():.3e},{arch[:,1].max():.3e}]")
    print(f"main_last cost[{main_last[:,0].min():.3e},{main_last[:,0].max():.3e}] wait[{main_last[:,1].min():.3e},{main_last[:,1].max():.3e}]")


if __name__ == "__main__":
    main()
