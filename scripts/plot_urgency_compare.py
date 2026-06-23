#!/usr/bin/env python3
"""緊急度 OFF vs ON の greedy を共通の真PFに重ねて gap 比較。
   「右に寄っただけ」ではなく「理想(真PF)に近づいたか」を gap で判定する。
   共通真PF = 非支配( random-p掃引 ∪ 両方策の best-of-k )=どちらにも公平な緑線。
   gap(待ち超過の正規化平均)が ON で下がれば真の改善。
   usage: OFFNPZ=.. ONNPZ=.. SEEDS=0,1 OUT=pf_urgency_compare.png python scripts/plot_urgency_compare.py
   ラベルは英語(matplotlib 日本語フォント回避)。図の解釈は HTML 側キャプションで日本語化する。"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

OFF = np.load(os.environ["OFFNPZ"]); ON = np.load(os.environ["ONNPZ"])
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
OUT = os.environ.get("OUT", "pf_urgency_compare.png")


def gap(greedy, truepf):
    ex = np.clip(greedy[:, 1] - np.interp(greedy[:, 0], truepf[:, 0], truepf[:, 1]), 0, None) / max(1e-9, truepf[:, 1].ptp())
    return float(ex.mean())


fig, axes = plt.subplots(1, len(SEEDS), figsize=(7 * len(SEEDS), 5.8)); axes = np.atleast_1d(axes)
for ax, sd in zip(axes, SEEDS):
    rp = OFF[f"rp_{sd}"]; og = OFF[f"greedy_{sd}"]; ng = ON[f"greedy_{sd}"]
    allp = np.vstack([rp, OFF[f"samp_{sd}"], ON[f"samp_{sd}"]])      # 共通の母集団
    nd = get_non_dominated_inds_minimize(allp); truepf = allp[nd]; truepf = truepf[np.argsort(truepf[:, 0])]
    g_off, g_on = gap(og, truepf), gap(ng, truepf)
    ax.scatter(rp[:, 0], rp[:, 1], s=12, c="#cfe8cf", alpha=0.5, zorder=0, label="random-p sweep")
    ax.plot(truepf[:, 0], truepf[:, 1], "-", color="#2ca02c", lw=2.2, zorder=2, label=f"common TRUE PF ({len(truepf)})")
    ax.scatter(og[:, 0], og[:, 1], s=30, c="#d62728", edgecolor="k", lw=0.3, zorder=4, label=f"urgency OFF  gap={g_off:.3f}")
    ax.scatter(ng[:, 0], ng[:, 1], s=30, c="#1a73e8", edgecolor="k", lw=0.3, marker="D", zorder=5, label=f"urgency ON   gap={g_on:.3f}")
    kind = "trained seed=0" if sd == 0 else f"unseen seed={sd}"
    ax.set_title(f"{kind}: gap-to-true  OFF {g_off:.3f} -> ON {g_on:.3f}  ({g_on - g_off:+.3f})", fontsize=11)
    ax.set_xlabel("Cost"); ax.set_ylabel("Avg Wait"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    print(f"seed={sd}: OFF gap={g_off:.3f}  ON gap={g_on:.3f}  delta={g_on - g_off:+.3f}")
plt.tight_layout(); plt.savefig(OUT, dpi=110); print("saved", OUT)
