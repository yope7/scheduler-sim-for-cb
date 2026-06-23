#!/usr/bin/env python3
"""film(baseline, Fourier無し) vs fourier(b2) の greedy を、共通の真PF に重ねて gap を比較。
   共通の真PF = 非支配( random-p掃引 ∪ 両方策の best-of-k )。どちらの方策にも公平な緑線。
   gap-to-true(待ち超過の正規化平均)が b2 で下がれば「赤/青がより真PFへ寄った」=成功。
   usage: FILM=truepf_film.npz FOUR=truepf_fourier.npz SEEDS=1,2 OUT=pf_b2_compare.png python scripts/plot_b2_compare.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

F = np.load(os.environ["FILM"]); G = np.load(os.environ["FOUR"])
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2").split(",")]
OUT = os.environ.get("OUT", "pf_b2_compare.png")


def gap(greedy, truepf):
    ex = np.clip(greedy[:, 1] - np.interp(greedy[:, 0], truepf[:, 0], truepf[:, 1]), 0, None) / max(1e-9, truepf[:, 1].ptp())
    return float(ex.mean())


fig, axes = plt.subplots(1, len(SEEDS), figsize=(7 * len(SEEDS), 5.8)); axes = np.atleast_1d(axes)
res = []
for ax, sd in zip(axes, SEEDS):
    rp = F[f"rp_{sd}"]; fg = F[f"greedy_{sd}"]; gg = G[f"greedy_{sd}"]
    allp = np.vstack([rp, F[f"samp_{sd}"], G[f"samp_{sd}"]])           # 共通の母集団
    nd = get_non_dominated_inds_minimize(allp); truepf = allp[nd]; truepf = truepf[np.argsort(truepf[:, 0])]
    gf, go = gap(fg, truepf), gap(gg, truepf); res.append((sd, gf, go))
    ax.scatter(rp[:, 0], rp[:, 1], s=12, c="#cfe8cf", alpha=0.5, zorder=0, label="random-p sweep")
    ax.plot(truepf[:, 0], truepf[:, 1], "-", color="#2ca02c", lw=2.2, zorder=2, label=f"common TRUE PF ({len(truepf)})")
    ax.scatter(fg[:, 0], fg[:, 1], s=26, c="#d62728", edgecolor="k", lw=0.3, zorder=4, label=f"baseline (no Fourier) gap={gf:.3f}")
    ax.scatter(gg[:, 0], gg[:, 1], s=26, c="#1a73e8", edgecolor="k", lw=0.3, marker="D", zorder=5, label=f"b2 Fourier gap={go:.3f}")
    kind = "SEEN (trained) seed=0" if sd == 0 else f"UNSEEN seed={sd}"
    ax.set_title(f"{kind}: gap-to-true  film {gf:.3f}  /  b2 {go:.3f}", fontsize=11)
    ax.set_xlabel("Cost"); ax.set_ylabel("Avg Wait"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    print(f"seed={sd}: film gap={gf:.3f}  b2 gap={go:.3f}  delta={go-gf:+.3f}")
_tag = os.environ.get("TAG", "")
fig.suptitle("b2 (Fourier command encoding) vs baseline: greedy vs each instance's TRUE PF (common reference)"
             + (f" — {_tag}" if _tag else ""), fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight"); print(f"saved {OUT}")
print("SUMMARY " + " | ".join(f"seed{sd}: {gf:.3f}->{go:.3f}" for sd, gf, go in res))
