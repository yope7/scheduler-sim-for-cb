#!/usr/bin/env python3
"""b2 vs baseline を全インスタンス集約。各 instance の共通真PFに対する gap を baseline/b2 で並べ、
   (左) baseline gap の大きい順の棒、(右) baseline gap vs b2 gap の散布（y=x の下=b2が良い）。
   「baseline gap が大きい（真PFから遠い）ほど b2 が効く」系統性と、平均/最悪 gap の低下を見る。
   判定語(no improvement 等)は入れない。usage: python scripts/plot_b2_summary.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

# (film_npz, fourier_npz, [seeds]) — 存在するものだけ使う
GROUPS = [
    ("truepf_film_s0.npz",  "truepf_fourier_s0.npz",  [0]),
    ("truepf_film.npz",     "truepf_fourier.npz",     [1, 2]),
    ("truepf_film_s34.npz", "truepf_fourier_s34.npz", [3, 4]),
]
OUT = os.environ.get("OUT", "pf_b2_summary.png")


def gap(greedy, truepf):
    ex = np.clip(greedy[:, 1] - np.interp(greedy[:, 0], truepf[:, 0], truepf[:, 1]), 0, None) / max(1e-9, truepf[:, 1].ptp())
    return float(ex.mean())


rows = []  # (seed, seen?, film_gap, b2_gap)
for fn, gn, seeds in GROUPS:
    if not (os.path.exists(fn) and os.path.exists(gn)):
        print(f"skip (missing): {fn} / {gn}"); continue
    F = np.load(fn); G = np.load(gn)
    for sd in seeds:
        if f"greedy_{sd}" not in F or f"greedy_{sd}" not in G:
            continue
        rp = F[f"rp_{sd}"]; fg = F[f"greedy_{sd}"]; gg = G[f"greedy_{sd}"]
        allp = np.vstack([rp, F[f"samp_{sd}"], G[f"samp_{sd}"]])
        nd = get_non_dominated_inds_minimize(allp); tp = allp[nd]; tp = tp[np.argsort(tp[:, 0])]
        rows.append((sd, sd == 0, gap(fg, tp), gap(gg, tp)))

rows.sort(key=lambda r: -r[2])  # baseline gap 降順
labels = [f"seed{sd}{'*' if seen else ''}" for sd, seen, _, _ in rows]
fg = np.array([r[2] for r in rows]); bg = np.array([r[3] for r in rows])
print("per-instance (sorted by baseline gap):")
for (sd, seen, a, b) in rows:
    print(f"  seed{sd}{'(seen)' if seen else ''}: baseline={a:.3f}  b2={b:.3f}  delta={b-a:+.3f}")
print(f"AVG  baseline={fg.mean():.3f}  b2={bg.mean():.3f}   WORST  baseline={fg.max():.3f}  b2={bg.max():.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
x = np.arange(len(rows)); w = 0.38
ax1.bar(x - w / 2, fg, w, color="#d62728", label="baseline (no Fourier)")
ax1.bar(x + w / 2, bg, w, color="#1a73e8", label="b2 Fourier")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.axhline(fg.mean(), color="#d62728", ls="--", lw=1, alpha=.6)
ax1.axhline(bg.mean(), color="#1a73e8", ls="--", lw=1, alpha=.6)
ax1.set_ylabel("gap-to-true PF (lower = better)")
ax1.set_title(f"per-instance gap (sorted by baseline)\nAVG {fg.mean():.3f} -> {bg.mean():.3f}   WORST {fg.max():.3f} -> {bg.max():.3f}", fontsize=11)
ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=.3)
ax1.text(0.02, -0.16, "* = seen (trained) instance", transform=ax1.transAxes, fontsize=8, color="#5f6368")

lim = max(fg.max(), bg.max()) * 1.1
ax2.plot([0, lim], [0, lim], "k--", lw=1, alpha=.5)
ax2.fill_between([0, lim], [0, lim], 0, color="#e6f4ea", alpha=.5)  # 下三角=b2が良い
for (sd, seen, a, b) in rows:
    ax2.scatter(a, b, s=90, c="#1a73e8" if b < a else "#d62728", edgecolor="k", lw=.4, marker="*" if seen else "o", zorder=3)
    ax2.annotate(f"seed{sd}", (a, b), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_xlabel("baseline gap"); ax2.set_ylabel("b2 gap")
ax2.set_title("baseline gap vs b2 gap\n(points below dashed line: b2 closer to true PF)", fontsize=11)
ax2.grid(alpha=.3)
fig.suptitle(f"b2 (Fourier command encoding) vs baseline across {len(rows)} instances (1 seen + {len(rows)-1} unseen)", fontsize=12.5, y=1.02)
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved {OUT}")
