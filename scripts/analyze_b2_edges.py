#!/usr/bin/env python3
"""「低ジョブ数で b2(Fourier) が端点を失う」を定量化。
   greedy(決定的)方策が真PFの両端（低コスト=全オンプレ角 / 高コスト=全クラウド角）に
   どれだけ届くかを baseline(film) と b2 で比較し、n_jobs を横断して図にする。
   端到達 = greedy の cost 最小/最大、wait 最小/最大 を真PF範囲で正規化。
   端ギャップ(低コスト) = (greedy_min_cost - truePF_min_cost)/truePF_cost_range（大=端を失っている）。
   usage: OUT=pf_b2_edges.png PYTHONPATH=. .venv/bin/python scripts/analyze_b2_edges.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

SIZES = [16, 32, 64, 128, 256, 512, 1024]
OUT = os.environ.get("OUT", "pf_b2_edges.png")


def names(nj):
    if nj == 128:
        return "truepf_film_s0.npz", "truepf_fourier_s0.npz"
    return f"truepf_film_{nj}_s0.npz", f"truepf_fourier_{nj}_s0.npz"


rows = []  # nj, truePF(cmin,cmax,wmin,wmax), film greedy, b2 greedy
for nj in SIZES:
    fn, gn = names(nj)
    if not (os.path.exists(fn) and os.path.exists(gn)):
        continue
    F = np.load(fn); G = np.load(gn)
    if "greedy_0" not in F or "greedy_0" not in G:
        continue
    fg = F["greedy_0"]; gg = G["greedy_0"]
    allp = np.vstack([F["rp_0"], F["samp_0"], G["samp_0"]])
    nd = get_non_dominated_inds_minimize(allp); tp = allp[nd]
    cmin, cmax = tp[:, 0].min(), tp[:, 0].max(); crange = max(1.0, cmax - cmin)
    wmin, wmax = tp[:, 1].min(), tp[:, 1].max(); wrange = max(1e-9, wmax - wmin)
    # 低コスト端ギャップ（greedy が安い角にどれだけ届かないか、正規化、大=端喪失）
    lowcost_gap_f = (fg[:, 0].min() - cmin) / crange
    lowcost_gap_b = (gg[:, 0].min() - cmin) / crange
    # 高wait端ギャップ（同じ全オンプレ角を wait 側から見る、大=端喪失）
    hiwait_gap_f = (wmax - fg[:, 1].max()) / wrange
    hiwait_gap_b = (wmax - gg[:, 1].max()) / wrange
    rows.append(dict(nj=nj, lcf=lowcost_gap_f, lcb=lowcost_gap_b, hwf=hiwait_gap_f, hwb=hiwait_gap_b))
    print(f"n={nj:>4}  low-cost-edge gap film={lowcost_gap_f:.3f} b2={lowcost_gap_b:.3f}  "
          f"high-wait-edge gap film={hiwait_gap_f:.3f} b2={hiwait_gap_b:.3f}  "
          f"{'b2 LOSES edge' if lowcost_gap_b > lowcost_gap_f + 0.02 else 'b2 ok/better'}")

if not rows:
    raise SystemExit("no data")

xs = np.array([r["nj"] for r in rows])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (kf, kb, title, ylab) in zip(
    (ax1, ax2),
    (("lcf", "lcb", "low-cost edge (all-on-prem corner): how far greedy FAILS to reach the cheap end",
      "normalized low-cost gap (higher = edge lost)"),
     ("hwf", "hwb", "high-wait edge (same corner, wait side): how far greedy FAILS to reach",
      "normalized high-wait gap (higher = edge lost)"))):
    f = np.array([r[kf] for r in rows]); b = np.array([r[kb] for r in rows])
    ax.fill_between(xs, f, b, where=(b > f), interpolate=True, color="#fce8e6", alpha=.8, label="b2 worse (edge lost)")
    ax.fill_between(xs, f, b, where=(b <= f), interpolate=True, color="#e6f4ea", alpha=.8, label="b2 better")
    ax.plot(xs, f, "o-", color="#d62728", lw=2, ms=8, label="baseline (no Fourier)")
    ax.plot(xs, b, "D-", color="#1a73e8", lw=2, ms=8, label="b2 Fourier")
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("n_jobs (log2)"); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9.5); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="best")
fig.suptitle("b2 loses the PF EDGE at small n_jobs (Fourier features can't extrapolate past the trained command range)", fontsize=11.5, y=1.02)
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved {OUT}  (sizes: {list(xs)})")
