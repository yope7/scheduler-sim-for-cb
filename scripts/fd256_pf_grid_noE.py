#!/usr/bin/env python3
"""Ver.2 256 の16セルPFグリッドから探索(E)を抜いた8セル版(E=OFF固定, F/W/Dの2^3=8)。
2行(F=OFF/ON)×4列(W×D)。各seedの達成パレートフロントを色分け(seed1=青/seed2=橙/seed3=緑)。
中央達成面(赤線)は描かない=run間のばらつきをそのまま見せる。
データ: W=OFF&D=OFF は rich48b_fc 流用, それ以外 v2rich_fd。出力 docs/figures/fd256_pf_grid_noE.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEED_COLORS = {1: "#3b82f6", 2: "#f0902f", 3: "#10b981"}


def load_pf(F, W, E, D, i):
    p = (f"/tmp/rich48b_fc{F}{W}{E}{D}_{i}.json" if (W == 0 and D == 0)
         else f"/tmp/v2rich_fd{F}{W}{E}{D}_{i}.json")
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        return json.loads(lines[-1]).get("pf")
    except Exception:
        return None


def nd_front(pf):
    if not pf:
        return None
    p = np.asarray(pf, float); p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0)]
    if len(p) < 2:
        return None
    keep = [k for k in range(len(p)) if not any((p[j, 0] <= p[k, 0]) and (p[j, 1] <= p[k, 1]) and (j != k) and ((p[j, 0] < p[k, 0]) or (p[j, 1] < p[k, 1])) for j in range(len(p)))]
    p = p[keep]; return p[np.argsort(p[:, 0])]


def cell_fronts(F, W, E, D):
    out = []
    for i in [1, 2, 3]:
        fr = nd_front(load_pf(F, W, E, D, i))
        if fr is not None:
            out.append((i, fr))
    return out


Frows = [0, 1]                                  # 行: F(フーリエ) OFF/ON
WDcols = [(0, 0), (0, 1), (1, 0), (1, 1)]       # 列: W(密度)×D(後回しoffset1)   ※E(探索)はOFF固定
onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
seen = set()
for r, F in enumerate(Frows):
    for c, (W, D) in enumerate(WDcols):
        ax = axes[r][c]
        fr_list = cell_fronts(F, W, 0, D)
        reuse = (W == 0 and D == 0)
        title = f"F:{onoff[F]}  W:{onoff[W]}  D:{onoff[D]}" + ("  [reuse]" if reuse else "")
        if fr_list:
            for i, fr in fr_list:
                ax.plot(fr[:, 0] / 1e8, fr[:, 1] / 1e3, "-o", c=SEED_COLORS[i],
                        ms=2.6, lw=1.5, alpha=0.9, label=f"seed{i}")
                seen.add(i)
            ax.set_title(f"{title}\n(explore E off, {len(fr_list)}seed)", fontsize=10)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=10, color="gray")
        ax.set_xlim(0, 5.8); ax.set_ylim(0, 160)
        ax.set_xlabel("Cost (x1e8)", fontsize=8); ax.set_ylabel("Wait (x1e3)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
handles = [Line2D([0], [0], color=SEED_COLORS[i], marker="o", lw=1.5, ms=5, label=f"seed{i}") for i in sorted(seen)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=len(handles), fontsize=10, framealpha=0.9)
fig.suptitle("Ver.2 256 PF grid (explore E removed)  8 cells = F(Fourier)/W(density)/D(defer offset1)   rows=F  cols=W x D\n"
             "each seed's achieved Pareto front, color-coded (seed1=blue / seed2=orange / seed3=green)   [reuse]=W&D both OFF from Ver.1", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("docs/figures/fd256_pf_grid_noE.png", dpi=95)
print("[SAVED] docs/figures/fd256_pf_grid_noE.png")
nfilled = sum(1 for F in Frows for (W, D) in WDcols if cell_fronts(F, W, 0, D))
print(f"埋まったマス: {nfilled}/8  seeds={sorted(seen)}")
