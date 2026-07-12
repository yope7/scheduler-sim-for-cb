#!/usr/bin/env python3
"""trace1024 完全16セルの全パターン達成PF図(4x4グリッド)。各セルの各seedを色分けで重ねる。
行=F(フーリエ)/W(密度重み), 列=E(探索)/D(後回しoffset1)。データ /tmp/ladrich_fdk*.json の "pf"。
軸はデータから自動スケール(崩壊セルの高待ち側も見切れないように)。出力 docs/figures/fk1024_pf_grid.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEED_COLORS = {1: "#3b82f6", 2: "#f0902f", 3: "#10b981"}  # seed1=青 / seed2=橙 / seed3=緑


def load_pf(F, W, E, D, i):
    p = f"/tmp/ladrich_fdk{F}{W}{E}{D}_{i}.json"
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1])
        return d.get("pf") if "err" not in d else None
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


FW = [(0, 0), (0, 1), (1, 0), (1, 1)]
ED = [(0, 0), (0, 1), (1, 0), (1, 1)]

# 全フロントから軸スケールを自動決定(固定軸だと崩壊セルの高待ち側が見切れるため)
allc, allw = [], []
for F, W in FW:
    for E, D in ED:
        for i in [1, 2, 3]:
            fr = nd_front(load_pf(F, W, E, D, i))
            if fr is not None:
                allc.append(float(fr[:, 0].max())); allw.append(float(fr[:, 1].max()))
cmax = max(allc) if allc else 1.0
wmax = max(allw) if allw else 1.0
cdiv = 10 ** (int(np.floor(np.log10(cmax))) if cmax > 0 else 0)
wdiv = 10 ** (int(np.floor(np.log10(wmax))) if wmax > 0 else 0)
xlim = cmax / cdiv * 1.05
ylim = wmax / wdiv * 1.05

onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
seen = set()
for r, (F, W) in enumerate(FW):
    for c, (E, D) in enumerate(ED):
        ax = axes[r][c]
        any_pf = False
        for i in [1, 2, 3]:
            fr = nd_front(load_pf(F, W, E, D, i))
            if fr is not None:
                any_pf = True; seen.add(i)
                ax.plot(fr[:, 0] / cdiv, fr[:, 1] / wdiv, "-o", c=SEED_COLORS[i],
                        ms=3, lw=1.2, alpha=0.85, label=f"seed{i}")
        ax.set_title(f"F:{onoff[F]} W:{onoff[W]} E:{onoff[E]} D:{onoff[D]}", fontsize=10)
        ax.set_xlim(0, xlim); ax.set_ylim(0, ylim)
        ax.set_xlabel(f"Cost (x{cdiv:.0e})", fontsize=8); ax.set_ylabel(f"Wait (x{wdiv:.0e})", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        if not any_pf:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", transform=ax.transAxes)
handles = [Line2D([0], [0], color=SEED_COLORS[i], marker="o", lw=1.2, ms=5, label=f"seed{i}") for i in sorted(seen)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=len(handles), fontsize=11, framealpha=0.9)
fig.suptitle("trace1024 full 2^4 ablation: achieved Pareto front (all 16 cells, seeds color-coded, auto-scaled axes)   "
             "rows=F(Fourier)/W(density)  cols=E(explore)/D(defer)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("docs/figures/fk1024_pf_grid.png", dpi=85)
print("[SAVED] docs/figures/fk1024_pf_grid.png")
nfilled = sum(1 for F, W in FW for E, D in ED if any(nd_front(load_pf(F, W, E, D, i)) is not None for i in [1, 2, 3]))
print(f"埋まったマス: {nfilled}/16  seeds={sorted(seen)}  (cost_max={cmax:.3g} wait_max={wmax:.3g} cdiv={cdiv:.0e} wdiv={wdiv:.0e})")
