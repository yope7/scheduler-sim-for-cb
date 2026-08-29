#!/usr/bin/env python3
"""Ver.2(fdタグ: 密度W + offset1)の16セルを 50%中央達成面で 4x4 グリッドに並べる。
データ源: W=OFF&D=OFF は Ver.1流用(/tmp/rich48b_fc{FWED}_i.json)、それ以外は /tmp/v2rich_fd{FWED}_i.json。
どちらも最終{行がJSON(v2richはstdoutにログ混在)。各JSONの"pf"を使う。
薄灰=各seedの達成PF, 太赤=各cost水準で待ち中央値の50%達成面。出力 docs/figures/factorial_pf_grid_v2.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COST_REF = 5.8e8
SEED_COLORS = {1: "#3b82f6", 2: "#f0902f", 3: "#10b981"}  # seed1=青 / seed2=橙 / seed3=緑


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
    p = np.asarray(pf, float)
    p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0)]
    if len(p) < 2:
        return None
    keep = []
    for k in range(len(p)):
        dom = any((p[j, 0] <= p[k, 0]) and (p[j, 1] <= p[k, 1]) and (j != k) and
                  ((p[j, 0] < p[k, 0]) or (p[j, 1] < p[k, 1])) for j in range(len(p)))
        if not dom:
            keep.append(p[k])
    p = np.array(keep)
    return p[np.argsort(p[:, 0])]


def cell_fronts(F, W, E, D):
    out = []
    for i in [1, 2, 3]:
        fr = nd_front(load_pf(F, W, E, D, i))
        if fr is not None:
            out.append((i, fr))
    return out


FW = [(0, 0), (0, 1), (1, 0), (1, 1)]
ED = [(0, 0), (0, 1), (1, 0), (1, 1)]
onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
nfilled = 0; seen = set()
for r, (F, W) in enumerate(FW):
    for c, (E, D) in enumerate(ED):
        ax = axes[r][c]
        fr_list = cell_fronts(F, W, E, D)
        reuse = (W == 0 and D == 0)
        title = f"F:{onoff[F]} W:{onoff[W]} E:{onoff[E]} D:{onoff[D]}" + ("  [reuse]" if reuse else "")
        if fr_list:
            nfilled += 1
            for i, fr in fr_list:
                ax.plot(fr[:, 0] / 1e8, fr[:, 1] / 1e3, "-o", c=SEED_COLORS[i],
                        ms=2.3, lw=1.3, alpha=0.9, label=f"seed{i}")
                seen.add(i)
            ax.set_title(f"{title}\n({len(fr_list)} seeds)", fontsize=10)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=10, color="gray")
        ax.set_xlim(0, 5.8); ax.set_ylim(0, 160)
        ax.set_xlabel("Cost (x1e8)", fontsize=8); ax.set_ylabel("Wait (x1e3)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
handles = [Line2D([0], [0], color=SEED_COLORS[i], marker="o", lw=1.3, ms=5, label=f"seed{i}") for i in sorted(seen)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=len(handles), fontsize=11, framealpha=0.9)
fig.suptitle("Ver.2 (density-W + defer offset=1) PF grid (each seed color-coded)  "
             "rows=F(Fourier)/W(density)  cols=E(explore)/D(defer offset1)\n"
             "seed1=blue / seed2=orange / seed3=green   [reuse]=W&D both OFF, taken from Ver.1", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/factorial_pf_grid_v2.png", dpi=85)
print("[SAVED] docs/figures/factorial_pf_grid_v2.png")
print(f"埋まったマス: {nfilled}/16  seeds={sorted(seen)}")
