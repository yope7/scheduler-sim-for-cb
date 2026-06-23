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

COST_REF = 5.8e8


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


def attain_wait(front, grid):
    out = np.full(len(grid), np.nan)
    c = front[:, 0]; w = front[:, 1]
    for gi, g in enumerate(grid):
        m = c <= g
        if m.any():
            out[gi] = w[m].min()
    return out


def cell_surfaces(F, W, E, D):
    fronts = []
    for i in [1, 2, 3]:
        fr = nd_front(load_pf(F, W, E, D, i))
        if fr is not None:
            fronts.append(fr)
    if not fronts:
        return [], None, None
    cmin = min(fr[:, 0].min() for fr in fronts)
    grid = np.linspace(cmin, COST_REF, 200)
    waits = np.vstack([attain_wait(fr, grid) for fr in fronts])
    med = np.full(len(grid), np.nan)
    for gi in range(len(grid)):
        col = waits[:, gi]; col = col[~np.isnan(col)]
        if len(col) >= (waits.shape[0] + 1) // 2:
            med[gi] = np.median(col)
    return fronts, grid, med


FW = [(0, 0), (0, 1), (1, 0), (1, 1)]
ED = [(0, 0), (0, 1), (1, 0), (1, 1)]
onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
nfilled = 0
for r, (F, W) in enumerate(FW):
    for c, (E, D) in enumerate(ED):
        ax = axes[r][c]
        fronts, grid, med = cell_surfaces(F, W, E, D)
        reuse = (W == 0 and D == 0)
        title = f"F:{onoff[F]} W:{onoff[W]} E:{onoff[E]} D:{onoff[D]}" + ("  [reuse]" if reuse else "")
        if fronts:
            nfilled += 1
            for fr in fronts:
                ax.plot(fr[:, 0] / 1e8, fr[:, 1] / 1e3, "-", c="gray", lw=0.9, alpha=0.45)
            ok = ~np.isnan(med)
            ax.plot(grid[ok] / 1e8, med[ok] / 1e3, "-", c="crimson", lw=2.6)
            ax.set_title(f"{title}\n3 seeds -> median front", fontsize=10)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=10, color="gray")
        ax.set_xlim(0, 5.8); ax.set_ylim(0, 160)
        ax.set_xlabel("Cost (x1e8)", fontsize=8); ax.set_ylabel("Wait (x1e3)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
fig.suptitle("Ver.2 (density-W + defer offset=1) PF grid (50% ATTAINMENT SURFACE)  "
             "rows=F(Fourier)/W(density)  cols=E(explore)/D(defer offset1)\n"
             "thin gray = each of 3 seeds   thick red = median front   [reuse]=W&D both OFF, taken from Ver.1", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/factorial_pf_grid_v2.png", dpi=85)
print("[SAVED] docs/figures/factorial_pf_grid_v2.png")
print(f"埋まったマス: {nfilled}/16")
