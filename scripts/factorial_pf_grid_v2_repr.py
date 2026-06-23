#!/usr/bin/env python3
"""Ver.2(fdタグ)の16セルを 代表seed(達成PF点数=最豊富なseed) の実フロントで 4x4 グリッドに。
中央達成面版(factorial_pf_grid_v2.py)と対になる代表seed版。Ver.1の2枚構成に合わせる。
データ源: W=OFF&D=OFF は rich48b_fc 流用、それ以外 v2rich_fd。各JSON最終{行の"pf"。
出力 docs/figures/factorial_pf_grid_v2_repr.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def repr_front(F, W, E, D):
    best = None; best_n = -1
    for i in [1, 2, 3]:
        fr = nd_front(load_pf(F, W, E, D, i))
        if fr is not None and len(fr) > best_n:
            best_n = len(fr); best = fr
    return best, best_n


FW = [(0, 0), (0, 1), (1, 0), (1, 1)]
ED = [(0, 0), (0, 1), (1, 0), (1, 1)]
onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
nfilled = 0
for r, (F, W) in enumerate(FW):
    for c, (E, D) in enumerate(ED):
        ax = axes[r][c]
        fr, n = repr_front(F, W, E, D)
        reuse = (W == 0 and D == 0)
        title = f"F:{onoff[F]} W:{onoff[W]} E:{onoff[E]} D:{onoff[D]}" + ("  [reuse]" if reuse else "")
        if fr is not None:
            nfilled += 1
            ax.plot(fr[:, 0] / 1e8, fr[:, 1] / 1e3, "-o", c="crimson", ms=3.5, lw=1.4)
            ax.set_title(f"{title}\nn_pf={n}", fontsize=10)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=10, color="gray")
        ax.set_xlim(0, 5.8); ax.set_ylim(0, 160)
        ax.set_xlabel("Cost (x1e8)", fontsize=8); ax.set_ylabel("Wait (x1e3)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
fig.suptitle("Ver.2 (density-W + defer offset=1) PF grid (REPRESENTATIVE seed, richest front)  "
             "rows=F(Fourier)/W(density)  cols=E(explore)/D(defer offset1)   red=achieved Pareto front", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/factorial_pf_grid_v2_repr.png", dpi=85)
print("[SAVED] docs/figures/factorial_pf_grid_v2_repr.png")
print(f"埋まったマス: {nfilled}/16")
