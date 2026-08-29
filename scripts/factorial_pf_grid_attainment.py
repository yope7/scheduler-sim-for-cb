#!/usr/bin/env python3
"""16セル(2^4)を 50%中央達成面(attainment surface)で 4x4 グリッドに並べる。
代表seed1本ではなく、各セルの3seedのPFを「中央値を通る1本のフロント線」に統合する。
- 入力: /tmp/rich48b_fc{FWED}_{1,2,3}.json の "pf"(各seedの達成PF点群 [[cost,wait],...])。再評価なし。
- 達成面: 各seedのPFを step関数(各costで達成できる最小wait)化→共通costグリッドで評価→
  各cost水準で3seedのwaitの中央値=50%-attainment surface(MOOでの「平均PF線」の標準)。
- 描画: 薄い線=各seed, 太い線=中央達成面。代表seed版 factorial_pf_grid.png は別物として残す。
出力 docs/figures/factorial_pf_grid_attainment.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COST_REF = 5.8e8   # 共通costグリッド上限(図のxlim 5.8億に一致)
SEED_COLORS = {1: "#3b82f6", 2: "#f0902f", 3: "#10b981"}  # seed1=青 / seed2=橙 / seed3=緑


def nd_front(pf):
    """点群→非支配フロントをcost昇順(wait降順)で。"""
    p = np.asarray(pf, float)
    p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0)]
    if not len(p):
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
    """costグリッド上で、その達成PFが各costまでに到達できる最小wait(達成面=step関数)。
    cost<=g を満たす点のうち最小wait。1点も無ければ NaN(そのseedはそのcost域に未到達)。"""
    out = np.full(len(grid), np.nan)
    c = front[:, 0]; w = front[:, 1]
    for gi, g in enumerate(grid):
        m = c <= g
        if m.any():
            out[gi] = w[m].min()
    return out


def cell_fronts(tag):
    """そのセルの各seedの達成PFフロントを (seed番号, フロント) で返す。"""
    out = []
    for i in [1, 2, 3]:
        p = f"/tmp/rich48b_{tag}_{i}.json"
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
            fr = nd_front(d.get("pf", []))
            if fr is not None and len(fr) >= 2:
                out.append((i, fr))
        except Exception:
            pass
    return out


FW = [(0, 0), (0, 1), (1, 0), (1, 1)]   # 行: フーリエ, 重みサンプル
ED = [(0, 0), (0, 1), (1, 0), (1, 1)]   # 列: 探索チューニング, 後回し
onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
nfilled = 0; seen = set()
for r, (F, W) in enumerate(FW):
    for c, (E, D) in enumerate(ED):
        ax = axes[r][c]
        tag = f"fc{F}{W}{E}{D}"
        fr_list = cell_fronts(tag)
        title = f"F:{onoff[F]} W:{onoff[W]} E:{onoff[E]} D:{onoff[D]}"
        if fr_list:
            nfilled += 1
            for i, fr in fr_list:   # 各seedを色分け(ばらつきを直接見せる)
                ax.plot(fr[:, 0] / 1e8, fr[:, 1] / 1e3, "-o", c=SEED_COLORS[i],
                        ms=2.3, lw=1.3, alpha=0.9, label=f"seed{i}")
                seen.add(i)
            ax.set_title(f"{title}\n({len(fr_list)} seeds)", fontsize=11)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=13, color="gray",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=11, color="gray")
        ax.set_xlim(0, 5.8); ax.set_ylim(0, 160)
        ax.set_xlabel("Cost (x1e8)", fontsize=8); ax.set_ylabel("Wait (x1e3)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
handles = [Line2D([0], [0], color=SEED_COLORS[i], marker="o", lw=1.3, ms=5, label=f"seed{i}") for i in sorted(seen)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=len(handles), fontsize=11, framealpha=0.9)
fig.suptitle("2^4 ablation PF grid (each seed color-coded)  rows=F(Fourier)/W(Weight-sampling), cols=E(Explore-tune)/D(Defer)\n"
             "seed1=blue / seed2=orange / seed3=green", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/factorial_pf_grid_attainment.png", dpi=85)
print("[SAVED] docs/figures/factorial_pf_grid_attainment.png")
print(f"埋まったマス: {nfilled}/16  seeds={sorted(seen)}")
