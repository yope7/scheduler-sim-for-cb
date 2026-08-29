#!/usr/bin/env python3
"""2万ジョブ・正容量(ρ=0.70)の主PF図。

系列は最大3つ(指示): pダイヤル(灰・破線) / 参照線=多族掃引の非支配包絡(オレンジ) / PCN(青点)。
左=全域、右=左端の拡大(参照線が効く領域)。x はクラウド代に対する割合(無次元)。

usage:
  FAM=results/eval_pf/famsweep_weekB20000_cap48000.npz \
  [PCN=results/eval_pf/xxx.npz] [NSGA=results/eval_pf/xxx.npz] \
  OUT=docs/figures/pf_main20000.png PYTHONPATH=. .venv/bin/python scripts/plot_pf_main20000.py
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
_F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_F):
    font_manager.fontManager.addfont(_F)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
import matplotlib.pyplot as plt
import numpy as np

FAM = os.environ.get("FAM", "results/eval_pf/famsweep_weekB20000_cap48000.npz")
PCN = os.environ.get("PCN", "")
NSGA = os.environ.get("NSGA", "")
OUT = os.environ.get("OUT", "docs/figures/pf_main20000.png")
# 全クラウド代(=cost の自然な正規化子)。weekB先頭2万・外れ値除外の Σ(pt*nodes)
CLOUD_ALL = float(os.environ.get("CLOUD_ALL", "5784096599"))

C_PCN = "#2563eb"   # 青: PCN(主役)
C_REF = "#d97706"   # オレンジ: 参照線(多族掃引)
C_PD = "#6b7280"    # 灰: pダイヤル(ベースライン。線種でも二次符号化)
C_NSGA = "#15803d"  # 緑: NSGA(あれば)


def nd(pts):
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    if not len(p):
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    keep, best = [], np.inf
    for i in range(len(p)):
        if p[i, 1] < best:
            keep.append(i)
            best = p[i, 1]
    return p[keep]


d = np.load(FAM)
ref = nd(d["pf"])
pdial = nd(d["pdial_pf"] if "pdial_pf" in d.files else d["pdial"])
pcn = nd(np.load(PCN)["pf"]) if PCN and os.path.exists(PCN) else None
nsga = nd(np.load(NSGA)["pf"]) if NSGA and os.path.exists(NSGA) else None

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for k, ax in enumerate(axes):
    x = lambda a: a[:, 0] / CLOUD_ALL * 100.0  # noqa: E731
    ax.plot(x(pdial), pdial[:, 1], "--o", color=C_PD, lw=2, ms=5, mfc="white",
            mew=1.5, label="pダイヤル(クラウド率を掃引)", zorder=2)
    if nsga is not None:
        ax.plot(x(nsga), nsga[:, 1], "-^", color=C_NSGA, lw=2, ms=6,
                label="NSGA-II", zorder=3)
    ax.plot(x(ref), ref[:, 1], "-s", color=C_REF, lw=2.5, ms=6,
            label="参照線(多族掃引の包絡)", zorder=4)
    if pcn is not None:
        ax.plot(x(pcn), pcn[:, 1], "o", color=C_PCN, ms=8, mew=1.5,
                mec="white", label="PCN(学習)", zorder=5)
    ax.set_xlabel("クラウド代 / 全部クラウドに出した場合 [%]")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("平均待ち時間 [秒]")
axes[0].set_title("全域", fontsize=11, loc="left")
axes[0].legend(fontsize=9, frameon=False)
zoom = max(x_ := (ref[:, 0].max() / CLOUD_ALL * 100.0), 1.0) * 1.15
axes[1].set_xlim(-zoom * 0.03, zoom)
axes[1].set_title(f"左端の拡大 (0〜{zoom:.0f}%)", fontsize=11, loc="left")

fig.suptitle("2万ジョブ・正容量(オンプレ48000/クラウド192000, ρ=0.70) weekB先頭2万",
             fontsize=12.5, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.94))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")

# 参考値
print(f"参照線: {len(ref)}点  cost {ref[:,0].min():.3e}..{ref[:,0].max():.3e} "
      f"wait {ref[:,1].min():.2f}..{ref[:,1].max():.2f}")
print(f"pダイヤル: {len(pdial)}点")
if pcn is not None:
    print(f"PCN: {len(pcn)}点  cost {pcn[:,0].min():.3e}..{pcn[:,0].max():.3e} "
          f"wait {pcn[:,1].min():.2f}..{pcn[:,1].max():.2f}")
