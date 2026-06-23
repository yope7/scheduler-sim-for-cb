#!/usr/bin/env python3
"""後回し defer の厳密図(研究用)。実装 event_native_env.py:_defer_rotate
  j = min(i + offset, N-1);  jobs = insert(delete(jobs, i), j, jobs[i])
を正確に再現し、offset=1 vs 4 での「並べ替えで動く要素数＝意思決定経路の擾乱量」を定量化。
cap(DEFER_MAX=3)との関係で「擾乱 vs 到達深さ」のトレードオフも示す。出力 docs/figures/impl_defer.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

N = 10
giant_i = 1                          # 巨大ジョブの位置
occ = np.array([1, 9, 1.4, 1.2, 1.6, 1.1, 1.3, 1.0, 1.5, 1.2])  # 占有量(pt×nodes), index1が巨大


def defer_rotate(order, i, offset):
    """実装どおり: i のジョブを j=min(i+offset, N-1) へ delete+insert。"""
    j = min(i + offset, len(order) - 1)
    o = order.copy()
    v = o.pop(i)
    o.insert(j, v)
    return o, j


def moved_count(before, after):
    return int(np.sum(np.array(before) != np.array(after)))


fig, axes = plt.subplots(2, 2, figsize=(13, 6.6),
                         gridspec_kw={"height_ratios": [1, 1], "hspace": 0.5, "wspace": 0.25})

# 上段: offset=1 と offset=4 の並べ替え(動く要素を色分け)
for col, offset in [(0, 1), (1, 4)]:
    ax = axes[0][col]
    before = list(range(N))
    after, j = defer_rotate(before, giant_i, offset)
    moved = moved_count(before, after)
    for x, idx in enumerate(after):
        is_giant = (idx == giant_i)
        is_moved = (after[x] != before[x])
        col_f = "#e0556f" if is_giant else ("#f0a050" if is_moved else "#56607a")
        ax.add_patch(mp.Rectangle((x, 0), 0.82, occ[idx], facecolor=col_f, edgecolor="k", lw=0.4))
    ax.annotate("", xy=(j + 0.4, 9.6), xytext=(giant_i + 0.4, 9.6),
                arrowprops=dict(arrowstyle="->", color="#e0556f", lw=1.6))
    ax.text((giant_i + j) / 2 + 0.4, 9.9, f"giant: +{offset} back", ha="center", color="#e0556f", fontsize=9)
    ax.set_title(f"offset={offset}:  moved elements = {moved}  (giant + {moved-1} shifted)", fontsize=11)
    ax.set_xlim(-0.3, N); ax.set_ylim(0, 11)
    ax.set_xlabel("queue position"); ax.set_ylabel("occupancy (pt×nodes)")
    ax.set_xticks(range(N)); ax.tick_params(labelsize=7); ax.grid(alpha=0.25, axis="y")

# 下段左: offset vs 動く要素数(擾乱) と 到達深さ(cap=3)
axD = axes[1][0]
offs = np.arange(1, 7)
disturb = offs + 1                      # 1回deferで動く要素数 = offset+1
depth = offs * 3                        # cap=3回 繰り返したときの最大後退距離
axD.plot(offs, disturb, "-o", color="#e0556f", label="path disturbance (moved per defer = offset+1)")
axD.plot(offs, depth, "-s", color="#4da3ff", label="max reach depth (offset × cap3)")
axD.axvline(1, color="#5ee0a0", ls="--", lw=1.2); axD.text(1.05, 17, "Ver.2\noffset=1", color="#2a8f5f", fontsize=9)
axD.axvline(4, color="#888", ls="--", lw=1.0); axD.text(4.05, 17, "Ver.1\noffset=4", color="#555", fontsize=9)
axD.set_title("(C) disturbance vs reach: offset=1 minimizes path perturbation", fontsize=10)
axD.set_xlabel("offset"); axD.set_ylabel("count / distance"); axD.grid(alpha=0.3); axD.legend(fontsize=8)

# 下段右: テキストまとめ(定量)
axT = axes[1][1]; axT.axis("off")
b1 = list(range(N)); a1, _ = defer_rotate(b1, giant_i, 1)
b4 = list(range(N)); a4, _ = defer_rotate(b4, giant_i, 4)
txt = (
    "Quantified path perturbation (1 defer):\n"
    f"  offset=1  -> moved {moved_count(b1,a1)} elements  (giant + 1 shift)\n"
    f"  offset=4  -> moved {moved_count(b4,a4)} elements  (giant + 4 shifts)\n\n"
    "Mechanism (event_native_env.py:_defer_rotate):\n"
    "  j = min(i+offset, N-1)\n"
    "  jobs = insert(delete(jobs, i), j, jobs[i])\n"
    "  index_next_job unchanged -> next = shifted small job\n\n"
    "Larger offset = more elements reorder between\n"
    "consecutive decisions = noisier decision path\n"
    "across seeds (-> higher following variance).\n"
    "Ver.2 picks offset=1 (minimal perturbation),\n"
    "depth via cap=3 repeats (1 step each)."
)
axT.text(0.0, 0.98, txt, transform=axT.transAxes, va="top", ha="left", fontsize=9.5,
         family="monospace", bbox=dict(boxstyle="round", fc="#0b0e14", ec="#56607a", alpha=0.9), color="#dfe6f0")

fig.suptitle("(3) defer offset (rigorous): moved-element count = decision-path disturbance  |  event_native_env.py:_defer_rotate :143-157",
             fontsize=11.5, y=0.99)
fig.savefig("docs/figures/impl_defer.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/impl_defer.png")
print(f"offset1 moved={moved_count(b1,a1)} offset4 moved={moved_count(b4,a4)}")
