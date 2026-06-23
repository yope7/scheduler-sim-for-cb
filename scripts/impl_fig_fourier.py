#!/usr/bin/env python3
"""フーリエ命令エンコーディングの厳密図(研究用)。実装 pcn_agent.py の Φ(c)=[c, sin(2^k c), cos(2^k c)]
(k=0..3, geometric f={1,2,4,8}) を正確に計算し、近接2命令の「分離の増幅」を定量化する。
- 左: 各周波数の sin(2^k c) 波形 + 近接2命令 c_A,c_B(正規化空間 Δz=0.15) の縦線。
- 右上: 9次元特徴 Φ の成分ごとの c_A,c_B 値。
- 右下: 成分ごとの寄与 (Φ_k(c_A)-Φ_k(c_B))^2 と、生スカラ距離 vs フーリエ後L2距離の比較(増幅率)。
出力 docs/figures/impl_fourier.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = [2 ** k for k in range(4)]   # f = 1,2,4,8 (geometric, pcn_agent.py:482)
cA, cB = 0.0, 0.15                   # 正規化(z-score)空間の近接2命令。実装の分解能目安 ~0.15


def phi(c):
    """Φ(c) = [c, sin(f1 c), cos(f1 c), ..., sin(fL c), cos(fL c)]  (生cを先頭, 1+2L=9次元)"""
    feats = [c]
    for f in BANDS:
        feats += [np.sin(f * c), np.cos(f * c)]
    return np.array(feats)


labels = ["c"] + sum([[f"sin({f}c)", f"cos({f}c)"] for f in BANDS], [])
pA, pB = phi(cA), phi(cB)
contrib = (pA - pB) ** 2
raw_dist = abs(cA - cB)                     # 生スカラの距離
four_dist = np.sqrt(contrib.sum())          # フーリエ後のL2距離
amp = four_dist / raw_dist                  # 増幅率

fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], hspace=0.42, wspace=0.28)

# 左: 波形
ax = fig.add_subplot(gs[:, 0])
cc = np.linspace(-1.5, 1.5, 800)
colors = ["#9aa7b8", "#4da3ff", "#f0a050", "#e0556f"]
for f, col in zip(BANDS, colors):
    ax.plot(cc, np.sin(f * cc), color=col, lw=1.6, label=f"sin({f}·c)")
ax.axvline(cA, color="#5ee0a0", ls="--", lw=1.4)
ax.axvline(cB, color="#5ee0a0", ls="--", lw=1.4)
ax.annotate("", xy=(cB, -1.18), xytext=(cA, -1.18), arrowprops=dict(arrowstyle="<->", color="#5ee0a0"))
ax.text((cA + cB) / 2, -1.34, f"Δz = {cB-cA:.2f}", ha="center", color="#2a8f5f", fontsize=10)
ax.text(cA, 1.18, "c_A", ha="center", color="#2a8f5f", fontsize=10)
ax.text(cB, 1.32, "c_B", ha="center", color="#2a8f5f", fontsize=10)
ax.set_title("Fourier basis sin(2^k c), f in {1,2,4,8}\nnearby commands separate more at higher freq", fontsize=11)
ax.set_xlabel("normalized command c  (z-score)"); ax.set_ylabel("value")
ax.set_ylim(-1.5, 1.6); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right", ncol=2)

# 右上: Φ成分の値
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(labels))
ax2.bar(x - 0.2, pA, 0.4, label="Φ(c_A)", color="#4da3ff")
ax2.bar(x + 0.2, pB, 0.4, label="Φ(c_B)", color="#e0556f")
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=60, fontsize=7, ha="right")
ax2.set_title("9-dim feature Phi(c): raw c + sin/cos x 4 freqs", fontsize=10)
ax2.axhline(0, color="#888", lw=0.6); ax2.grid(alpha=0.3, axis="y"); ax2.legend(fontsize=8)

# 右下: 成分寄与 + 増幅率
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(x, contrib, 0.6, color="#f0a050")
ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=60, fontsize=7, ha="right")
ax3.set_title("separation contribution (Phi_k(c_A) - Phi_k(c_B))^2  (high freq dominates)", fontsize=10)
ax3.grid(alpha=0.3, axis="y")
ax3.text(0.98, 0.92,
         f"raw scalar dist |c_A-c_B| = {raw_dist:.3f}\nFourier L2 dist ||Phi(c_A)-Phi(c_B)|| = {four_dist:.3f}\n-> separation amplified x{amp:.1f}",
         transform=ax3.transAxes, ha="right", va="top", fontsize=9,
         bbox=dict(boxstyle="round", fc="#0b0e14", ec="#5ee0a0", alpha=0.9), color="#dfe6f0")

fig.suptitle("(1) Fourier command encoding (rigorous)   Phi(c)=[c, sin(2^k c), cos(2^k c)], k=0..3   |   pcn_agent.py:_encode_cmd",
             fontsize=12.5, y=0.99)
fig.savefig("docs/figures/impl_fourier.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/impl_fourier.png")
print(f"raw_dist={raw_dist:.4f} four_dist={four_dist:.4f} amp=x{amp:.2f}")
