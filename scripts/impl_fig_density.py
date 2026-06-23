#!/usr/bin/env python3
"""距離ベース密度重みの厳密図(研究用)。実装 pcn_agent.py:2449-2461 を正確に再現:
  各PF点を正規化空間に置き、k番目最近傍距離 r_k(k=2) を測り w = WEIGHT*(r_k/mean)^ALPHA
  (WEIGHT=8, K=2, ALPHA=1)。中間に密集したPF(自己強化後)に適用し、
  (A)疎な端ほど重みが大きく自動配分されること、(B)手動帯(固定%境界)は実分布とズレて
  疎領域(端)を取りこぼすこと、を定量化する。出力 docs/figures/impl_density.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
WEIGHT, K, ALPHA = 8.0, 2, 1.0

# 中間に密集した非支配フロント(端は疎)
u = np.sort(rng.beta(2.0, 2.0, 44)); u = np.concatenate([[0.0], u, [1.0]])
cost = u
wait = (1.0 - np.sqrt(u)) * 0.9 + 0.05
pf = np.column_stack([cost, wait])               # 正規化済み [0,1]^2

# k番目最近傍距離 r_k (実装: d=sort(全点距離, 自分=0含む); r_k=d[:,K])
D = np.sqrt(((pf[:, None, :] - pf[None, :, :]) ** 2).sum(-1))
d_sorted = np.sort(D, axis=1)
r_k = d_sorted[:, K]                              # K=2 → 2番目最近傍
w = WEIGHT * (r_k / r_k.mean()) ** ALPHA          # 密度逆比重み

order = np.argsort(pf[:, 0])                       # cost昇順
i_end = [int(np.argmin(pf[:, 0])), int(np.argmax(pf[:, 0]))]
w_end = w[i_end].mean()
w_mid = w[(pf[:, 0] > 0.3) & (pf[:, 0] < 0.7)].mean()

# 手動帯(workload_pcn_profile.py の cost_frac 既定): MID[.06,.38], KNEE[.04,.12], LOW_SLOPE[.0,.18]
bands = [("MID(16)", 0.06, 0.38, "#4da3ff"), ("KNEE(8)", 0.04, 0.12, "#5ee0a0"),
         ("LOW_SLOPE(6)", 0.0, 0.18, "#f0a050")]
# 手動帯がカバーするcost範囲の和集合 = [0, .38]。端(cost>.38)とcost=1端は素の重み(=1相当)
covered = (pf[:, 0] <= 0.38)
uncovered_frac = float(np.mean(~covered))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

# A: PF + 密度重み(サイズ/色)
sc = axA.scatter(pf[:, 0], pf[:, 1], s=20 + w * 26, c=w, cmap="viridis", zorder=3, edgecolor="k", linewidth=0.3)
axA.plot(pf[:, 0], pf[:, 1], "-", color="#9aa7b8", lw=1.0, zorder=1)
for i in i_end:
    axA.annotate(f"w={w[i]:.1f}", (pf[i, 0], pf[i, 1]), fontsize=9, color="#e0556f",
                 xytext=(6, 6), textcoords="offset points")
cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.04); cb.set_label("density weight w = 8*(r_k/mean)")
axA.set_title("(A) density weight per PF point  (k=2 nearest-neighbor)\n"
              f"sparse ENDS get large w  |  end/mid weight ratio = {w_end/w_mid:.1f}x", fontsize=11)
axA.set_xlabel("cost (norm)"); axA.set_ylabel("wait (norm)"); axA.grid(alpha=0.3)

# B: cost順の r_k と w, 手動帯の被覆ズレ
xb = pf[order, 0]
axB.bar(np.arange(len(order)), w[order], color="#888", width=0.9, label="density weight w")
for name, lo, hi, col in bands:
    axB.axvspan(np.searchsorted(xb, lo), np.searchsorted(xb, hi) - 0.0, color=col, alpha=0.18)
    axB.text((np.searchsorted(xb, lo) + np.searchsorted(xb, hi)) / 2, w.max() * 0.92, name,
             ha="center", fontsize=7, color=col)
axB.axvspan(np.searchsorted(xb, 0.38), len(order), color="#e0556f", alpha=0.10)
axB.text(np.searchsorted(xb, 0.38) + (len(order) - np.searchsorted(xb, 0.38)) / 2, w.max() * 0.55,
         f"manual bands MISS this\n({uncovered_frac*100:.0f}% of points,\nincl. cost-end)", ha="center", fontsize=8, color="#e0556f")
axB.set_title("(B) density w along cost-sorted PF vs manual %-bands\n"
              "manual bands are fixed [0,0.38]; the sparse high-cost end is uncovered", fontsize=11)
axB.set_xlabel("PF point index (cost ascending)"); axB.set_ylabel("density weight w")
axB.grid(alpha=0.3, axis="y"); axB.legend(fontsize=8, loc="upper right")

fig.suptitle("(2) distance-based density weight (rigorous)  w = WEIGHT*(r_k/mean)^ALPHA, k=2  |  pcn_agent.py:2449-2461",
             fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("docs/figures/impl_density.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/impl_density.png")
print(f"w_end={w_end:.2f} w_mid={w_mid:.2f} ratio={w_end/w_mid:.2f} uncovered_by_manual={uncovered_frac:.3f}")
