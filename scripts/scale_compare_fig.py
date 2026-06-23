#!/usr/bin/env python3
"""トレース規模(256 vs 1024)での4機能の主効果比較。裾が重いほどフーリエ/deferが強まり、
密度は中規模(256)のみ効くことを視覚化。値は確定集計から(Ver.2 fd=256, fk1024=1024)。
出力 docs/figures/scale_compare.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

feat = ["Fourier", "density-W", "explore", "defer(offset1)"]
# HV main effect (ON - OFF)
hv256 = [+0.116, +0.019, +0.016, -0.013]
hv1024 = [+0.239, -0.007, -0.034, +0.051]
# following-distance main effect (ON - OFF), lower(more negative)=better
cd256 = [-0.072, -0.026, -0.030, -0.071]
cd1024 = [-0.068, +0.069, +0.011, -0.158]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(4)

axA.bar(x - 0.2, hv256, 0.4, label="trace256", color="#4da3ff")
axA.bar(x + 0.2, hv1024, 0.4, label="trace1024", color="#e0556f")
axA.axhline(0, color="#444", lw=0.8)
axA.set_xticks(x); axA.set_xticklabels(feat, rotation=15, fontsize=9, ha="right")
axA.set_title("(A) HV main effect (ON-OFF)\n Fourier strengthens with tail; density vanishes at 1024", fontsize=11)
axA.set_ylabel("ΔHV (higher=better)"); axA.legend(fontsize=9); axA.grid(alpha=0.3, axis="y")
for i in range(4):
    axA.text(i - 0.2, hv256[i] + (0.008 if hv256[i] >= 0 else -0.02), f"{hv256[i]:+.2f}", ha="center", fontsize=7)
    axA.text(i + 0.2, hv1024[i] + (0.008 if hv1024[i] >= 0 else -0.02), f"{hv1024[i]:+.2f}", ha="center", fontsize=7)

axB.bar(x - 0.2, cd256, 0.4, label="trace256", color="#4da3ff")
axB.bar(x + 0.2, cd1024, 0.4, label="trace1024", color="#e0556f")
axB.axhline(0, color="#444", lw=0.8)
axB.set_xticks(x); axB.set_xticklabels(feat, rotation=15, fontsize=9, ha="right")
axB.set_title("(B) following-distance main effect (ON-OFF)\n defer strengthens with tail (more negative=better)", fontsize=11)
axB.set_ylabel("Δfollowing (lower=better)"); axB.legend(fontsize=9); axB.grid(alpha=0.3, axis="y")
for i in range(4):
    axB.text(i - 0.2, cd256[i] + (0.005 if cd256[i] >= 0 else -0.015), f"{cd256[i]:+.2f}", ha="center", fontsize=7)
    axB.text(i + 0.2, cd1024[i] + (0.005 if cd1024[i] >= 0 else -0.015), f"{cd1024[i]:+.2f}", ha="center", fontsize=7)

fig.suptitle("Scale dependence of feature effects: Fourier(HV) & defer(following) dominate more as the tail gets heavier", fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("docs/figures/scale_compare.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/scale_compare.png")
