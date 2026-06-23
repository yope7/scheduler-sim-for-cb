#!/usr/bin/env python3
"""trace1024 完全16セルアブレーションの主効果図。(A)各機能のHV/追従 主効果(ON8 vs OFF8)、
(B)16セルのHVヒートマップ(F×W×E×Dを格子表示)。データ /tmp/ladrich_fdk*.json。
出力 docs/figures/fk1024_maineffect.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(F, W, E, D, i):
    p = f"/tmp/ladrich_fdk{F}{W}{E}{D}_{i}.json"
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1]); return d if "hv" in d else None
    except Exception:
        return None


cells = {}
for F in [0, 1]:
 for W in [0, 1]:
  for E in [0, 1]:
   for D in [0, 1]:
    hv = []; cd = []
    for i in [1, 2]:
        d = load(F, W, E, D, i)
        if d:
            hv.append(d["hv"]); cd.append(d["cmd_dist"])
    cells[(F, W, E, D)] = (np.mean(hv) if hv else np.nan, np.mean(cd) if cd else np.nan)

names = ["Fourier", "density-W", "explore", "defer(offset1)"]
hv_on = [np.mean([cells[k][0] for k in cells if k[idx] == 1]) for idx in range(4)]
hv_off = [np.mean([cells[k][0] for k in cells if k[idx] == 0]) for idx in range(4)]
cd_on = [np.mean([cells[k][1] for k in cells if k[idx] == 1]) for idx in range(4)]
cd_off = [np.mean([cells[k][1] for k in cells if k[idx] == 0]) for idx in range(4)]

fig, (axA, axC, axB) = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})

x = np.arange(4)
axA.bar(x - 0.2, hv_on, 0.4, label="ON", color="#5ee0a0")
axA.bar(x + 0.2, hv_off, 0.4, label="OFF", color="#9aa7b8")
for i in range(4):
    axA.text(i, max(hv_on[i], hv_off[i]) + 0.01, f"{hv_on[i]-hv_off[i]:+.3f}", ha="center", fontsize=9,
             color="#2a8f5f" if hv_on[i] > hv_off[i] else "#888")
axA.set_xticks(x); axA.set_xticklabels(names, rotation=20, fontsize=9, ha="right")
axA.set_title("(A) HV main effect (ON vs OFF)\n Fourier dominates HV", fontsize=11); axA.set_ylabel("HV"); axA.legend(fontsize=8); axA.grid(alpha=0.3, axis="y")

axC.bar(x - 0.2, cd_on, 0.4, label="ON", color="#4da3ff")
axC.bar(x + 0.2, cd_off, 0.4, label="OFF", color="#9aa7b8")
for i in range(4):
    axC.text(i, max(cd_on[i], cd_off[i]) + 0.01, f"{cd_on[i]-cd_off[i]:+.3f}", ha="center", fontsize=9,
             color="#1a73e8" if cd_on[i] < cd_off[i] else "#888")
axC.set_xticks(x); axC.set_xticklabels(names, rotation=20, fontsize=9, ha="right")
axC.set_title("(C) following-distance main effect\n defer(offset1) dominates (lower=better)", fontsize=11); axC.set_ylabel("following distance"); axC.legend(fontsize=8); axC.grid(alpha=0.3, axis="y")

# B: 16セルHVヒートマップ 行=F×W(4), 列=E×D(4)
grid = np.full((4, 4), np.nan)
rows = [(0, 0), (0, 1), (1, 0), (1, 1)]; colsED = [(0, 0), (0, 1), (1, 0), (1, 1)]
for r, (F, W) in enumerate(rows):
    for c, (E, D) in enumerate(colsED):
        grid[r, c] = cells[(F, W, E, D)][0]
im = axB.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=0.5)
axB.set_xticks(range(4)); axB.set_xticklabels(["E-/D-", "E-/D+", "E+/D-", "E+/D+"], fontsize=8)
axB.set_yticks(range(4)); axB.set_yticklabels(["F-/W-", "F-/W+", "F+/W-", "F+/W+"], fontsize=8)
for r in range(4):
    for c in range(4):
        axB.text(c, r, f"{grid[r,c]:.2f}", ha="center", va="center", color="w" if grid[r, c] < 0.3 else "k", fontsize=9)
axB.set_title("(B) HV heatmap of 16 cells\n top rows (F:ON) clearly brighter", fontsize=11)
fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label="HV")

fig.suptitle("trace1024 full 2^4 ablation: Fourier dominates HV / defer(offset1) dominates following  -  density & explore inert at 1024", fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("docs/figures/fk1024_maineffect.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/fk1024_maineffect.png")
