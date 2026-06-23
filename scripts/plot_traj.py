#!/usr/bin/env python
"""warm-start / baseline の iter別ckpt評価から「効率の時系列」を描く。
early-stop判定: 離脱が単調劣化(=離脱前ベスト時点を拾える→early-stopで救える) か、非単調(=救えない) か。
参照: baseline(repro final) + warm(final) の全点 union から共通真PF/cmax/ref。
入力: results/eval_pf/truepf_trace512_{tag}{i}_iter{X}_s0.npz  (tag=warm|repro)
出力: docs/figures/pf_512_traj_{tag}.png, print表
usage: TAG=warm PYTHONPATH=. .venv/bin/python scripts/plot_traj.py
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

TAG = os.environ.get("TAG", "warm")
SCALE = os.environ.get("SCALE", "512")


def hv2d(pts, ref):
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0; prev = ref[0]
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev - c) * (ref[1] - w); prev = c
    return hv


# 共通土俵: baseline + warm の final 全点
finals = glob.glob(f"results/eval_pf/truepf_trace{SCALE}_repro*_s0.npz") + \
         [f for f in glob.glob(f"results/eval_pf/truepf_trace{SCALE}_warm*_s0.npz") if "_iter" not in f]
allpts = np.vstack([np.vstack([np.load(f)["greedy_0"], np.load(f)["rp_0"]]) for f in finals])
cmax = max(np.load(f)["rp_0"][:, 0].max() for f in finals)
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf = allpts[get_non_dominated_inds_minimize(allpts)]
hvt = hv2d(pf, ref)


def wat(g, q):
    f = g[get_non_dominated_inds_minimize(g)]; f = f[np.argsort(f[:, 0])]
    return float(np.interp(q * cmax, f[:, 0], f[:, 1]))


# iter別 npz を run ごとに収集
traj = {}
for fn in glob.glob(f"results/eval_pf/truepf_trace{SCALE}_{TAG}*_iter*_s0.npz"):
    m = re.search(rf"{TAG}(\d+)_iter(\d+)", fn)
    if not m:
        continue
    i, it = int(m.group(1)), int(m.group(2))
    g = np.load(fn)["greedy_0"]
    traj.setdefault(i, []).append((it, hv2d(g, ref) / hvt if hvt > 0 else 0.0, wat(g, 0.25), wat(g, 0.10)))
for i in traj:
    traj[i].sort()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
cmap = plt.cm.tab10(np.linspace(0, 1, 10))
print(f"\n[{TAG}] iter別 HV%(共通真PF比) / wait@q0.25 / wait@q0.1")
for i in sorted(traj):
    arr = np.array([(it, hv, w25, w10) for it, hv, w25, w10 in traj[i]])
    its, hvs, w25, w10 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    ax1.plot(its, hvs * 100, "-o", c=cmap[i], lw=1.8, ms=5, label=f"{TAG}{i} (fin {hvs[-1]:.0%})")
    ax2.plot(its, w25, "-o", c=cmap[i], lw=1.8, ms=5, label=f"{TAG}{i}")
    peak = its[int(np.argmax(hvs))]
    print(f" {TAG}{i}: " + " ".join(f"it{int(it)}={hv:.0%}" for it, hv, _, _ in traj[i]) +
          f"  | HVpeak@it{int(peak)}={hvs.max():.0%} final={hvs[-1]:.0%} "
          f"{'(単調劣化:peak=最初)' if peak==its[0] and hvs[-1]<hvs[0]-0.05 else '(非単調/維持)'}")
ax1.axhline(80, color="#d93025", ls=":", lw=1.3, label="efficient (HV80%)")
ax1.set_xlabel("Phase3 iteration"); ax1.set_ylabel("greedy HV (% true PF)")
ax1.set_title(f"{TAG}: efficiency over training (does it drift off rep4's basin?)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(alpha=.3); ax1.set_ylim(0, 105)
ax2.set_xlabel("Phase3 iteration"); ax2.set_ylabel("wait @ cost q=0.25 (lower=better)")
ax2.set_title(f"{TAG}: knee wait over training", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(alpha=.3)
out = f"docs/figures/pf_512_traj_{TAG}.png"
fig.savefig(out, dpi=125, bbox_inches="tight")
print("saved", out)
