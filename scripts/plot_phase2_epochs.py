#!/usr/bin/env python
"""Phase2 学習量(SUPERVISED_EPOCHS) sweep の結果を可視化。
truepf_trace{SCALE}_p2e{E}_s0.npz を自動検出。左=HV vs epochs 曲線、右=PF重ね描き。
HV は全 epoch 条件の和集合から作る共通真PF比(条件間で公平)。
出力: docs/figures/pf_phase2_epochs.png
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

SCALE = int(os.environ.get("SCALE", "64"))


def load(fn):
    d = np.load(fn)
    return d["greedy_0"], d["samp_0"], d["rp_0"]


def hv2d(pts, ref):
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0
    prev = ref[0]
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev - c) * (ref[1] - w)
        prev = c
    return hv


found = []
for fn in os.listdir("."):
    m = re.match(rf"truepf_trace{SCALE}_p2e(\d+)_s0\.npz$", fn)
    if m:
        found.append((int(m.group(1)), fn))
found.sort()
if not found:
    raise SystemExit(f"no truepf_trace{SCALE}_p2e*_s0.npz found")

data = {e: load(fn) for e, fn in found}
allpts = np.vstack([np.vstack([g, s, r]) for (g, s, r) in data.values()])
cmax = max(r[:, 0].max() for (_, _, r) in data.values())
nd = get_non_dominated_inds_minimize(allpts)
pf = allpts[nd]
pf = pf[np.argsort(pf[:, 0])]
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
hvt = hv2d(pf, ref)

epochs = [e for e, _ in found]
hvs = [hv2d(data[e][0], ref) / hvt if hvt > 0 else 0.0 for e in epochs]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))
# 左: HV vs epochs
axL.plot(epochs, [h * 100 for h in hvs], "-o", c="#1a73e8", lw=2, ms=8)
for e, h in zip(epochs, hvs):
    axL.annotate(f"{h:.0%}", (e, h * 100), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
axL.axvline(50, color="#999", ls="--", lw=1)
axL.text(50, axL.get_ylim()[0], " recipe default 50", color="#666", fontsize=8, va="bottom")
axL.set_xlabel("Phase2 supervised epochs (DISTRIBUTED_PCN_SUPERVISED_EPOCHS)")
axL.set_ylabel("greedy hypervolume  (% of true PF)")
axL.set_title(f"trace n_jobs={SCALE}: does the AMOUNT of Phase2 matter?", fontsize=12, fontweight="bold")
axL.grid(alpha=.3)
# 右: PF 重ね描き
cmap = plt.cm.viridis(np.linspace(0, 0.92, len(epochs)))
axR.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#2ca02c", lw=2.2, label="true PF", zorder=2)
for (e, col) in zip(epochs, cmap):
    g = data[e][0]
    order = np.argsort(g[:, 0])
    axR.plot(g[order, 0] / cmax, g[order, 1], "-o", c=col, lw=1, ms=4, alpha=.8,
             label=f"epochs={e} (HV {hv2d(g,ref)/hvt:.0%})", zorder=3)
axR.axvspan(0, 0.12, color="#cfe8cf", alpha=.5, zorder=0)
axR.set_xlim(-0.03, 1.03)
axR.set_xlabel("Cost  (fraction of all-cloud; 0 = all on-prem)")
axR.set_ylabel("Average wait time")
axR.set_title("achieved fronts by Phase2 epochs", fontsize=12, fontweight="bold")
axR.legend(fontsize=8.5, loc="upper right")
axR.grid(alpha=.3)
fig.suptitle("Phase2 training AMOUNT sweep (count-norm OFF, seed ON fixed; only SUPERVISED_EPOCHS varies)", fontsize=10, y=1.01)
fig.tight_layout()
out = "docs/figures/pf_phase2_epochs.png"
fig.savefig(out, dpi=125, bbox_inches="tight")
print("saved", out)
print(f"\ntrace n_jobs={SCALE}: HV vs Phase2 epochs")
for e, h in zip(epochs, hvs):
    print(f"  epochs={e:4d}:  HV={h:.0%}")
