#!/usr/bin/env python
"""512 trace 再現性: results/eval_pf/truepf_trace512_repro{i}_s0.npz を集めて
  左 : 全本の達成greedy PF を共通真PFに重ね描き(広さ=端到達+中間充填が一目で分かる)
  右上: 本ごとの HV(共通真PF比)バー + 平均±ばらつき
  右下: 本ごとのコスト10分位占有(中間が埋まっているか=bipolarでないか)
再現性の判定材料(HV平均/分散, 最安角到達, 充填分位)を表で print。
出力: docs/figures/pf_512_repro.png
usage: SCALE=512 .venv/bin/python scripts/plot_repro512.py
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

SCALE = os.environ.get("SCALE", "512")
OUT = os.environ.get("OUT", "docs/figures/pf_512_repro.png")
PAT = os.environ.get("PAT", f"results/eval_pf/truepf_trace{SCALE}_repro*_s0.npz")


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


files = sorted(glob.glob(PAT), key=lambda s: int(re.search(r"repro(\d+)", s).group(1)))
if not files:
    raise SystemExit(f"no files match {PAT}")
runs = {}
for fn in files:
    i = int(re.search(r"repro(\d+)", fn).group(1))
    d = np.load(fn)
    runs[i] = (d["greedy_0"], d["rp_0"])

# 共通真PF基準: 全本 greedy + rp の和集合から非劣解を作り、共通参照点でHVを正規化
allpts = np.vstack([np.vstack([g, r]) for (g, r) in runs.values()])
cmax = max(r[:, 0].max() for (_, r) in runs.values())
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf = allpts[get_non_dominated_inds_minimize(allpts)]
pf = pf[np.argsort(pf[:, 0])]
hvt = hv2d(pf, ref)

rows = []
for i, (g, r) in runs.items():
    hv = hv2d(g, ref) / hvt if hvt > 0 else 0.0
    cf = g[:, 0] / cmax
    occ = np.histogram(cf, bins=10, range=(0, 1))[0]
    filled = int((occ > 0).sum())
    # bipolar = 端2分位以外がほぼ空(中央8分位の占有<=1)
    bipolar = int((occ[1:9] > 0).sum()) <= 1
    rows.append(dict(run=i, hv=hv, reach=float(cf.min()), span=float(cf.max() - cf.min()),
                     filled=filled, bipolar=bipolar, occ=occ))

hvs = np.array([r["hv"] for r in rows])
fig = plt.figure(figsize=(15, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1], wspace=0.24, hspace=0.42)

# ---- 左: PF 重ね描き ----
axL = fig.add_subplot(gs[:, 0])
axL.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#188038", lw=2.4, label="union true PF", zorder=2)
cmap = plt.cm.viridis(np.linspace(0, 0.85, len(rows)))
for r, col in zip(rows, cmap):
    g = runs[r["run"]][0]
    o = np.argsort(g[:, 0])
    axL.plot(g[o, 0] / cmax, g[o, 1], "-o", c=col, lw=1.1, ms=4, alpha=.85,
             label=f"repro{r['run']} (HV {r['hv']:.0%}, fill {r['filled']}/10{'  BIPOLAR' if r['bipolar'] else ''})")
axL.axvspan(0, 0.10, color="#d7ecd9", alpha=.6, zorder=0)
axL.set_xlim(-0.03, 1.03)
axL.set_xlabel("Cost  (fraction of all-cloud;  0 = all on-prem = cheapest)")
axL.set_ylabel("Average wait time")
axL.set_title(f"trace n_jobs={SCALE}: achieved Pareto fronts across {len(rows)} independent runs",
              fontsize=12.5, fontweight="bold")
axL.legend(fontsize=8.6, loc="upper right")
axL.grid(alpha=.3)

# ---- 右上: HV バー ----
axT = fig.add_subplot(gs[0, 1])
xs = [r["run"] for r in rows]
axT.bar(xs, hvs * 100, color=["#d93025" if r["bipolar"] else "#1a73e8" for r in rows], edgecolor="#333")
axT.axhline(hvs.mean() * 100, color="#188038", ls="--", lw=1.6, label=f"mean {hvs.mean():.0%}")
axT.fill_between([min(xs) - .5, max(xs) + .5], (hvs.mean() - hvs.std()) * 100, (hvs.mean() + hvs.std()) * 100,
                 color="#188038", alpha=.12)
for r in rows:
    axT.text(r["run"], r["hv"] * 100 + 1.5, f"{r['hv']:.0%}", ha="center", fontsize=8.5)
axT.set_xticks(xs)
axT.set_ylim(0, 109)
axT.set_ylabel("greedy HV (% of true PF)")
axT.set_title(f"reproducibility of PF breadth   (mean {hvs.mean():.0%} ± {hvs.std():.0%}, "
              f"min {hvs.min():.0%}–max {hvs.max():.0%})", fontsize=10.5, fontweight="bold")
axT.legend(fontsize=8.5, loc="lower right")
axT.grid(axis="y", alpha=.3)

# ---- 右下: 分位占有ヒートマップ ----
axB = fig.add_subplot(gs[1, 1])
M = np.array([r["occ"] for r in rows])
im = axB.imshow(M, aspect="auto", cmap="Blues", vmin=0)
axB.set_yticks(range(len(rows))); axB.set_yticklabels([f"repro{r['run']}" for r in rows], fontsize=8.5)
axB.set_xticks(range(10)); axB.set_xticklabels([f"{d*10}%" for d in range(10)], fontsize=7.5)
axB.set_xlabel("cost decile (0%=cheap … 90%=expensive); cell=#greedy pts")
axB.set_title("middle-filling per run  (empty mid = bipolar)", fontsize=10.5, fontweight="bold")
for yi, r in enumerate(rows):
    for xi, v in enumerate(r["occ"]):
        if v:
            axB.text(xi, yi, int(v), ha="center", va="center", fontsize=7,
                     color="white" if v > M.max() * 0.6 else "#333")

n_good = int(((hvs >= 0.6) & np.array([not r["bipolar"] for r in rows])).sum())
fig.suptitle(f"512 trace PCN — is wide-PF reproducible?   "
             f"{n_good}/{len(rows)} runs both HV≥60% and middle-filled", fontsize=12.5, y=1.005, fontweight="bold")
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("saved", OUT)
print(f"\n{'run':>4} {'HV%':>6} {'reach':>7} {'span':>6} {'fill/10':>8} {'bipolar':>8}")
for r in rows:
    print(f"{r['run']:>4} {r['hv']*100:>5.0f}% {r['reach']:>7.3f} {r['span']:>6.2f} {r['filled']:>6}/10 {str(r['bipolar']):>8}")
print(f"\nHV mean={hvs.mean():.1%} std={hvs.std():.1%} min={hvs.min():.1%} max={hvs.max():.1%}  "
      f"CV={hvs.std()/max(hvs.mean(),1e-9):.2f}")
print(f"wide&filled runs: {n_good}/{len(rows)}")
