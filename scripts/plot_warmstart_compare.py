#!/usr/bin/env python
"""/goal 実験A′(warm-start蒸留) の判定: baseline(repro) 5本 vs warm-start 5本 を同一土俵で比較。
共通真PF/参照点 = 全10本(repro+warm)の greedy+rp の union から作成 → 両群を同じ基準でHV/効率指標化。
主判定:
  - 広さ: HV%(共通真PF比), fill/10, bipolar
  - 効率: wait@cost分位(q=0.1/0.25/0.5)。深い膝に届くほど小さい(rep4基準)。
  - 再現性: 群内の HV mean/std/CV、効率run本数(HV>=0.8 を「効率かつ広い」代理)。
出力: docs/figures/pf_512_warmstart.png, print表
usage: PYTHONPATH=. .venv/bin/python scripts/plot_warmstart_compare.py
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
OUT = os.environ.get("OUT", "docs/figures/pf_512_warmstart.png")


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


def load_group(tag):
    runs = {}
    for fn in sorted(glob.glob(f"results/eval_pf/truepf_trace{SCALE}_{tag}*_s0.npz"),
                     key=lambda s: int(re.search(rf"{tag}(\d+)", s).group(1))):
        i = int(re.search(rf"{tag}(\d+)", fn).group(1))
        d = np.load(fn)
        runs[i] = (d["greedy_0"], d["rp_0"])
    return runs


base = load_group("repro")
warm = load_group("warm")
assert base and warm, f"missing npz: base={len(base)} warm={len(warm)}"

# 共通土俵: 全点 union
allpts = np.vstack([np.vstack([g, r]) for grp in (base, warm) for (g, r) in grp.values()])
cmax = max(r[:, 0].max() for grp in (base, warm) for (_, r) in grp.values())
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf = allpts[get_non_dominated_inds_minimize(allpts)]
pf = pf[np.argsort(pf[:, 0])]
hvt = hv2d(pf, ref)


def wait_at(g, q):
    f = g[get_non_dominated_inds_minimize(g)]
    f = f[np.argsort(f[:, 0])]
    return float(np.interp(q * cmax, f[:, 0], f[:, 1]))


def summarize(runs):
    rows = []
    for i, (g, r) in runs.items():
        cf = g[:, 0] / cmax
        occ = np.histogram(cf, bins=10, range=(0, 1))[0]
        rows.append(dict(
            run=i, hv=hv2d(g, ref) / hvt if hvt > 0 else 0.0,
            reach=float(cf.min()), span=float(cf.max() - cf.min()),
            filled=int((occ > 0).sum()), bipolar=int((occ[1:9] > 0).sum()) <= 1,
            w10=wait_at(g, 0.10), w25=wait_at(g, 0.25), w50=wait_at(g, 0.50), occ=occ))
    return rows


rb, rw = summarize(base), summarize(warm)


def grp_stats(rows, name):
    hv = np.array([x["hv"] for x in rows])
    eff = int((hv >= 0.8).sum())
    print(f"\n===== {name} (n={len(rows)}) =====")
    print(f"{'run':>4} {'HV%':>5} {'fill':>5} {'bipo':>5} {'w@.1':>8} {'w@.25':>8} {'w@.5':>8}")
    for x in rows:
        print(f"{x['run']:>4} {x['hv']*100:>4.0f}% {x['filled']:>3}/10 {str(x['bipolar'])[0]:>5} "
              f"{x['w10']:>8.0f} {x['w25']:>8.0f} {x['w50']:>8.0f}")
    print(f"HV mean={hv.mean():.1%} std={hv.std():.1%} min={hv.min():.1%} max={hv.max():.1%} "
          f"CV={hv.std()/max(hv.mean(),1e-9):.2f} | 効率run(HV>=80%)={eff}/{len(rows)}")
    w25 = np.array([x["w25"] for x in rows])
    print(f"wait@q0.25 mean={w25.mean():.0f} std={w25.std():.0f} min={w25.min():.0f} max={w25.max():.0f}")
    return hv, eff


hvb, effb = grp_stats(rb, "baseline (repro)")
hvw, effw = grp_stats(rw, "warm-start (rep4蒸留)")

# ---- 図 ----
fig = plt.figure(figsize=(16, 6.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0], height_ratios=[1, 1], wspace=0.3, hspace=0.45)

axL = fig.add_subplot(gs[:, 0])
axL.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#188038", lw=2.6, label="union true PF (10 runs)", zorder=3)
for x in rb:
    g = base[x["run"]][0]; o = np.argsort(g[:, 0])
    axL.plot(g[o, 0] / cmax, g[o, 1], "-", c="#c0c4cc", lw=1.0, alpha=.7, zorder=1,
             label="baseline runs" if x["run"] == rb[0]["run"] else None)
for x in rw:
    g = warm[x["run"]][0]; o = np.argsort(g[:, 0])
    axL.plot(g[o, 0] / cmax, g[o, 1], "-o", c="#1a73e8", lw=1.2, ms=3, alpha=.8, zorder=2,
             label="warm-start runs" if x["run"] == rw[0]["run"] else None)
axL.axvspan(0, 0.10, color="#d7ecd9", alpha=.6, zorder=0)
axL.set_xlim(-0.03, 1.03)
axL.set_xlabel("Cost  (fraction of all-cloud;  0 = cheapest)")
axL.set_ylabel("Average wait time")
axL.set_title(f"trace {SCALE}: baseline(grey) vs warm-start(blue) achieved PFs", fontsize=12, fontweight="bold")
axL.legend(fontsize=8.6, loc="upper right")
axL.grid(alpha=.3)

axT = fig.add_subplot(gs[0, 1:])
xs = np.arange(5)
axT.bar(xs - 0.2, sorted([x["hv"] for x in rb], reverse=True), width=0.38, color="#c0c4cc", edgecolor="#333", label=f"baseline (mean {hvb.mean():.0%})")
axT.bar(xs + 0.2, sorted([x["hv"] for x in rw], reverse=True), width=0.38, color="#1a73e8", edgecolor="#333", label=f"warm-start (mean {hvw.mean():.0%})")
axT.axhline(0.8, color="#d93025", ls=":", lw=1.4, label="efficient threshold (HV80%)")
axT.set_xticks(xs); axT.set_xticklabels([f"#{i+1}" for i in xs])
axT.set_ylim(0, 1.09); axT.set_ylabel("greedy HV (frac of true PF)")
axT.set_title(f"HV reproducibility: efficient runs {effb}/5 → {effw}/5", fontsize=10.5, fontweight="bold")
axT.legend(fontsize=8.3, loc="lower left"); axT.grid(axis="y", alpha=.3)

axB = fig.add_subplot(gs[1, 1:])
for tag, rows, col in [("baseline", rb, "#c0c4cc"), ("warm-start", rw, "#1a73e8")]:
    w25 = np.array([x["w25"] for x in rows])
    parts = axB.boxplot(w25, positions=[0 if tag == "baseline" else 1], widths=0.5,
                        patch_artist=True, labels=[tag])
    for p in parts["boxes"]:
        p.set_facecolor(col)
axB.set_ylabel("wait @ cost q=0.25")
axB.set_title("efficiency at the knee (lower=better; deep knee≈rep4)", fontsize=10.5, fontweight="bold")
axB.grid(axis="y", alpha=.3)

fig.suptitle(f"512 trace — does rep4 warm-start make efficiency reproducible?   "
             f"efficient runs {effb}/5 → {effw}/5", fontsize=12.5, y=1.01, fontweight="bold")
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("\nsaved", OUT)
