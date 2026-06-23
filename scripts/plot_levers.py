#!/usr/bin/env python
"""Phase2 A/B集計: 劣化を学習で防ぐ3レバーの最終ckpt PF を群比較(early-stop OFF)。
results/eval_pf/truepf_trace512_{EXP}{i}_s0.npz を群として読み、共通union真PFで
HV平均/効率run(>=80%)/std を出す。a0_base が baseline(全レバーOFF)。
出力: docs/figures/pf_512_levers_ab.png
usage: PYTHONPATH=. .venv/bin/python scripts/plot_levers.py  (EXPS=a0_base,a1_frozen,... で群指定可)
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
EXPS = os.environ.get("EXPS", "a0_base,a1_frozen,a2_ema99,a3_ema999,a4_lrcos,a5_combo").split(",")


def hv2d(pts, ref):
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0; prev = ref[0]
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev - c) * (ref[1] - w); prev = c
    return hv


def load_group(exp):
    runs = {}
    for fn in sorted(glob.glob(f"results/eval_pf/truepf_trace{SCALE}_{exp}[0-9]_s0.npz"),
                     key=lambda s: int(re.search(rf"{exp}(\d+)_s0", s).group(1))):
        i = int(re.search(rf"{exp}(\d+)_s0", fn).group(1))
        d = np.load(fn)
        runs[i] = (d["greedy_0"], d["rp_0"])
    return runs


G = {e: load_group(e) for e in EXPS}
G = {e: v for e, v in G.items() if v}  # 存在する群のみ
if not G:
    raise SystemExit("no A/B npz found in results/eval_pf/")
allpts = np.vstack([np.vstack([g, r]) for grp in G.values() for (g, r) in grp.values()])
cmax = max(r[:, 0].max() for grp in G.values() for (_, r) in grp.values())
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf = allpts[get_non_dominated_inds_minimize(allpts)]
hvt = hv2d(pf, ref)


def wat(g, q):
    f = g[get_non_dominated_inds_minimize(g)]; f = f[np.argsort(f[:, 0])]
    return float(np.interp(q * cmax, f[:, 0], f[:, 1]))


stats = {}
print(f"\n{'EXP':>12} {'n':>2} {'HV mean':>8} {'std':>6} {'eff/n':>6} {'w@.25 med':>10}  per-run HV")
for e, grp in G.items():
    hv = np.array([hv2d(g, ref) / hvt for (g, _) in grp.values()])
    w25 = np.array([wat(g, 0.25) for (g, _) in grp.values()])
    stats[e] = dict(hv=hv, eff=int((hv >= 0.8).sum()), n=len(hv), w25med=np.median(w25))
    print(f"{e:>12} {len(hv):>2} {hv.mean():>7.0%} {hv.std():>5.0%} {int((hv>=0.8).sum())}/{len(hv):<3} "
          f"{np.median(w25):>10.0f}  " + " ".join(f"{x:.0%}" for x in sorted(hv, reverse=True)))

base = stats.get("a0_base")
if base is not None:
    print(f"\n=== vs baseline(a0_base mean={base['hv'].mean():.0%} eff={base['eff']}/{base['n']}) ===")
    for e, s in stats.items():
        if e == "a0_base":
            continue
        dh = (s["hv"].mean() - base["hv"].mean()) * 100
        print(f"  {e:>12}: HV {s['hv'].mean():.0%} (Δ{dh:+.0f}pt)  eff {s['eff']}/{s['n']}  std {s['hv'].std():.0%}")

# 図: 群別 平均HV(±std) バー + 効率run注記
names = list(stats.keys())
fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 5.2))
means = [stats[n]["hv"].mean() * 100 for n in names]
stds = [stats[n]["hv"].std() * 100 for n in names]
cols = ["#c0c4cc" if n == "a0_base" else "#1a73e8" for n in names]
ax.bar(range(len(names)), means, yerr=stds, color=cols, edgecolor="#333", capsize=5)
for i, n in enumerate(names):
    ax.text(i, means[i] + 2, f"{means[i]:.0f}%\neff{stats[n]['eff']}/{stats[n]['n']}", ha="center", fontsize=8.5)
if base is not None:
    ax.axhline(base["hv"].mean() * 100, color="#d93025", ls="--", lw=1.3, label=f"baseline {base['hv'].mean():.0%}")
    ax.legend(fontsize=9)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("mean final-ckpt HV (% true PF)"); ax.set_ylim(0, 100)
ax.set_title(f"trace {SCALE} — 劣化防止レバー (early-stop OFF, 最終ckpt): HV平均±std", fontsize=11.5, fontweight="bold")
ax.grid(axis="y", alpha=.3)
fig.savefig("docs/figures/pf_512_levers_ab.png", dpi=125, bbox_inches="tight")
print("\nsaved docs/figures/pf_512_levers_ab.png")
