#!/usr/bin/env python
"""trace512 実験群の PF を横断集計し、NSGA-II の PF と重ねた総括図を作る。

出力 (docs/figures/):
  pf512_hv_overview.png   全グループ横断: greedy HV(共通真PF比) の mean±std 横バー
  pf512_gallery.png       新規グループのPF重ねギャラリー(5run + 真PF + NSGA-II)
  pf512_nsga2_compare.png NSGA-II vs PCN 最良/代表グループの詳細比較
  pf512_nsga2_convergence.png NSGA-II の世代収束(HV vs 世代/評価数)
共通真PF = 全グループ greedy ∪ rp ∪ NSGA-II PF の非劣解(これまでの plot_repro512.py 流儀の拡張)。

usage: PYTHONPATH=. .venv/bin/python scripts/plot_pf_summary_nsga2.py
"""
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.pcn_agent import get_non_dominated_inds_minimize

OUTDIR = os.environ.get("OUTDIR", "docs/figures")
NSGA_NPZ = os.environ.get("NSGA_NPZ", "results/eval_pf/nsga2_trace512_s0.npz")
# 既存実装の既定パラメータ (mut=0.1/bit) での比較走行。妥当性チェックの実証用
NSGA_NPZ_ALT = os.environ.get("NSGA_NPZ_ALT", "results/eval_pf/nsga2_trace512_s0_mut01.npz")
os.makedirs(OUTDIR, exist_ok=True)

# (表示名, npzパターン{i}=1..5, 短い説明, 新規グループか)
GROUPS = [
    ("repro", "truepf_trace512_repro{i}_s0.npz", "base repro (NITER=100)", False),
    ("es", "truepf_trace512_es{i}_s0.npz", "early-stop v1", False),
    ("esnew", "truepf_trace512_esnew{i}_s0.npz", "early-stop v2", False),
    ("estop", "truepf_trace512_estop{i}_s0.npz", "early-stop v3", False),
    ("warm", "truepf_trace512_warm{i}_s0.npz", "warm-start (repro4 ckpt)", False),
    ("anchor", "truepf_trace512_anchor{i}_s0.npz", "anchor weights LW=25 KNEE=18", False),
    ("seed", "truepf_trace512_seed{i}_s0.npz", "threshold seeding", False),
    ("a0_base", "truepf_trace512_a0_base{i}_s0.npz", "levers: base", False),
    ("a1_frozen", "truepf_trace512_a1_frozen{i}_s0.npz", "levers: frozen", False),
    ("a2_ema99", "truepf_trace512_a2_ema99{i}_s0.npz", "levers: EMA .99", False),
    ("a3_ema999", "truepf_trace512_a3_ema999{i}_s0.npz", "levers: EMA .999", False),
    ("a4_lrcos", "truepf_trace512_a4_lrcos{i}_s0.npz", "levers: LR cosine", False),
    ("a5_combo", "truepf_trace512_a5_combo{i}_s0.npz", "levers: combo", False),
    ("a6_lrlow", "truepf_trace512_a6_lrlow{i}_s0.npz", "levers: LR low", False),
    ("a7_lrema", "truepf_trace512_a7_lrema{i}_s0.npz", "levers: LR+EMA", False),
    ("final", "truepf_trace512_final512_{i}_s0.npz", "LR cosine + early-stop", True),
    ("b2", "truepf_trace512_b2512_{i}_s0.npz", "FiLM+Fourier B4", True),
    ("lin", "truepf_trace512_lin512512_{i}_s0.npz", "Fourier mode=linear", True),
    ("gau", "truepf_trace512_gau512512_{i}_s0.npz", "Fourier mode=gaussian", True),
    ("nup200", "truepf_trace512_nup200512_{i}_s0.npz", "b2 + N_UPDATES=200", True),
    ("dens", "truepf_trace512_dens512_{i}_s0.npz", "nup200 + PF density w8", True),
    ("dens2", "truepf_trace512_dens2512_{i}_s0.npz", "N_UPD300 + PF density w4", True),
    ("dens3", "truepf_trace512_dens3512_{i}_s0.npz", "N_UPD300 + PF weight 12", True),
    ("dens4", "truepf_trace512_dens4512_{i}_s0.npz", "dens3 + gaussian", True),
    ("dens5", "truepf_trace512_dens5512_{i}_s0.npz", "dens3 + low-band cond", True),
    ("sched", "truepf_trace512_sched512_{i}_s0.npz", "PF weight ramp (peak3, start .5)", True),
    ("schedlr", "truepf_trace512_schedlr512_{i}_s0.npz", "sched + LR cosine", True),
    ("spike", "truepf_trace512_spike512_{i}_s0.npz", "PF w12 + loss-spike skip", True),
]


def hv2d(pts, ref):
    if len(pts) == 0:
        return 0.0
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


# ---- 読み込み ----
runs = {}  # name -> {i: greedy}
rp_all = []
for name, pat, desc, is_new in GROUPS:
    g = {}
    for i in range(1, 6):
        fn = os.path.join("results/eval_pf", pat.format(i=i))
        if os.path.exists(fn):
            d = np.load(fn)
            g[i] = d["greedy_0"]
            rp_all.append(d["rp_0"])
    if g:
        runs[name] = g

def load_nsga(path):
    if not os.path.exists(path):
        return None
    nd = np.load(path, allow_pickle=True)
    return {"pf": nd["pf"], "gen_pf": nd["gen_pf"], "meta": dict((str(k), v) for k, v in nd["meta"])}


nsga = load_nsga(NSGA_NPZ)
nsga_alt = load_nsga(NSGA_NPZ_ALT)
if nsga is not None:
    print(f"NSGA-II: {NSGA_NPZ} PF {len(nsga['pf'])}点")
else:
    print(f"(NSGA npz {NSGA_NPZ} なし: PCN のみで作図)")

# ---- 共通真PF ----
parts = [np.vstack(list(g.values())) for g in runs.values()] + rp_all
if nsga is not None:
    parts.append(nsga["pf"])
allpts = np.vstack(parts)
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf_true = allpts[get_non_dominated_inds_minimize(allpts)]
pf_true = pf_true[np.argsort(pf_true[:, 0])]
hvt = hv2d(pf_true, ref)
cmax = allpts[:, 0].max()
print(f"共通真PF: {len(pf_true)}点 (全 {len(allpts)}点から) ref={ref}")

# ---- グループごとの HV ----
stats = {}
for name, pat, desc, is_new in GROUPS:
    if name not in runs:
        continue
    hvs = np.array([hv2d(g, ref) / hvt for g in runs[name].values()])
    stats[name] = dict(hvs=hvs, desc=desc, is_new=is_new, n=len(hvs))
hv_nsga = hv2d(nsga["pf"], ref) / hvt if nsga is not None else None

# ---- 図1: 横断 HV 横バー ----
names = [n for n, *_ in GROUPS if n in stats]
fig, ax = plt.subplots(figsize=(11, 0.42 * len(names) + 2.2))
ys = np.arange(len(names))[::-1]
for y, n in zip(ys, names):
    s = stats[n]
    col = "#1a73e8" if s["is_new"] else "#9aa0a6"
    ax.barh(y, s["hvs"].mean() * 100, xerr=s["hvs"].std() * 100, color=col, edgecolor="#333",
            height=0.62, error_kw=dict(lw=1.2, capsize=3))
    ax.scatter(s["hvs"] * 100, [y] * len(s["hvs"]), s=14, c="#202124", zorder=3, alpha=.75)
    ax.text(1.0, y, f"{n}  ({s['desc']})", va="center", fontsize=8, color="white" if s["hvs"].mean() > 0.18 else "#333")
    ax.text(s["hvs"].mean() * 100 + s["hvs"].std() * 100 + 1.2, y,
            f"{s['hvs'].mean():.0%}±{s['hvs'].std():.0%}", va="center", fontsize=8)
if hv_nsga is not None:
    ax.axvline(hv_nsga * 100, color="#d93025", lw=2.0, ls="--",
               label=f"NSGA-II {hv_nsga:.0%} (pop{nsga['meta']['pop']}×gen{nsga['meta']['gen']})")
    ax.legend(loc="lower right", fontsize=9)
ax.set_yticks([])
ax.set_xlabel("greedy HV (% of union true PF)")
ax.set_xlim(0, 104)
ax.set_title("trace512 (job_seed=0): PF quality (HV) across all groups — 5 independent runs each (dots = runs)\n"
             "gray = previously reported, blue = new (Jun 9-10), red dashed = NSGA-II (search baseline)",
             fontsize=10.5, fontweight="bold")
ax.grid(axis="x", alpha=.3)
fig.savefig(f"{OUTDIR}/pf512_hv_overview.png", dpi=125, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUTDIR}/pf512_hv_overview.png")

# ---- 図2: 新規グループのギャラリー ----
new_names = [n for n, *_ in GROUPS if n in stats and stats[n]["is_new"]]
ncol = 4
nrow = int(np.ceil(len(new_names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.3 * nrow), squeeze=False)
for k, n in enumerate(new_names):
    ax = axes[k // ncol][k % ncol]
    ax.plot(pf_true[:, 0] / cmax, pf_true[:, 1], "-", c="#188038", lw=2.0, label="union true PF", zorder=2)
    if nsga is not None:
        p = nsga["pf"]
        ax.plot(p[:, 0] / cmax, p[:, 1], "-", c="#d93025", lw=1.6, alpha=.9, label="NSGA-II", zorder=3)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, 5))
    for (i, g), col in zip(sorted(runs[n].items()), cmap):
        o = np.argsort(g[:, 0])
        ax.plot(g[o, 0] / cmax, g[o, 1], "-o", c=col, lw=0.9, ms=2.6, alpha=.8, zorder=4)
    s = stats[n]
    ax.set_title(f"{n}: {s['desc']}\nHV {s['hvs'].mean():.0%}±{s['hvs'].std():.0%} "
                 f"(min {s['hvs'].min():.0%})", fontsize=9.5)
    ax.grid(alpha=.3)
    ax.tick_params(labelsize=7.5)
    if k == 0:
        ax.legend(fontsize=7.5)
for k in range(len(new_names), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
fig.suptitle("trace512 new groups: achieved PF (5 runs) vs union true PF vs NSGA-II   "
             "(x = cost / all-cloud, y = average wait)", fontsize=12, fontweight="bold", y=1.002)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/pf512_gallery.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUTDIR}/pf512_gallery.png")

# ---- 図3 & 4: NSGA-II 詳細 ----
if nsga is not None:
    best = max(stats, key=lambda n: stats[n]["hvs"].mean())
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(pf_true[:, 0] / cmax, pf_true[:, 1], "-", c="#188038", lw=2.4, label=f"union true PF ({len(pf_true)} pts)")
    rp = rp_all[0]
    ax.scatter(rp[:, 0] / cmax, rp[:, 1], s=18, c="#9aa0a6", alpha=.6, label="random-p sweep (ref)")
    p = nsga["pf"]
    ax.plot(p[:, 0] / cmax, p[:, 1], "-s", c="#d93025", lw=1.8, ms=4,
            label=f"NSGA-II PF {len(p)} pts (HV {hv_nsga:.0%})")
    cmap = plt.cm.viridis(np.linspace(0, 0.85, 5))
    for (i, g), col in zip(sorted(runs[best].items()), cmap):
        o = np.argsort(g[:, 0])
        ax.plot(g[o, 0] / cmax, g[o, 1], "-o", c=col, lw=1.0, ms=3.4, alpha=.85,
                label=f"PCN {best} run{i} (HV {hv2d(g, ref)/hvt:.0%})")
    ax.set_xlabel("Cost (fraction of all-cloud; 0 = all on-prem = cheapest)")
    ax.set_ylabel("Average wait time")
    ax.set_title(f"NSGA-II vs best PCN group ({best}: {stats[best]['desc']}) — trace512 job_seed=0",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.4)
    ax.grid(alpha=.3)
    fig.savefig(f"{OUTDIR}/pf512_nsga2_compare.png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTDIR}/pf512_nsga2_compare.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for nz, col, tag in ((nsga, "#d93025", "mut=1/n (fixed)"), (nsga_alt, "#f9ab00", "mut=0.1/bit (original default)")):
        if nz is None:
            continue
        hv_gen = np.array([hv2d(np.asarray(g, dtype=float), ref) / hvt for g in nz["gen_pf"]])
        m = nz["meta"]
        ax.plot(np.arange(len(hv_gen)), hv_gen * 100, "-", c=col, lw=1.8,
                label=f"{tag}: final HV {hv_gen[-1]:.0%} ({int(m['n_evaluations'])} evals, "
                      f"{float(m['elapsed_sec']):.0f}s)")
        print(f"[convergence] {tag}: final HV {hv_gen[-1]:.1%}, gen10 {hv_gen[10]:.1%}, "
              f"gen50 {hv_gen[50]:.1%}, PF {len(nz['pf'])}点")
    bm = stats[best]["hvs"].mean()
    ax.axhline(bm * 100, color="#1a73e8", ls="--", lw=1.4, label=f"PCN {best} mean {bm:.0%}")
    ax.set_xlabel("generation")
    ax.set_ylabel("HV (% of union true PF)")
    meta = nsga["meta"]
    ax.set_title(f"NSGA-II convergence (pop={meta['pop']}, gen={meta['gen']}): "
                 f"mutation-rate sanity check", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.6)
    ax.grid(alpha=.3)
    fig.savefig(f"{OUTDIR}/pf512_nsga2_convergence.png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTDIR}/pf512_nsga2_convergence.png")

# ---- 表 print ----
print(f"\n{'group':>10} {'HV mean':>8} {'std':>6} {'min':>6} {'max':>6}  desc")
for n in names:
    s = stats[n]
    print(f"{n:>10} {s['hvs'].mean():>7.1%} {s['hvs'].std():>6.1%} {s['hvs'].min():>6.1%} "
          f"{s['hvs'].max():>6.1%}  {s['desc']}")
if hv_nsga is not None:
    print(f"{'NSGA-II':>10} {hv_nsga:>7.1%} {'—':>6} {'—':>6} {'—':>6}  探索ベースライン")
