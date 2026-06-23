#!/usr/bin/env python3
"""既存の Phase A 裾スイープ(run_synth512_tailL{00..10}512_{1..5}, 30run, 再学習なし)を、
各裾レベルの真PF nsga2_synthL0x_s0.npz で正規化した HV比(スケールフリー[0,1])で分析。
裾→追従カーブ(mean±std over 5seed)を描き、「崖」の有無を判定する。
出力: docs/figures/tail_sweep_curve.png
"""
import os
import glob
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents.pcn_agent import get_non_dominated_inds_minimize

LEVELS = ["00", "02", "04", "06", "08", "10"]
LEVELF = {"00": 0.0, "02": 0.2, "04": 0.4, "06": 0.6, "08": 0.8, "10": 1.0}


def hv2d(front, ref):
    """2D minimize HV (cost,wait), reference=(ref_c,ref_w)。front=Nx2。"""
    f = np.asarray(front, float)
    f = f[(f[:, 0] < ref[0]) & (f[:, 1] < ref[1])]
    if not len(f):
        return 0.0
    nd = get_non_dominated_inds_minimize(f)
    f = f[nd]
    f = f[np.argsort(f[:, 0])]
    hv = 0.0
    prev_c = ref[0]
    # sweep from high cost to low
    for c, w in sorted(f.tolist(), key=lambda x: -x[0]):
        hv += (prev_c - c) * (ref[1] - w)
        prev_c = c
    return hv


def final_eval_pf(run_dir):
    p = os.path.join(run_dir, "pcn_mo_hv.json")
    with open(p) as f:
        d = json.load(f)
    pfs = d.get("pareto_fronts_per_eval", [])
    if not pfs:
        return None
    return np.asarray(pfs[-1], float)


rows = {}
for L in LEVELS:
    tp = f"results/eval_pf/nsga2_synthL{L}_s0.npz"
    if not os.path.exists(tp):
        print(f"L{L}: true PF missing")
        continue
    true_pf = np.asarray(np.load(tp, allow_pickle=True)["pf"], float)
    ref = (true_pf[:, 0].max() * 1.05, true_pf[:, 1].max() * 1.05)
    true_hv = hv2d(true_pf, ref)
    ratios = []
    cmaxes = []
    for run in sorted(glob.glob(f"experiments/distributed_pcn/run_synth512_tailL{L}512_*")):
        subs = sorted(glob.glob(run + "/2026*"))
        if not subs:
            continue
        pf = final_eval_pf(subs[-1])
        if pf is None:
            continue
        r = hv2d(pf, ref) / true_hv if true_hv > 0 else 0.0
        ratios.append(r)
        cmaxes.append(pf[:, 0].max())
    rows[L] = dict(
        ratios=ratios,
        mean=float(np.mean(ratios)) if ratios else 0.0,
        std=float(np.std(ratios)) if ratios else 0.0,
        true_cmax=float(true_pf[:, 0].max()),
        ach_cmax=float(np.mean(cmaxes)) if cmaxes else 0.0,
    )
    print(
        f"L{L} (frac{LEVELF[L]}): HVratio mean={rows[L]['mean']:.3f} std={rows[L]['std']:.3f} "
        f"n={len(ratios)} | trueCmax={rows[L]['true_cmax']:.3g} achCmax={rows[L]['ach_cmax']:.3g} "
        f"reach={rows[L]['ach_cmax']/max(1,rows[L]['true_cmax']):.2f} | ratios={[round(x,2) for x in ratios]}"
    )

xs = [LEVELF[L] for L in LEVELS if L in rows]
means = [rows[L]["mean"] for L in LEVELS if L in rows]
stds = [rows[L]["std"] for L in LEVELS if L in rows]
reach = [rows[L]["ach_cmax"] / max(1, rows[L]["true_cmax"]) for L in LEVELS if L in rows]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, lw=2, color="navy")
for L in LEVELS:
    if L in rows:
        x = LEVELF[L]
        ax.scatter([x] * len(rows[L]["ratios"]), rows[L]["ratios"], s=18, c="orange", alpha=0.6, zorder=5)
ax.set_xlabel("Tail level (SYNTH_TAIL_LEVEL)")
ax.set_ylabel("HV ratio  (achieved / true PF)")
ax.set_title("Phase A: tail -> following (HV-normalized, 5 seeds)")
ax.grid(alpha=0.3)
ax2 = axes[1]
ax2.plot(xs, reach, marker="s", lw=2, color="darkred")
ax2.axhline(1.0, ls="--", c="gray", alpha=0.6)
ax2.set_xlabel("Tail level")
ax2.set_ylabel("Cost reach  (achieved Cmax / true Cmax)")
ax2.set_title("expensive-end reach vs tail")
ax2.grid(alpha=0.3)
fig.tight_layout()
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/tail_sweep_curve.png", dpi=95)
print("[SAVED] docs/figures/tail_sweep_curve.png")
