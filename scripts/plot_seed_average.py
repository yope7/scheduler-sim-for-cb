#!/usr/bin/env python3
"""複数シード(run-to-run)の崩壊カーブを config ごとに平均し、mean±std 帯で比較する。
   各 config は collapse_{cfg}_synth{NJ}.npz と collapse_{cfg}_r*_synth{NJ}.npz を 1 シードずつ束ねる。
   崩壊は時刻ベース seed で確率的なので、単発比較ではなく分布で語るための図。
   usage: NJ=24 CONFIGS=baseline,frozen_sens_nanskip OUT=seed_avg.png python scripts/plot_seed_average.py"""
import os, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NJ = os.environ.get("NJ", "24")
CONFIGS = [c for c in os.environ.get("CONFIGS", "").split(",") if c]
OUT = os.environ.get("OUT", f"seed_avg_synth{NJ}.png")
COLORS = {"baseline": "#999999", "frozen": "#1a73e8", "frozen_sens": "#2ca02c",
          "frozen_sens_nanskip": "#d62728"}
PAL = ["#1a73e8", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

def load_seeds(cfg):
    files = [f"collapse_{cfg}_synth{NJ}.npz"] + sorted(
        glob.glob(f"collapse_{cfg}_r*_synth{NJ}.npz"))
    curves = []
    for f in files:
        if os.path.exists(f):
            a = np.load(f)["curve"]
            curves.append({int(r[0]): (r[1], r[2]) for r in a})  # iter -> (F, span)
    return curves

fig, (axF, axS) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
rows = []
for i, cfg in enumerate(CONFIGS):
    seeds = load_seeds(cfg)
    if not seeds:
        print(f"{cfg}: no curves found"); continue
    iters = sorted(set().union(*[set(s) for s in seeds]))
    Fm, Fs, Sm, Ss = [], [], [], []
    for it in iters:
        Fv = [s[it][0] for s in seeds if it in s]
        Sv = [s[it][1] for s in seeds if it in s]
        Fm.append(np.mean(Fv)); Fs.append(np.std(Fv))
        Sm.append(np.mean(Sv)); Ss.append(np.std(Sv))
    iters = np.array(iters); Fm = np.array(Fm); Fs = np.array(Fs); Sm = np.array(Sm); Ss = np.array(Ss)
    c = COLORS.get(cfg, PAL[i % len(PAL)])
    axF.plot(iters, Fm, "-o", color=c, lw=2.4, ms=5, label=f"{cfg} (n={len(seeds)})")
    axF.fill_between(iters, Fm - Fs, Fm + Fs, color=c, alpha=0.18)
    axS.plot(iters, Sm, "-s", color=c, lw=2.4, ms=5, label=f"{cfg} (n={len(seeds)})")
    axS.fill_between(iters, Sm - Ss, Sm + Ss, color=c, alpha=0.18)
    auc = float(np.trapz(np.clip(Fm, 0, 1), iters))
    late = Fm[iters >= 80]
    rows.append((cfg, len(seeds), auc, float(np.mean(late)), float(np.mean(Sm))))

axF.axhline(0.8, color="#444", ls=":", lw=1.2); axF.axhline(0.0, color="#bbb", lw=0.8)
axF.set_ylabel("F = corr(commanded, achieved cost)"); axF.set_ylim(-0.9, 1.08)
axF.grid(alpha=0.3); axF.legend(loc="lower left", fontsize=9)
axF.set_title(f"Seed-averaged collapse curves (mean±std) — synth n_jobs={NJ}\n"
              "stochastic collapse: judge by the averaged band, not single runs")
axS.set_ylabel("span = achieved / commanded cost range"); axS.set_xlabel("training iteration")
axS.set_ylim(-0.02, 1.0); axS.grid(alpha=0.3); axS.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")

print(f"saved {OUT}")
print(f"{'config':24s} {'n':>2s} {'AUC(F+)':>8s} {'meanF[>=80]':>11s} {'mean_span':>9s}")
for cfg, n, auc, lateF, ms in rows:
    print(f"{cfg:24s} {n:>2d} {auc:>8.1f} {lateF:>+11.2f} {ms:>9.3f}")
