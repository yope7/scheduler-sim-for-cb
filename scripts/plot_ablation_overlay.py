#!/usr/bin/env python3
"""全アブレーションの崩壊カーブ(collapse_*_synth{NJ}.npz)を1枚に重ねて比較する。
   F = corr(commanded cost, achieved cost) と span を上下2段で表示。
   usage: NJ=24 ORDER=baseline,frozen,... OUT=ablation_overlay.png python scripts/plot_ablation_overlay.py"""
import os, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NJ = os.environ.get("NJ", "24")
OUT = os.environ.get("OUT", f"ablation_overlay_synth{NJ}.png")
pat = re.compile(rf"collapse_(.+)_synth{NJ}\.npz$")

files = {}
for f in sorted(glob.glob(f"collapse_*_synth{NJ}.npz")):
    m = pat.search(f)
    if m:
        files[m.group(1)] = f

# 表示順（指定があれば優先、残りは名前順で後ろに）
order = [s for s in os.environ.get("ORDER", "").split(",") if s]
names = [n for n in order if n in files] + [n for n in sorted(files) if n not in order]

COLORS = ["#999999", "#1a73e8", "#d62728", "#2ca02c", "#9467bd",
          "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#bcbd22"]

fig, (axF, axS) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
summary = []
for i, n in enumerate(names):
    a = np.load(files[n])["curve"]
    it, F, span = a[:, 0], a[:, 1], a[:, 2]
    c = COLORS[i % len(COLORS)]
    lw = 2.6 if n == "baseline" else 2.0
    ls = "--" if n == "baseline" else "-"
    axF.plot(it, F, ls, color=c, lw=lw, marker="o", ms=4, label=n)
    axS.plot(it, span, ls, color=c, lw=lw, marker="s", ms=4, label=n)
    # 健全さの要約: F>=0.8 を保った最大iter, AUC(F), 終端F, 平均span
    healthy = it[F >= 0.8]
    last_healthy = int(healthy.max()) if len(healthy) else 0
    summary.append((n, last_healthy, float(np.trapz(np.clip(F, 0, 1), it)),
                    float(F[-1]), float(np.mean(span))))

axF.axhline(0.8, color="#444", ls=":", lw=1.2, label="healthy (0.8)")
axF.axhline(0.0, color="#bbb", ls="-", lw=0.8)
axF.set_ylabel("F = corr(commanded, achieved cost)")
axF.set_ylim(-0.9, 1.08); axF.grid(alpha=0.3); axF.legend(loc="lower left", fontsize=8, ncol=2)
axF.set_title(f"Ablation overlay — command-following fidelity over training (synth, n_jobs={NJ})\n"
              "high & flat = command followed throughout; dip<0 = inversion; →0 = command ignored")
axS.set_ylabel("span = achieved / commanded cost range")
axS.set_xlabel("training iteration"); axS.set_ylim(-0.02, 1.0); axS.grid(alpha=0.3)
axS.legend(loc="upper right", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")

print(f"saved {OUT}  ({len(names)} ablations)")
print(f"{'ablation':22s} {'F>=0.8_until':>12s} {'AUC(F+)':>8s} {'F_end':>7s} {'mean_span':>9s}")
for n, lh, auc, fend, ms in sorted(summary, key=lambda r: -r[1]):
    print(f"{n:22s} {lh:>12d} {auc:>8.1f} {fend:>+7.2f} {ms:>9.3f}")
