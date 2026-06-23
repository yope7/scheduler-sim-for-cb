#!/usr/bin/env python3
"""各レシピの seed 横断 full-span 信頼性を積み上げ棒で可視化。
   各 seed を late-half の F/span で FULL / compressed / inverted に分類。
   usage: python scripts/plot_reliability_summary.py"""
import glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# (label, [run basenames]) — 実験ごとの seed 群
RECIPES = [
    ("baseline\n(start)", ["baseline", "baseline_r2", "baseline_r3"]),
    ("fix\n(frozen+sens+nanskip)", ["frozen_sens_nanskip", "frozen_sens_nanskip_r2", "frozen_sens_nanskip_r3"]),
    ("+lowband\n(r1_sweep) 1x", ["fix_lb_r1sweep", "fix_lb_r1sweep_r2", "fix_lb_r1sweep_r3"]),
    ("+lowband 2x", ["r1sweep_u100", "r1sweep_u100_r2", "r1sweep_u100_r3", "r1sweep_u100_r4"]),
    ("recent_w=30", ["rw30", "rw30_r2", "rw30_r3"]),
    ("both mode", ["both", "both_r2"]),
    ("FiLM 1x", ["film", "film_r2", "film_r3", "film_r4", "film_r5"]),
    ("FiLM 2x", ["film2x", "film2x_r2", "film2x_r3"]),
]

def classify(nm):
    fs = glob.glob(f"collapse_{nm}_synth24.npz")
    if not fs: return None
    a = np.load(fs[0])["curve"]; lF = float(np.mean(a[5:, 1])); lS = float(np.mean(a[5:, 2]))
    if lF < 0.0: return "inverted"
    if lF >= 0.8 and lS >= 0.85: return "FULL"
    if lF >= 0.5: return "compressed"
    return "inverted"

labels, full, comp, inv, spans = [], [], [], [], []
for lbl, runs in RECIPES:
    cats = [classify(r) for r in runs]; cats = [c for c in cats if c]
    if not cats: continue
    labels.append(f"{lbl}\n(n={len(cats)})")
    full.append(cats.count("FULL")); comp.append(cats.count("compressed")); inv.append(cats.count("inverted"))
    sp = []
    for r in runs:
        fs = glob.glob(f"collapse_{r}_synth24.npz")
        if fs: sp.append(float(np.mean(np.load(fs[0])["curve"][5:, 2])))
    spans.append(np.mean(sp) if sp else 0)

x = np.arange(len(labels))
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2.2, 1])
full = np.array(full); comp = np.array(comp); inv = np.array(inv)
ax.bar(x, full, color="#2ca02c", label="FULL span (F≥0.8, span≥0.85)")
ax.bar(x, comp, bottom=full, color="#ff7f0e", label="compressed (follows, narrow)")
ax.bar(x, inv, bottom=full + comp, color="#d62728", label="inverted / collapsed")
for i in range(len(labels)):
    tot = full[i] + comp[i] + inv[i]
    ax.text(i, tot + 0.05, f"{full[i]}/{tot}", ha="center", fontweight="bold", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("seed count"); ax.legend(loc="upper left", fontsize=9)
ax.set_title("Span-reliability across recipes (24-job) — how many seeds reach the FULL cost-0→max front\n"
             "FiLM (architectural) is the only recipe that makes full-span reliable")
ax.grid(axis="y", alpha=0.3)
ax2.bar(x, spans, color="#1a73e8", alpha=0.8)
ax2.axhline(1.0, color="#888", ls="--", lw=1, label="span=1.0 (full coverage)")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("mean span\n(achieved/commanded)"); ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("reliability_summary_synth24.png", dpi=120, bbox_inches="tight")
print("saved reliability_summary_synth24.png")
for lbl, f, c, i in zip(labels, full, comp, inv):
    print(f"  {lbl.split(chr(10))[0]:24s} FULL={f} comp={c} inv={i}")
