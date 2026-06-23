#!/usr/bin/env python3
"""b2 のスケール依存を1枚に：各ジョブ数 NJ の seed0 共通真PF gap を baseline/b2 で並べ、
   gap-vs-n_jobs（横軸 log）の折れ線にする。b2 線が baseline 線より下＝b2 が効く領域。
   損益分岐点（baseline と b2 が交差する規模）を可視化する。
   npz 命名：128 だけ truepf_film_s0.npz（無印）、他は truepf_film_{NJ}_s0.npz。存在するサイズだけ使う。
   usage: OUT=pf_b2_scale.png PYTHONPATH=. .venv/bin/python scripts/plot_b2_scale.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize

SIZES = [16, 32, 64, 128, 256, 512, 1024]
OUT = os.environ.get("OUT", "pf_b2_scale.png")


def names(nj):
    if nj == 128:  # 128 は無印で保存されている（run_b2_jobsize.sh 以前の本流）
        return "truepf_film_s0.npz", "truepf_fourier_s0.npz"
    return f"truepf_film_{nj}_s0.npz", f"truepf_fourier_{nj}_s0.npz"


def gap(greedy, truepf):
    ex = np.clip(greedy[:, 1] - np.interp(greedy[:, 0], truepf[:, 0], truepf[:, 1]), 0, None) / max(1e-9, truepf[:, 1].ptp())
    return float(ex.mean())


xs, fb, bb = [], [], []
for nj in SIZES:
    fn, gn = names(nj)
    if not (os.path.exists(fn) and os.path.exists(gn)):
        print(f"skip n_jobs={nj} (missing {fn} / {gn})"); continue
    F = np.load(fn); G = np.load(gn)
    if "greedy_0" not in F or "greedy_0" not in G:
        print(f"skip n_jobs={nj} (no seed0 in npz)"); continue
    rp = F["rp_0"]; fg = F["greedy_0"]; gg = G["greedy_0"]
    allp = np.vstack([rp, F["samp_0"], G["samp_0"]])
    nd = get_non_dominated_inds_minimize(allp); tp = allp[nd]; tp = tp[np.argsort(tp[:, 0])]
    a, b = gap(fg, tp), gap(gg, tp)
    xs.append(nj); fb.append(a); bb.append(b)
    print(f"n_jobs={nj:>4}: baseline={a:.3f}  b2={b:.3f}  delta={b-a:+.3f}  {'b2 better' if b < a else 'b2 worse'}")

if not xs:
    raise SystemExit("no data — run scripts/run_b2_jobsize.sh for some sizes first")

xs = np.array(xs); fb = np.array(fb); bb = np.array(bb)
fig, ax = plt.subplots(figsize=(8.4, 5.4))
# b2 が良い/悪い領域を薄く塗る
ax.fill_between(xs, fb, bb, where=(bb <= fb), interpolate=True, color="#e6f4ea", alpha=.8, label="b2 better (b2≤baseline)")
ax.fill_between(xs, fb, bb, where=(bb > fb), interpolate=True, color="#fce8e6", alpha=.8, label="b2 worse")
ax.plot(xs, fb, "o-", color="#d62728", lw=2, ms=8, label="baseline (no Fourier)")
ax.plot(xs, bb, "D-", color="#1a73e8", lw=2, ms=8, label="b2 Fourier")
for x, a, b in zip(xs, fb, bb):
    ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#d62728")
    ax.annotate(f"{b:.3f}", (x, b), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#1a73e8")
ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
ax.set_xlabel("n_jobs (log2)"); ax.set_ylabel("seed0 gap-to-true PF (lower = better)")
ax.set_title("b2 scale dependence: small n = high-variance PF (b2 hurts) / large n = stable PF (b2 helps)\n(red band = b2 worse, green band = b2 better, crossover = break-even)", fontsize=10.5)
ax.grid(alpha=.3); ax.legend(fontsize=9, loc="best")
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved {OUT}  (sizes: {list(xs)})")
