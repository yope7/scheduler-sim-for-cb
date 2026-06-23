#!/usr/bin/env python
"""trace の before(戦略OFF)/after(戦略ON) greedy を共通の真PF上に重ねる(合成 pf_actual_before_after.png と同形式)。
存在する truepf_trace{SCALE}_{before,after}_s0.npz を自動検出して 1xN パネルで描画。
使い方: PYTHONPATH=. .venv/bin/python scripts/plot_trace_compare.py
出力: docs/figures/pf_trace_before_after.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize


def load(fn):
    d = np.load(fn)
    return d["greedy_0"], d["samp_0"], d["rp_0"]


def hv2d(pts, ref):
    """2D hypervolume (minimize both), staircase. ref = nadir point dominated by all."""
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0
    prev_cost = ref[0]
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev_cost - c) * (ref[1] - w)
        prev_cost = c
    return hv


# scale -> fix α (scale依存α: 256→0.5, 512+→1.0)
ALPHA = {256: "0.5", 512: "1", 1024: "1"}
scales = []
for nj in (256, 512, 1024):
    bf = f"truepf_trace{nj}_before_s0.npz"
    ff = f"truepf_trace{nj}_after_s0.npz"
    if os.path.exists(bf) and os.path.exists(ff):
        scales.append((nj, bf, ff))
if not scales:
    raise SystemExit("no truepf_trace*_{before,after}_s0.npz found")

n = len(scales)
fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.3), squeeze=False)
axes = axes[0]
print("trace before/after (reach=min cost/all-cloud, span=range/all-cloud):")
for ax, (nj, bf, ff) in zip(axes, scales):
    gb, sb, rp = load(bf)
    gf, sf, rpf = load(ff)
    # 同一インスタンス検証(rp一致)
    same = np.allclose(rp[:, 0].max(), rpf[:, 0].max())
    cmax = rp[:, 0].max()
    allpts = np.vstack([rp, sb, sf, gb, gf])
    nd = get_non_dominated_inds_minimize(allpts)
    pf = allpts[nd]
    pf = pf[np.argsort(pf[:, 0])]
    # hypervolume (reference-PF-independent quality): fraction of true-PF HV
    ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
    hvb = hv2d(gb, ref); hvf = hv2d(gf, ref); hvt = hv2d(pf, ref)
    fb = hvb / hvt if hvt > 0 else 0.0
    ff_ = hvf / hvt if hvt > 0 else 0.0
    ax.plot(pf[:, 0] / cmax, pf[:, 1], "-", c="#2ca02c", lw=2.4,
            label="true PF (best achievable)", zorder=2)
    ax.scatter(gb[:, 0] / cmax, gb[:, 1], s=40, c="#d62728",
               label=f"before  (HV {fb:.0%}, gapmax {np.diff(np.sort(gb[:,0]/cmax)).max():.2f})", zorder=3, edgecolor="k", lw=.3)
    ax.scatter(gf[:, 0] / cmax, gf[:, 1], s=40, c="#1a73e8", marker="D",
               label=f"fix α={ALPHA.get(nj,'?')}  (HV {ff_:.0%}, gapmax {np.diff(np.sort(gf[:,0]/cmax)).max():.2f})", zorder=4, edgecolor="k", lw=.3)
    ax.axvspan(0, 0.12, color="#cfe8cf", alpha=.6, zorder=0)
    ax.text(0.06, 0.97, "cheap\ncorner", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="#1b7a1b")
    ax.set_xlim(-0.03, 1.03)
    ax.set_title(f"trace  n_jobs = {nj}" + ("" if same else "  [!instance mismatch]"),
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cost  (fraction of all-cloud;  0 = all on-prem / cheapest)")
    ax.set_ylabel("Average wait time")
    ax.legend(fontsize=9, loc="upper right", framealpha=.95)
    ax.grid(alpha=.3)
    rb, spb = gb[:, 0].min() / cmax, (gb[:, 0].max() - gb[:, 0].min()) / cmax
    rf, spf = gf[:, 0].min() / cmax, (gf[:, 0].max() - gf[:, 0].min()) / cmax
    print(f"  trace n={nj:4d}: before HV={fb:.0%} span={spb:.3f} gapmax={np.diff(np.sort(gb[:,0]/cmax)).max():.3f}  |  "
          f"after HV={ff_:.0%} span={spf:.3f} gapmax={np.diff(np.sort(gf[:,0]/cmax)).max():.3f}  same_instance={same}")
fig.suptitle("Job trace (job_type=2): before(red)=strategy OFF, fix(blue)=count-normalization (scale-dependent α).  "
             "Same NON-MONOTONIC scale trade-off as synthetic: fix HELPS at 512 (HV 31%→54%, bipolar→continuous) but HURTS at 256 (HV 72%→64%, over-spreads an already-good front).  "
             "The synthetic cheap-corner loss does NOT appear on trace — before reaches cost 0 at both scales.",
             fontsize=9.6, y=1.005)
fig.tight_layout()
out = "docs/figures/pf_trace_before_after.png"
fig.savefig(out, dpi=125, bbox_inches="tight")
print("saved", out)
