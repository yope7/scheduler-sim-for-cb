#!/usr/bin/env python3
"""段階的削減アブレーションのPF図。(A)1024の4構成(FWD/FD/F/base)の達成PF重ね=deferを外すと崩壊を視覚化。
(B)三者(合成256/trace256最良fd1101/trace1024最良)の正規化PF=裾の重さで形が違う。
データ: /tmp/ladrich_lad_t_*.json, /tmp/ladrich_lad_s_*.json, /tmp/v2rich_fd1101_*.json の "pf"。
出力 docs/figures/ladder_pf.png
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pf(path):
    if not os.path.exists(path):
        return None
    lines = [l for l in open(path) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1]); return np.asarray(d.get("pf", []), float) if d.get("pf") else None, d.get("hv")
    except Exception:
        return None


def best_pf(prefix, seeds=(1, 2, 3)):
    """hv最大のseedの非支配PFを返す。"""
    best = None; best_hv = -1
    for i in seeds:
        r = load_pf(f"/tmp/{prefix}_{i}.json")
        if r and r[0] is not None and r[1] is not None and r[1] > best_hv:
            best_hv = r[1]; best = r[0]
    return best


def nd_sort(pf):
    if pf is None or len(pf) < 2:
        return pf
    keep = [k for k in range(len(pf)) if not any((pf[j, 0] <= pf[k, 0]) and (pf[j, 1] <= pf[k, 1]) and (j != k) and ((pf[j, 0] < pf[k, 0]) or (pf[j, 1] < pf[k, 1])) for j in range(len(pf)))]
    p = pf[keep]; return p[np.argsort(p[:, 0])]


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

# A: 1024 の4構成
cfgs = [("FWD", "F+W+D", "#5ee0a0"), ("FD", "F+D (density off)", "#4da3ff"),
        ("F", "F (defer OFF)", "#e0556f"), ("base", "baseline", "#9aa7b8")]
for st, lab, col in cfgs:
    pf = nd_sort(best_pf(f"ladrich_lad_t_{st}", (1, 2)))
    if pf is not None:
        axA.plot(pf[:, 0] / 1e8, pf[:, 1] / 1e3, "-o", color=col, ms=4, lw=1.6, label=lab)
axA.set_title("(A) trace1024: stepwise removal\n removing defer (F, red) COLLAPSES the front", fontsize=11)
axA.set_xlabel("cost (x1e8)"); axA.set_ylabel("wait (x1e3)"); axA.grid(alpha=0.3); axA.legend(fontsize=8)

# B: 三者の最良構成(正規化)
best3 = [("synth256", nd_sort(best_pf("ladrich_lad_s_FWD", (1, 2, 3))), "#f0a050"),
         ("trace256 (fd1101)", nd_sort(best_pf("v2rich_fd1101", (1, 2, 3))), "#4da3ff"),
         ("trace1024 (FD)", nd_sort(best_pf("ladrich_lad_t_FD", (1, 2))), "#e0556f")]
for lab, pf, col in best3:
    if pf is not None and len(pf):
        c = pf[:, 0] / pf[:, 0].max(); w = pf[:, 1] / pf[:, 1].max()
        axB.plot(c, w, "-o", color=col, ms=4, lw=1.6, label=lab)
axB.set_title("(B) best config per dataset (normalized)\n heavier tail -> deeper knee", fontsize=11)
axB.set_xlabel("cost (normalized)"); axB.set_ylabel("wait (normalized)"); axB.grid(alpha=0.3); axB.legend(fontsize=8)

fig.suptitle("Stepwise-removal ablation PF: defer dominates at 1024 / effective features depend on tail weight", fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("docs/figures/ladder_pf.png", dpi=110, bbox_inches="tight")
print("[SAVED] docs/figures/ladder_pf.png")
