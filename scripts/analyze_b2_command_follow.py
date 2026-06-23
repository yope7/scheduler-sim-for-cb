#!/usr/bin/env python3
"""メカニズム可視化: 「指令コスト vs 達成コスト」。eval は rp の最小(=全オンプレ角)〜最大まで
   コスト指令を掃引している（eval_b2_compare.py: cg=linspace(rp.min,rp.max,NCMD)、greedy[i]↔cg[i]）。
   なので npz から指令列を再構成し、各指令に対する greedy 達成コストを描けば、
   「端の指令に方策が追従できているか」が分かる。y=x が完全追従。
   仮説: 低ジョブ数では b2 が安い端で飽和（達成コストが下がりきらない=外挿不能）し、端点を失う。
   モデル再実行なし=GPU不要=OOM安全。usage: OUT=pf_b2_cmdfollow.png PYTHONPATH=. .venv/bin/python scripts/analyze_b2_command_follow.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SIZES = [16, 32, 64, 128, 256, 512, 1024]
OUT = os.environ.get("OUT", "pf_b2_cmdfollow.png")
NCMD = int(os.environ.get("NCMD", "40"))


def names(nj):
    if nj == 128:
        return "truepf_film_s0.npz", "truepf_fourier_s0.npz"
    return f"truepf_film_{nj}_s0.npz", f"truepf_fourier_{nj}_s0.npz"


avail = []
for nj in SIZES:
    fn, gn = names(nj)
    if os.path.exists(fn) and os.path.exists(gn):
        F = np.load(fn); G = np.load(gn)
        if "greedy_0" in F and "greedy_0" in G:
            avail.append((nj, F, G))
if not avail:
    raise SystemExit("no data")

fig, axes = plt.subplots(1, len(avail), figsize=(4.6 * len(avail), 4.7)); axes = np.atleast_1d(axes)
for ax, (nj, F, G) in zip(axes, avail):
    rp = F["rp_0"]
    cg = np.linspace(float(rp[:, 0].min()), float(rp[:, 0].max()), NCMD)  # eval が出した指令コスト列
    fa = F["greedy_0"][:, 0]; ba = G["greedy_0"][:, 0]                    # 達成コスト（film / b2）
    n = min(len(cg), len(fa), len(ba)); cg, fa, ba = cg[:n], fa[:n], ba[:n]
    lim_lo = min(cg.min(), fa.min(), ba.min()); lim_hi = max(cg.max(), fa.max(), ba.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=.5, label="perfect follow (y=x)")
    ax.scatter(cg, fa, s=22, c="#d62728", label="baseline achieved", zorder=3)
    ax.scatter(cg, ba, s=22, c="#1a73e8", marker="D", label="b2 achieved", zorder=4)
    cheap_cmd = cg.min()
    ax.axvline(cheap_cmd, color="#999", ls=":", lw=1)
    ax.set_title(f"n_jobs={nj}\ncheapest command={cheap_cmd:.0f}\nfilm reaches {fa.min():.0f} / b2 reaches {ba.min():.0f}", fontsize=9.5)
    ax.set_xlabel("commanded cost (desired return)"); ax.set_ylabel("achieved cost (greedy)")
    ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Command-following at the cheap edge: at small n_jobs b2 SATURATES (can't follow extreme commands -> edge lost); at large n_jobs it tracks y=x",
             fontsize=11, y=1.04)
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved {OUT}")
for nj, F, G in avail:
    rp = F["rp_0"]; cg = np.linspace(float(rp[:, 0].min()), float(rp[:, 0].max()), NCMD)
    fa = F["greedy_0"][:, 0]; ba = G["greedy_0"][:, 0]
    print(f"n={nj:>4}  cheapest_cmd={cg.min():>8.0f}  film_reaches={fa.min():>8.0f}  b2_reaches={ba.min():>8.0f}  "
          f"b2 shortfall vs film={ba.min()-fa.min():>8.0f}")
