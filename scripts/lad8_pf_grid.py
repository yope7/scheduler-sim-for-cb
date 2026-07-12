#!/usr/bin/env python3
"""探索E除外の8セル(F/W/D)PFグリッド。trace/合成共通(PFXで切替)。2行(F:OFF/ON)×4列(W×D)。
各seedの達成パレートフロントを色分け表示(seed1=青 / seed2=橙 / seed3=緑)。中央達成面(赤線)は描かない。
データ /tmp/ladrich_{PFX}{F}{W}0{D}_{i}.json の "pf"。軸はデータから自動スケール。
usage: PFX=fdk DS='trace1024' OUT=docs/figures/lad8_pf_grid_trace1024.png uv run python scripts/lad8_pf_grid.py
       PFX=fsk DS='合成1024' OUT=docs/figures/lad8_pf_grid_synth1024.png uv run python scripts/lad8_pf_grid.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

for p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["axes.unicode_minus"] = False

PFX = os.environ.get("PFX", "fdk")
DS = os.environ.get("DS", PFX)
OUT = os.environ.get("OUT", f"docs/figures/lad8_pf_grid_{PFX}.png")

SEED_COLORS = {1: "#3b82f6", 2: "#f0902f", 3: "#10b981"}  # seed1=青 / seed2=橙 / seed3=緑


def load_pf(F, W, D, i):
    p = f"/tmp/ladrich_{PFX}{F}{W}0{D}_{i}.json"
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1])
        return d.get("pf") if "err" not in d else None
    except Exception:
        return None


def nd_front(pf):
    if not pf:
        return None
    p = np.asarray(pf, float); p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0)]
    if len(p) < 2:
        return None
    keep = [k for k in range(len(p)) if not any((p[j, 0] <= p[k, 0]) and (p[j, 1] <= p[k, 1]) and (j != k) and ((p[j, 0] < p[k, 0]) or (p[j, 1] < p[k, 1])) for j in range(len(p)))]
    p = p[keep]; return p[np.argsort(p[:, 0])]


def cell_fronts(F, W, D):
    out = []
    for i in [1, 2, 3]:
        fr = nd_front(load_pf(F, W, D, i))
        if fr is not None:
            out.append((i, fr))
    return out


# 全フロントから軸スケール(最大cost/wait)を自動決定
allpts = []
for F in [0, 1]:
    for W in [0, 1]:
        for D in [0, 1]:
            for _i, fr in cell_fronts(F, W, D):
                allpts.append(fr)
if not allpts:
    raise SystemExit(f"no data for PFX={PFX}")
allc = np.concatenate([f[:, 0] for f in allpts]); allw = np.concatenate([f[:, 1] for f in allpts])
cmax = float(allc.max()); wmax = float(allw.max())
cdiv = 10 ** (int(np.floor(np.log10(cmax))) if cmax > 0 else 0)
wdiv = 10 ** (int(np.floor(np.log10(wmax))) if wmax > 0 else 0)
xlim = cmax / cdiv * 1.05; ylim = wmax / wdiv * 1.05

Frows = [0, 1]
WDcols = [(0, 0), (0, 1), (1, 0), (1, 1)]
onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
seen = set()
for r, F in enumerate(Frows):
    for c, (W, D) in enumerate(WDcols):
        ax = axes[r][c]
        fr_list = cell_fronts(F, W, D)
        title = f"F:{onoff[F]}  W:{onoff[W]}  D:{onoff[D]}"
        if fr_list:
            for i, fr in fr_list:
                ax.plot(fr[:, 0] / cdiv, fr[:, 1] / wdiv, "-o", c=SEED_COLORS[i],
                        ms=2.6, lw=1.5, alpha=0.9, label=f"seed{i}")
                seen.add(i)
            ax.set_title(f"{title}\n({len(fr_list)}seed)", fontsize=10)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=10, color="gray")
        ax.set_xlim(0, xlim); ax.set_ylim(0, ylim)
        ax.set_xlabel(f"Cost (x{cdiv:.0e})", fontsize=8); ax.set_ylabel(f"Wait (x{wdiv:.0e})", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)

handles = [Line2D([0], [0], color=SEED_COLORS[i], marker="o", lw=1.5, ms=5, label=f"seed{i}")
           for i in sorted(seen)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
           ncol=len(handles), fontsize=10, framealpha=0.9)
fig.suptitle(f"{DS} PFグリッド(探索E除外) 8セル = F(フーリエ)/W(密度重み)/D(後回しoffset1)  行=F  列=W×D\n"
             "各seedの達成パレートフロントを色分け（seed1=青 / seed2=橙 / seed3=緑）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=95)
print(f"[SAVED] {OUT}")
nfilled = sum(1 for F in Frows for (W, D) in WDcols if cell_fronts(F, W, D))
print(f"埋まったマス: {nfilled}/8  seeds={sorted(seen)}  (cost_max={cmax:.3g} wait_max={wmax:.3g} cdiv={cdiv:.0e} wdiv={wdiv:.0e})")
