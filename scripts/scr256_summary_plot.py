#!/usr/bin/env python3
"""trace256 スクリーンのまとめ図: 各seedを (追従[小=良] x HV[大=良]) 平面に置く。
理想=左上。崩壊(追従>0.15)は×印。ct03(軽いcmd-track)が左上に密集=strict win。
+ 右パネル: configごとの追従の seed間ばらつき(箱)。出力 docs/figures/cmdtrack_summary_trace256.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

for p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try: fm.fontManager.addfont(p)
    except Exception: pass
plt.rcParams["font.family"] = "Noto Serif CJK JP"; plt.rcParams["axes.unicode_minus"] = False

CFG = [("base", "base（無対策）", "#3b82f6", [1,2,3,4,5]),
       ("ct", "cmd-track w=1.0", "#f0902f", [1,2,3,4,5]),
       ("ct03", "cmd-track w=0.3", "#10b981", [1,2,3]),
       ("fb2", "Fourier帯2", "#a855f7", [1,2]),
       ("wd", "weight_decay", "#9ca3af", [1,2])]


def load(tag, i):
    p = f"/tmp/scr256_{tag}_{i}.json"
    if not os.path.exists(p): return None
    try:
        d = json.loads([l for l in open(p) if l.strip().startswith("{")][-1])
        return d if "hv" in d else None
    except: return None


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios":[1.5,1]})
follow_by = {}
for tag, lab, col, seeds in CFG:
    xs=[]; ys=[]
    for i in seeds:
        d = load(tag, i)
        if not d: continue
        cd = d["cmd_dist"]; hv = d["hv"]; xs.append(cd); ys.append(hv)
        mk = "X" if cd > 0.15 else "o"
        ax.scatter(cd, hv, c=col, marker=mk, s=140 if mk=="X" else 90, edgecolors="k", linewidths=0.6, zorder=3, alpha=0.9)
    follow_by[lab] = xs
    if xs: ax.scatter([], [], c=col, label=lab)  # legend proxy
ax.axvline(0.15, ls="--", c="#ef4444", lw=1, alpha=0.6)
ax.text(0.152, ax.get_ylim()[0], " 崩壊しきい値", color="#ef4444", fontsize=8, va="bottom")
ax.annotate("理想\n（追従よく・HV高い）", xy=(0.01, 0.86), fontsize=10, color="#10b981", weight="bold")
ax.set_xlabel("指令追従距離 cmd_dist（小さいほど良い）", fontsize=11)
ax.set_ylabel("HV（大きいほど良い）", fontsize=11)
ax.set_title("trace256: 各seedの (追従 × HV)  ×=崩壊seed", fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")

# 右: 追従のばらつき(各seed点 + mean)
labs=[c[1] for c in CFG]; cols=[c[2] for c in CFG]
for j,(tag,lab,col,seeds) in enumerate(CFG):
    xs=follow_by[lab]
    if not xs: continue
    ax2.scatter([j]*len(xs), xs, c=col, s=70, edgecolors="k", linewidths=0.5, zorder=3)
    ax2.plot([j-0.2,j+0.2],[np.mean(xs)]*2, c=col, lw=3)
ax2.set_xticks(range(len(CFG))); ax2.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
ax2.axhline(0.15, ls="--", c="#ef4444", lw=1, alpha=0.6)
ax2.set_ylabel("指令追従距離（小=良）", fontsize=11)
ax2.set_title("追従の seed間ばらつき（横線=平均）", fontsize=12); ax2.grid(alpha=0.3)

fig.suptitle("軽いcmd-track loss(w=0.3) が strict win: base の確率的崩壊(×)を消し、HVも追従も改善・最安定\n"
             "（weight_decay は崩壊×2=反証。Fourier帯2も追従改善だがHV低め）", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig("docs/figures/cmdtrack_summary_trace256.png", dpi=110)
print("[SAVED] docs/figures/cmdtrack_summary_trace256.png")
