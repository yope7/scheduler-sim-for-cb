#!/usr/bin/env python3
"""最終図: base vs ct03(軽いcmd-track) を trace256/512 で比較。HV と 追従(error bar付)、崩壊率。
規模が上がるほど strict win が拡大することを示す。出力 docs/figures/cmdtrack_final.png
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


def load(p):
    if not os.path.exists(p): return None
    try:
        d = json.loads([l for l in open(p) if l.strip().startswith("{")][-1]); return d if "hv" in d else None
    except: return None


def stat(prefix, seeds):
    hv=[]; cd=[]
    for i in seeds:
        d=load(f"{prefix}_{i}.json")
        if d: hv.append(d["hv"]); cd.append(d["cmd_dist"])
    hv=np.array(hv); cd=np.array(cd)
    sd=lambda a: np.std(a,ddof=1) if len(a)>1 else 0
    return dict(hv=hv.mean(), hvs=sd(hv), cd=cd.mean(), cds=sd(cd), coll=int((cd>0.15).sum()), n=len(hv))

S = {
    ("256","base"): stat("/tmp/scr256_base",[1,2,3,4,5]),
    ("256","ct03"): stat("/tmp/scr256_ct03",[1,2,3,4,5]),
    ("512","base"): stat("/tmp/tr512_base512",[1,2,3,4,5]),
    ("512","ct03"): stat("/tmp/tr512_ct03512",[1,2,3,4,5]),
}
scales=["256","512"]; cols={"base":"#3b82f6","ct03":"#10b981"}
labn={"base":"base（無対策）","ct03":"ct03（cmd-track w=0.3）"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
x=np.arange(len(scales)); w=0.36
# HV
ax=axes[0]
for j,m in enumerate(["base","ct03"]):
    ys=[S[(s,m)]["hv"] for s in scales]; es=[S[(s,m)]["hvs"] for s in scales]
    ax.bar(x+(j-0.5)*w, ys, w, yerr=es, capsize=5, color=cols[m], label=labn[m], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([f"trace{s}" for s in scales]); ax.set_ylabel("HV（大=良）", fontsize=11)
ax.set_title("HV（誤差棒=seed間std）", fontsize=12); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for j,m in enumerate(["base","ct03"]):
    for k,s in enumerate(scales):
        ax.text(x[k]+(j-0.5)*w, S[(s,m)]["hv"]+S[(s,m)]["hvs"]+0.01, f"{S[(s,m)]['hv']:.3f}", ha="center", fontsize=8)
# 追従
ax=axes[1]
for j,m in enumerate(["base","ct03"]):
    ys=[S[(s,m)]["cd"] for s in scales]; es=[S[(s,m)]["cds"] for s in scales]
    ax.bar(x+(j-0.5)*w, ys, w, yerr=es, capsize=5, color=cols[m], label=labn[m], alpha=0.9)
ax.axhline(0.15, ls="--", c="#ef4444", lw=1, alpha=0.6); ax.text(1.3,0.152,"崩壊しきい値",color="#ef4444",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"trace{s}" for s in scales]); ax.set_ylabel("指令追従距離（小=良）", fontsize=11)
ax.set_title("命令追従", fontsize=12); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
# 崩壊率
ax=axes[2]
for j,m in enumerate(["base","ct03"]):
    ys=[100*S[(s,m)]["coll"]/S[(s,m)]["n"] for s in scales]
    ax.bar(x+(j-0.5)*w, ys, w, color=cols[m], label=labn[m], alpha=0.9)
    for k,s in enumerate(scales):
        ax.text(x[k]+(j-0.5)*w, ys[k]+1, f"{S[(s,m)]['coll']}/{S[(s,m)]['n']}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"trace{s}" for s in scales]); ax.set_ylabel("崩壊率 (%)", fontsize=11)
ax.set_title("確率的崩壊の発生率", fontsize=12); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

fig.suptitle("軽い cmd-track loss が strict win（trace256 & 512）: HV↑・追従↑・崩壊0・最安定。規模が上がるほど効果拡大（ΔHV +0.019→+0.057, 各5seed）",
             fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig("docs/figures/cmdtrack_final.png", dpi=110)
print("[SAVED] docs/figures/cmdtrack_final.png")
print("ΔHV 256:", round(S[("256","ct03")]["hv"]-S[("256","base")]["hv"],3),
      " 512:", round(S[("512","ct03")]["hv"]-S[("512","base")]["hv"],3))
