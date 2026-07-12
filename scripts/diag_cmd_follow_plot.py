#!/usr/bin/env python3
"""指令cost→達成cost 追従曲線の作図。/tmp/cf_*.json を読み、y=x 対角に対する乗り方で
「追従(対角に乗る) vs 崩壊(対角の上=指令を超えるオーバーシュート)」を可視化。
出力 docs/figures/cmdfollow_trace256.png
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

PANELS = [("base_1", "base seed1（追従成功）", "#3b82f6"),
          ("base_2", "base seed2（崩壊＝指令無視）", "#ef4444"),
          ("ct_1", "cmd-track seed1", "#10b981"),
          ("ct_2", "cmd-track seed2", "#10b981")]

DIV = 1e8
fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
for ax, (tag, title, col) in zip(axes, PANELS):
    p = f"/tmp/cf_{tag}.json"
    if not os.path.exists(p):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes); ax.set_title(title); continue
    d = json.load(open(p))
    cc = np.array(d["cmd_cost"]) / DIV; ac = np.array(d["ach_cost"]) / DIV
    lim = max(cc.max(), ac.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", c="gray", lw=1.2, label="y=x（理想追従）")
    ax.plot(cc, ac, "-o", c=col, ms=5, lw=1.6, label="達成")
    # オーバーシュート領域を塗る
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#ef4444", alpha=0.06)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("指令 cost（×1e8）", fontsize=10); ax.set_ylabel("達成 cost（×1e8）", fontsize=10)
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("指令cost→達成cost の追従曲線（trace256）  対角=理想追従／上に外れる=指令を超えるオーバーシュート（崩壊）\n"
             "base はseedで崩壊しうる（seed2は指令を無視して高コスト側へオーバーシュート）。cmd-track loss は対角に乗せる。", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("docs/figures/cmdfollow_trace256.png", dpi=95)
print("[SAVED] docs/figures/cmdfollow_trace256.png")
