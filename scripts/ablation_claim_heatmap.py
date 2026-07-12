#!/usr/bin/env python3
"""3機能アブレーション表を『良かった点を赤で主張』する色分けに。各指標で良い結果ほど赤(濃)、
悪いは白。機能列は ON=濃グレー/OFF=薄グレー。n_pfは色なし。日本語=Noto Serif CJK JP。
既存 ablation_heatmap.png(緑=良)は残し、別ファイルで出力。出力 docs/figures/ablation_claim.png
"""
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

for p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["axes.unicode_minus"] = False

rows = [
    ((0, 0, 0), 0.602, 0.112, 0.248, 0.114, 0.017, 111),
    ((0, 0, 1), 0.619, 0.058, 0.225, 0.123, 0.031, 44),
    ((0, 1, 0), 0.730, 0.038, 0.264, 0.177, 0.057, 28),
    ((0, 1, 1), 0.502, 0.220, 0.126, 0.051, 0.014, 141),
    ((1, 0, 0), 0.718, 0.100, 0.140, 0.135, 0.041, 61),
    ((1, 0, 1), 0.598, 0.170, 0.182, 0.130, 0.014, 48),
    ((1, 1, 0), 0.792, 0.043, 0.132, 0.081, 0.023, 39),
    ((1, 1, 1), 0.825, 0.022, 0.095, 0.043, 0.036, 36),
]
HV = np.array([r[1] for r in rows]); CD = np.array([r[3] for r in rows]); SP = np.array([r[5] for r in rows])


def good_hi(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def good_lo(v):
    return 1 - (v - v.min()) / (v.max() - v.min() + 1e-9)


score = {"HV": good_hi(HV), "cd": good_lo(CD), "sp": good_lo(SP)}
cmap = cm.Reds  # 良いほど赤(濃)

func_labels = ["Fourier\n特徴埋め込み", "経験サンプリング\n動的調整", "先頭ジョブ\nスキップ追加"]
metric_labels = ["HV\n大=良", "乖離(追従)\n小=良", "Spacing\n小=良", "n_pf"]

fig, ax = plt.subplots(figsize=(12.5, 6.8))
ncol = 3 + 4; nrow = len(rows)
ax.set_xlim(0, ncol); ax.set_ylim(0, nrow + 1.2); ax.invert_yaxis(); ax.axis("off")
for c, lab in enumerate(func_labels):
    ax.text(c + 0.5, 0.6, lab, ha="center", va="center", fontsize=10.5, weight="bold")
for c, lab in enumerate(metric_labels):
    ax.text(3 + c + 0.5, 0.6, lab, ha="center", va="center", fontsize=11, weight="bold")

for r, row in enumerate(rows):
    y = r + 1.2
    (F, S, K) = row[0]
    for c, on in enumerate([F, S, K]):
        col = "#555555" if on else "#ededed"
        ax.add_patch(Rectangle((c, y), 1, 1, facecolor=col, edgecolor="w", lw=1.5))
        ax.text(c + 0.5, y + 0.5, "ON" if on else "OFF", ha="center", va="center",
                fontsize=11, color="w" if on else "#999", weight="bold" if on else "normal")
    for ci, (key, val, std) in enumerate([("HV", row[1], row[2]), ("cd", row[3], row[4]), ("sp", row[5], None)]):
        sc = score[key][r]; col = cmap(0.08 + 0.82 * sc)
        ax.add_patch(Rectangle((3 + ci, y), 1, 1, facecolor=col, edgecolor="w", lw=1.5))
        txt = f"{val:.3f}\n±{std:.3f}" if std is not None else f"{val:.3f}"
        ax.text(3 + ci + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=9.5,
                color="w" if sc > 0.62 else "#222")
    ax.add_patch(Rectangle((6, y), 1, 1, facecolor="white", edgecolor="#cccccc", lw=1.5))
    ax.text(6.5, y + 0.5, f"{row[6]}", ha="center", va="center", fontsize=10.5)

fig.suptitle("各指標の『良かった結果』を赤で主張  （赤いほど良い ・ n_pf は色なし）", fontsize=14, weight="bold", y=0.99)
fig.text(0.5, 0.015, "Fourier ON の行(下4)とサンプリング動的調整 ON の行が赤く染まる＝効いている。全機能ONが最も赤い＝最良(HV 0.825・乖離 0.095)。",
         ha="center", fontsize=10, color="#444")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig("docs/figures/ablation_claim.png", dpi=130, bbox_inches="tight")
print("[SAVED] docs/figures/ablation_claim.png")
