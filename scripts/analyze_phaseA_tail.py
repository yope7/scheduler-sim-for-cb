#!/usr/bin/env python
"""Phase A 集計: 裾レベル L ごとに崩壊率・HV を出し、崖(collapse onset)を特定する。
図は「読み方が図だけで分かる」ことを最優先に注釈を多めに入れる。
usage: PYTHONPATH=. .venv/bin/python scripts/analyze_phaseA_tail.py
出力: docs/figures/phaseA_tail_cliff.png, docs/figures/phaseA_tail_pf.png
"""
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from src.agents.pcn_agent import get_non_dominated_inds_minimize

_FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FP).get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

LEVELS = [("tailL00", 0.0), ("tailL02", 0.2), ("tailL04", 0.4),
          ("tailL06", 0.6), ("tailL08", 0.8), ("tailL10", 1.0)]
SCALE = "512"


def hv2d(pts, ref):
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(c, w) for c, w in nd if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0
    prev = ref[0]
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev - c) * (ref[1] - w)
        prev = c
    return hv


def load(tag):
    runs = {}
    for fn in sorted(glob.glob(f"results/eval_pf/truepf_trace{SCALE}_{tag}{SCALE}_[0-9]_s0.npz")):
        i = int(re.search(rf"{tag}{SCALE}_(\d+)_s0", fn).group(1))
        runs[i] = np.load(fn)["greedy_0"]
    return runs


rows = []
pf_data = []
for tag, L in LEVELS:
    runs = load(tag)
    if not runs:
        continue
    allpts = np.vstack(list(runs.values()))
    ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
    cmax = allpts[:, 0].max()
    pf = allpts[get_non_dominated_inds_minimize(allpts)]
    pf = pf[np.argsort(pf[:, 0])]
    hvt = hv2d(pf, ref)
    hvs = np.array([hv2d(g, ref) / max(hvt, 1e-9) for g in runs.values()])
    collapse = int((hvs < 0.3).sum())
    rows.append((L, tag, hvs, collapse, len(runs)))
    pf_data.append((L, tag, runs, ref, cmax, pf, hvt))
    print(f"L={L} ({tag}, n={len(runs)}): HV " + " ".join(f"{x:.0%}" for x in sorted(hvs, reverse=True)) +
          f" | mean={hvs.mean():.0%} std={hvs.std():.0%} collapse(<30%)={collapse}/{len(runs)}")

if not rows:
    print("\nNo results yet — run the sweep first.")
    raise SystemExit(0)

# ============ 図1: 崖曲線（1軸・seed個別点つき・注釈多め） ============
Ls = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(9.5, 6.2))

# 崩壊ゾーン（下の赤帯）と trace 領域（右の縦帯）
ax.axhspan(0, 30, color="#fce8e6", zorder=0)
ax.text(0.02, 15, "崩壊ゾーン（HV<30% ＝ 学習が壊滅した run）", fontsize=10, color="#a50e0e", va="center")
ax.axvspan(0.83, 1.02, color="#fff0d9", zorder=0)
ax.text(0.925, 97, "実トレースの裾は\nこの重さ", fontsize=9.5, color="#b05a00", ha="center", va="top")

# 各seedの個別点（小さい灰点）と平均（青丸＋線）
for L, tag, hvs, collapse, n in rows:
    jitter = (np.arange(len(hvs)) - (len(hvs) - 1) / 2) * 0.012
    ax.scatter(np.full(len(hvs), L) + jitter, hvs * 100, s=26, c="#9aa0a6", zorder=3,
               label="各seedの結果（5回の独立学習）" if L == rows[0][0] else None)
means = [r[2].mean() * 100 for r in rows]
ax.plot(Ls, means, "-o", c="#1a73e8", lw=2.8, ms=11, zorder=4, label="5seedの平均")
for L, m in zip(Ls, means):
    ax.annotate(f"{m:.0f}%", (L, m), textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=11, fontweight="bold", color="#1a73e8")

# 注釈（何が起きているか言葉で）
if len(rows) >= 1:
    ax.annotate("均質（裾なし）:\n的が曖昧で凡庸", (Ls[0], means[0]),
                xytext=(Ls[0] + 0.06, means[0] - 24), fontsize=9.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#666"))
if len(rows) >= 3:
    ax.annotate("軽い裾:\n明確なレバレッジ点が\n学習の的になり改善", (Ls[2], means[2]),
                xytext=(Ls[2] - 0.13, means[2] - 38), fontsize=9.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#666"))

ax.set_xlabel("裾ダイヤル L（0＝全ジョブ同サイズの均質合成 → 1＝実トレース級の巨大ジョブ混入）", fontsize=11)
ax.set_ylabel("最終モデルの成績\n（理想PFにどれだけ乗れたか, %。高いほど良い）", fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 105)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(alpha=.3)
ax.legend(fontsize=10, loc="lower right")
done_n = len(rows)
ax.set_title(f"裾ダイヤルを回すと学習結果はどう変わるか（early-stop なしの最終モデルで判定, {done_n}/6 レベル完了）",
             fontsize=12.5, fontweight="bold", pad=12)
fig.savefig("docs/figures/phaseA_tail_cliff.png", dpi=125, bbox_inches="tight")
print("\nsaved docs/figures/phaseA_tail_cliff.png")

# ============ 図2: 各レベルの達成PF（run別に色分け＝「どこまで届いたか」が見える） ============
# ポイント: 5本を同じ色で重ねると全部「緑に張り付いて」見えて差が分からない。
# 差の正体は「左（安い側）までカバーできたか」。runを成績順に色分けし、legendにHV%を出す。
RUN_COLORS = ["#0b57d0", "#5e97f6", "#9aa0a6", "#e37400", "#d93025"]  # 1位→5位
n = len(pf_data)
ncol = min(3, n)
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.0 * nrow), squeeze=False)
for k, (L, tag, runs, ref, cmax, pf, hvt) in enumerate(pf_data):
    ax = axes[k // ncol][k % ncol]
    ax.plot(pf[:, 0] / cmax, pf[:, 1] / 1000, "-", c="#188038", lw=3.4, zorder=2,
            alpha=.85, label="理想の限界線（真PF）")
    scored = sorted(((hv2d(g, ref) / max(hvt, 1e-9), i, g) for i, g in runs.items()),
                    reverse=True)
    ncoll = sum(1 for h, _, _ in scored if h < 0.3)
    for rank, (h, i, g) in enumerate(scored):
        o = np.argsort(g[:, 0])
        c = RUN_COLORS[min(rank, len(RUN_COLORS) - 1)]
        ax.plot(g[o, 0] / cmax, g[o, 1] / 1000, "-o", c=c, lw=1.7, ms=3.0,
                alpha=.95, zorder=4 + (len(scored) - rank),
                label=f"run{i}: {h:.0%}{'（崩壊）' if h < 0.3 else ''}")
        # 各runの「左端＝どこまで安くできたか」を▼で強調
        xl = g[o, 0].min() / cmax
        yl = g[o][0, 1] / 1000
        ax.plot([xl], [yl], "v", c=c, ms=9, zorder=10)
    ax.set_title(f"L={L}（崩壊 {ncoll}/{len(scored)}）", fontsize=12, fontweight="bold")
    ax.set_xlabel("クラウド利用額（→右ほど高い）", fontsize=9.5)
    ax.set_ylabel("平均待ち時間 [千秒]（↑上ほど待つ）", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right", title="▼=各runの左端（安さの限界）", title_fontsize=8)
    ax.grid(alpha=.3)
    if k == 0:
        ax.annotate("どの線も緑には乗れている。\n差は「▼がどこまで左に届くか」\n＝安い領域までカバーできた run が高得点",
                    xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9, color="#333", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.45", fc="#fffde7", ec="#999"))
for k in range(n, nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
fig.suptitle("各裾レベルの達成PF — 線の色＝そのrunの成績順位。差は「左（安い側）への到達範囲」に出る",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig("docs/figures/phaseA_tail_pf.png", dpi=120, bbox_inches="tight")
print("saved docs/figures/phaseA_tail_pf.png")
