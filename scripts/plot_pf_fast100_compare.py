#!/usr/bin/env python3
"""v9 / main100 / fast100 の到達パレートフロントを比較描画する。

データ源は各runの pcn_mo_hv.json(pareto_fronts_per_eval)。再評価はしない。
真PF(緑線)は weekB 5万ジョブ用の参照が存在しないため描かない(捏造しない)。
代わりに両端の解析解(全オンプレ / 全クラウド)を灰色の参照点として置く。

usage: .venv/bin/python scripts/plot_pf_fast100_compare.py [出力先.png]
"""
from __future__ import annotations

import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 両端の解析解(env の workload プロファイル由来。v9/main100/fast100 で同一)
COST_ALL_CLOUD = 1.95819e10
WAIT_ALL_ONPREM = 2286.43024

# dataviz カテゴリ配色 slot1/2/3(validate_palette.js で PASS 済み)
RUNS = [
    ("run_j50000_gpu_v9_100iter", "v9", "#2a78d6"),
    ("run_j50000_v10_main100", "main100", "#008300"),
    ("run_j50000_v10_fast100", "fast100", "#e87ba4"),
]


def load_fronts(run: str) -> list[np.ndarray]:
    fs = glob.glob(f"experiments/distributed_pcn/{run}/*/pcn_mo_hv.json")
    if not fs:
        return []
    pfs = json.load(open(fs[0])).get("pareto_fronts_per_eval", [])
    out = []
    for pf in pfs:
        a = np.unique(np.array(pf, dtype=float), axis=0)
        out.append(a[np.argsort(a[:, 0])])
    return out


def hv(front: np.ndarray) -> float:
    """参照点(全クラウドcost, 全オンプレwait)からの正規化ハイパーボリューム。"""
    a = front[(front[:, 0] <= COST_ALL_CLOUD) & (front[:, 1] <= WAIT_ALL_ONPREM)]
    if len(a) == 0:
        return 0.0
    tot, prev_w = 0.0, WAIT_ALL_ONPREM
    for c, w in a:
        if w < prev_w:
            tot += (COST_ALL_CLOUD - c) * (prev_w - w)
            prev_w = w
    return tot / (COST_ALL_CLOUD * WAIT_ALL_ONPREM)


def draw_front(ax, front, color, label, *, lw=2.0, ms=8, alpha=1.0, zorder=3):
    ax.step(front[:, 0], front[:, 1], where="post", color=color, lw=lw,
            alpha=alpha, zorder=zorder, solid_capstyle="round")
    ax.plot(front[:, 0], front[:, 1], "o", color=color, ms=ms, alpha=alpha,
            zorder=zorder + 1, markeredgecolor="white", markeredgewidth=1.5,
            label=label)


def anchors(ax):
    ax.plot([0], [WAIT_ALL_ONPREM], "s", color="#8a8a80", ms=7, zorder=2)
    ax.plot([COST_ALL_CLOUD], [0], "s", color="#8a8a80", ms=7, zorder=2)
    ax.annotate("全オンプレ", (0, WAIT_ALL_ONPREM), textcoords="offset points",
                xytext=(13, -20), fontsize=9, color="#5c5c55")
    ax.annotate("全クラウド", (COST_ALL_CLOUD, 0), textcoords="offset points",
                xytext=(-14, 10), fontsize=9, color="#5c5c55", ha="right")


def style(ax, title, xlab="コスト", ylab="平均待ち時間 [秒]"):
    ax.set_title(title, fontsize=11.5, pad=9, color="#2b2b26", loc="left")
    ax.set_xlabel(xlab, fontsize=10, color="#5c5c55")
    ax.set_ylabel(ylab, fontsize=10, color="#5c5c55")
    ax.grid(True, lw=0.6, color="#e3e3dd", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c0")
    ax.tick_params(colors="#5c5c55", labelsize=9)


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/pf_fast100_compare.png"
    data = {run: load_fronts(run) for run, _, _ in RUNS}

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), facecolor="#fcfcfb")
    fig.suptitle("5万ジョブ・100 iter — 到達パレートフロントの比較（左下ほど良い）",
                 fontsize=14, fontweight="bold", color="#2b2b26", x=0.055, ha="left")

    # A: 最終PF(全域)
    ax = axes[0][0]
    for run, label, color in RUNS:
        if data[run]:
            draw_front(ax, data[run][-1], color, label)
    anchors(ax)
    style(ax, "A. 最終評価のPF（全域）")
    ax.set_xlim(-4e8, COST_ALL_CLOUD * 1.06)
    ax.set_ylim(-70, WAIT_ALL_ONPREM * 1.06)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    # B: 最終PF(安い側の拡大)
    ax = axes[0][1]
    for run, label, color in RUNS:
        if data[run]:
            draw_front(ax, data[run][-1], color, label)
    style(ax, "B. 同じデータの拡大（コスト 0〜4e9 の中域）")
    ax.set_xlim(-1.2e8, 4e9)
    ax.set_ylim(-70, WAIT_ALL_ONPREM * 1.06)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    # C: fast100 の評価5回の推移(逐次ランプ=単一色相の明→暗)
    ax = axes[1][0]
    fronts = data["run_j50000_v10_fast100"]
    ramp = ["#bcd7f5", "#8dbaec", "#5c9ae2", "#2a78d6", "#17509a"]
    for i, fr in enumerate(fronts):
        draw_front(ax, fr, ramp[i], f"評価{i + 1}（iter {(i + 1) * 20}）",
                   lw=1.8, ms=7, zorder=3 + i)
    anchors(ax)
    style(ax, "C. fast100 の推移（評価5回・薄い=序盤、濃い=終盤）")
    ax.set_xlim(-4e8, COST_ALL_CLOUD * 1.06)
    ax.set_ylim(-70, WAIT_ALL_ONPREM * 1.06)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # D: 正規化HVの推移
    ax = axes[1][1]
    for run, label, color in RUNS:
        if not data[run]:
            continue
        ys = [hv(f) for f in data[run]]
        xs = [(i + 1) * 20 for i in range(len(ys))]
        ax.plot(xs, ys, "-o", color=color, lw=2.0, ms=8, label=label,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{label} {ys[-1]:.3f}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(-8, 11 if label != "main100" else -18),
                    fontsize=10, color="#2b2b26", ha="right", fontweight="bold")
    style(ax, "D. 正規化ハイパーボリューム（大きいほど良い）",
          xlab="イテレーション", ylab="正規化HV")
    ax.set_ylim(0.35, 0.92)
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    fig.text(0.055, 0.017,
             "※真PF（緑線）は weekB 5万ジョブ用の参照が存在しないため描いていない。"
             "灰色の四角は解析的に分かる両端（全オンプレ = コスト0/待ち2286秒、全クラウド = コスト1.96e10/待ち0）。\n"
             "※fast100 は v9・main100 と待ち時間まわりのフラグが2つ異なる（PCN_CMD_TRACK_WAIT_WEIGHT / PCN_CMD_WAIT_ZERO）ため、"
             "この差は高速化ではなくレシピの効果として読むこと。",
             fontsize=9, color="#5c5c55", ha="left", va="bottom")

    fig.tight_layout(rect=(0.008, 0.055, 0.995, 0.965))
    fig.savefig(out, dpi=140, facecolor="#fcfcfb")
    print(f"saved: {out}")
    for run, label, _ in RUNS:
        if data[run]:
            print(f"  {label:8s} 最終HV={hv(data[run][-1]):.4f} "
                  f"点数={len(data[run][-1])} "
                  f"cost[{data[run][-1][:,0].min():.3g},{data[run][-1][:,0].max():.3g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
