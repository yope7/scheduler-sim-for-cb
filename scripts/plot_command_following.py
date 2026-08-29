#!/usr/bin/env python3
"""「指令した点」と「実際に着地した点」を線で結んで、指令追従を目で見る。

背景: 学習中の eval は archive の達成点を指令に使い、結果のうち非支配な点だけ残す
(distributed_pcn.py:2964)。外した分が消えるので「数打ち当たる」と区別がつかない。
ここでは外から機械的に切った格子を指令にし、外した点も全部描く。

入力: eval_uniform_pf_lockstep.py が吐く npz (commands, points, nd_mask)
  commands = [r0=-total_wait, r1=-cost] / points = [total_cost, avg_wait]

usage: .venv/bin/python scripts/plot_command_following.py <npz> <out.png> [n_jobs]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_CMD = "#2a78d6"   # 指令
C_ACH = "#e87ba4"   # 着地
WAIT_ALL_ONPREM = 2286.43024


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
    npz_path = sys.argv[1]
    out = sys.argv[2]
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 50000

    d = np.load(npz_path)
    cmd_r, ach = d["commands"], d["points"]
    # desired_return -> (cost, avg_wait)
    cmd = np.stack([-cmd_r[:, 1], -cmd_r[:, 0] / n_jobs], axis=1)

    # [訂正 2026-08-28] 当初「待ち0は到達不能」として除外していたが誤り。同じrunのPhase1ランダムに
    # objective_values=[1.958e10, 944226, 0.0] が実在し、workload較正も PCN_WORKLOAD_WAIT_CL=0
    # (全クラウド端の待ち=0)。全部クラウドに出せば待ちは0になる。よって除外しない。
    impossible = np.zeros(len(cmd), bool)
    miss = np.linalg.norm(
        np.stack([(ach[:, 0] - cmd[:, 0]) / 1.95819e10,
                  (ach[:, 1] - cmd[:, 1]) / WAIT_ALL_ONPREM], axis=1), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor="#fcfcfb")
    fig.suptitle("指令した点 → 実際に着地した点（fast100 の最終モデル・外から切った6×6格子）",
                 fontsize=14, fontweight="bold", color="#2b2b26", x=0.045, ha="left")

    for ax, (lo, hi, ttl) in zip(axes, [
        (None, None, "A. 全36指令（矢の根が指令、先が着地）"),
        (0, 8e9, "B. 安い側の拡大"),
    ]):
        sel = np.ones(len(cmd), bool) if lo is None else (~impossible)
        for i in np.where(sel)[0]:
            ax.annotate("", xy=(ach[i, 0], ach[i, 1]), xytext=(cmd[i, 0], cmd[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="#b8b8ae", lw=1.2,
                                        shrinkA=4, shrinkB=4), zorder=2)
        ax.plot(cmd[sel & ~impossible, 0], cmd[sel & ~impossible, 1], "o", color=C_CMD,
                ms=8, markerfacecolor="none", markeredgewidth=1.8, zorder=3, label="指令（到達可能）")
        if sel.all() and impossible.any():
            ax.plot(cmd[impossible, 0], cmd[impossible, 1], "x", color="#9ca3af",
                    ms=9, markeredgewidth=2, zorder=3, label="指令（待ち0＝到達不能）")
        ax.plot(ach[sel, 0], ach[sel, 1], "o", color=C_ACH, ms=8, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2, label="着地")
        style(ax, ttl)
        if lo is not None:
            ax.set_xlim(lo, hi)
        ax.legend(frameon=False, fontsize=10, loc="upper right")

    fig.text(0.045, 0.015,
             f"※矢が短いほど「言うことを聞いている」。全36指令の正規化ズレ: "
             f"中央値 {np.median(miss):.3f} / 最大 {miss.max():.3f}"
             f"（コストは全クラウド1.958e10、待ちは2286秒で正規化した2次元距離）\n"
             "※学習時と同じ条件付け（PCN_COMMAND_BALANCE=1 / logexpand）で測定。"
             "これを外すと指令が10倍ずれてFourierがエイリアスを起こし、測定が無効になる。",
             fontsize=9, color="#5c5c55", ha="left", va="bottom")

    fig.tight_layout(rect=(0.006, 0.06, 0.995, 0.955))
    fig.savefig(out, dpi=140, facecolor="#fcfcfb")
    print(f"saved: {out}")
    print(f"  到達可能な指令 {int((~impossible).sum())} / 全 {len(cmd)}")
    print(f"  正規化ズレ 中央値={np.median(miss[~impossible]):.4f} "
          f"最大={miss[~impossible].max():.4f} 最小={miss[~impossible].min():.4f}")
    print("  指令コスト帯ごとの中央ズレ:")
    for lo, hi in [(0, 2e9), (2e9, 5e9), (5e9, 1.2e10)]:
        m = (~impossible) & (cmd[:, 0] >= lo) & (cmd[:, 0] < hi)
        if m.any():
            print(f"    cost[{lo:.0e},{hi:.0e}) n={int(m.sum())} 中央ズレ={np.median(miss[m]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
