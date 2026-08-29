#!/usr/bin/env python3
"""スケール学習(1024/4096/40960)の結果比較: 生PF + 正規化PF + 時間/健全性表。"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
import matplotlib.pyplot as plt

SCALES = [1024, 4096, 40960]
COLORS = {1024: "#2563eb", 4096: "#7c3aed", 40960: "#d94a4a"}
OUT = os.environ.get("OUT", "pf_scale_chain.png")


def load(N):
    fs = sorted(glob.glob(f"experiments/distributed_pcn/run_synth{N}_scale{N}/*/uniform_cmd_pts_iter_*.npz"))
    if not fs:
        return None
    z = np.load(fs[-1])
    it = int(fs[-1].split("iter_")[-1].split(".")[0])
    return dict(pf=z["eval_pf"], iter=it)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    got = {}
    for N in SCALES:
        d = load(N)
        if d is None:
            continue
        got[N] = d
        pf = d["pf"][np.argsort(d["pf"][:, 0])]
        # 左: 生PF
        axes[0].plot(pf[:, 0], pf[:, 1], "-o", ms=3, lw=1.4, color=COLORS[N],
                     label=f"N={N} (PF {len(pf)}点, iter{d['iter']})")
        # 右: 正規化PF(コスト/最大, 待ち/最大)
        cn = pf[:, 0] / max(pf[:, 0].max(), 1e-9)
        wn = pf[:, 1] / max(pf[:, 1].max(), 1e-9)
        axes[1].plot(cn, wn, "-o", ms=3, lw=1.4, color=COLORS[N], label=f"N={N}")
    axes[0].set_title("生のPF(スケールで軸が変わる)")
    axes[0].set_xlabel("クラウドコスト(総額)"); axes[0].set_ylabel("平均待ち時間")
    axes[1].set_title("正規化PF(形の比較)")
    axes[1].set_xlabel("コスト / 最大"); axes[1].set_ylabel("待ち / 最大")
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("スケール学習: 正しい容量(オンプレ1520/クラウド65536)・小ジョブ・各N専用学習のPF")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT} (scales={list(got)})")


if __name__ == "__main__":
    main()
