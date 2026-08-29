#!/usr/bin/env python3
"""N混合スクリーンの採点+作図: 混合1モデル vs 各N専用学習 を N別に比較。

共通真PF = 非支配( 全モデルの rp ∪ samp ∪ greedy )  … スクリーン流儀(NSGA無しの synth 用)
指標:
  gap  = mean( clip(greedy_wait − interp(greedy_cost→truePF_wait), 0) ) / truePF_wait_range
         (current_state.md §4 の設計判断4と同式)
  HV比 = HV(greedy非支配) / HV(共通真PF)   (参照点=真PF max×1.02)

usage: PYTHONPATH=. .venv/bin/python scripts/plot_nmix_screen.py
入力: truepf_nm_mix_r{1..3}_n{N}.npz / truepf_nm_ded{N}_r{1..3}.npz
出力: pf_nmix_screen.png + 表
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
import matplotlib.pyplot as plt

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
from src.agents.pcn_agent import get_non_dominated_inds_minimize as nd
from scripts.eval_done_criteria import hv2d

NS = [128, 256, 512]
REPS = [1, 2, 3]
SEED = 0
OUT = os.environ.get("OUT", "pf_nmix_screen.png")


def load(path):
    z = np.load(path)
    return z[f"greedy_{SEED}"], z[f"samp_{SEED}"], z[f"rp_{SEED}"]


def gap_to_truepf(greedy, tp):
    """tp: cost昇順の共通真PF。greedy各点の wait 超過を正規化して平均。"""
    wr = max(tp[:, 1].max() - tp[:, 1].min(), 1.0)
    interp_w = np.interp(greedy[:, 0], tp[:, 0], tp[:, 1])
    return float(np.clip(greedy[:, 1] - interp_w, 0, None).mean() / wr)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    print(f"{'N':>4} {'モデル':<10} {'rep別gap':<24} {'gap中央値':>9} {'rep別HV比':<24} {'HV比中央値':>9}")
    summary = {}
    for ax, n in zip(axes, NS):
        mixes = [load(f"truepf_nm_mix_r{r}_n{n}.npz") for r in REPS]
        deds = [load(f"truepf_nm_ded{n}_r{r}.npz") for r in REPS]
        pool = np.vstack([np.vstack([g, s, rp]) for g, s, rp in mixes + deds])
        tp = pool[nd(pool)]
        tp = tp[np.argsort(tp[:, 0])]
        ref_pt = np.array([tp[:, 0].max() * 1.02, tp[:, 1].max() * 1.02])
        hv_tp = hv2d(tp, ref_pt)

        stats = {}
        for label, runs in [("mix", mixes), ("ded", deds)]:
            gaps, hvs = [], []
            for g, _, _ in runs:
                gaps.append(gap_to_truepf(g, tp))
                gnd = g[nd(g)]
                hvs.append(hv2d(gnd, ref_pt) / max(hv_tp, 1e-9))
            stats[label] = (gaps, hvs)
            print(f"{n:>4} {label:<10} {' '.join(f'{x:.3f}' for x in gaps):<24} "
                  f"{np.median(gaps):>9.3f} {' '.join(f'{x:.3f}' for x in hvs):<24} "
                  f"{np.median(hvs):>9.3f}")
        summary[n] = stats

        ax.plot(tp[:, 0], tp[:, 1], "--", color="#16a34a", lw=1.8, zorder=1,
                label="共通真PF (全モデル∪rp∪samp)")
        for i, (g, _, _) in enumerate(deds):
            ax.scatter(g[:, 0], g[:, 1], s=42, marker="^", facecolors="none",
                       edgecolors="#2563eb", lw=1.4, alpha=0.75, zorder=3,
                       label=(f"専用 nded{n} ×3rep  gap中央"
                              f"{np.median(summary[n]['ded'][0]):.3f}") if i == 0 else None)
        for i, (g, _, _) in enumerate(mixes):
            ax.scatter(g[:, 0], g[:, 1], s=34, marker="o", c="#7c3aed",
                       alpha=0.75, zorder=4,
                       label=(f"混合 nmix ×3rep  gap中央"
                              f"{np.median(summary[n]['mix'][0]):.3f}") if i == 0 else None)
        ax.set_title(f"N={n} で推論 (synth, 学習と同一インスタンス)")
        ax.set_xlabel("クラウドコスト (総額)")
        ax.set_ylabel("平均待ち時間")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("N混合学習スクリーン: 混合1モデル(N∈{128,256,512}, 基準512) vs 各N専用学習 (greedy, 3rep全表示)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
