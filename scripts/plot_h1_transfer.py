#!/usr/bin/env python3
"""H1「大は小を兼ねる」検証の採点+作図。

win1024(1024学習)を256/512へhorizon縮小適用したPF(転移) vs 各スケール専用学習のPF(専用)を、
真PF(NSGA + 厳密cost0端点)の上で比較する。指標は eval_done_criteria.py と同式
(端点正規化距離 / 中央gap / HV比)。

usage: PYTHONPATH=. OBS_URGENCY=0 OBS_OCCUPANCY=1 .venv/bin/python scripts/plot_h1_transfer.py
入力: truepf_h1_{1024to256,cb05_256,1024to512,win512_512}.npz (eval_b2_compare.py 出力)
出力: pf_h1_transfer.png + 指標の標準出力
"""
import os
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
if os.environ.get("OBS_OCCUPANCY", "1") == "1":
    os.environ.setdefault("SCHEDULER_OBS_OCCUPANCY", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
import matplotlib.pyplot as plt

from scripts.eval_done_criteria import true_endpoints, hv2d
from src.agents.pcn_agent import get_non_dominated_inds_minimize as nd

CASES = [
    dict(nj=256, cfg="experiments/distributed_pcn/job_trace_256_pcn.yml",
         nsga="results/eval_pf/nsga2_trace256_s0.npz",
         transfer="truepf_h1_1024to256.npz", transfer_label="win1024→256 素転移",
         h1b="truepf_h1b_1024to256.npz", h1b_label="win1024→256 正規化合わせ",
         spec="truepf_h1_cb05_256.npz", spec_label="cb05 (256専用)"),
    dict(nj=512, cfg="experiments/distributed_pcn/job_trace_512_pcn.yml",
         nsga="results/eval_pf/nsga2_trace512_s0.npz",
         transfer="truepf_h1_1024to512.npz", transfer_label="win1024→512 素転移",
         h1b="truepf_h1b_1024to512.npz", h1b_label="win1024→512 正規化合わせ",
         spec="truepf_h1_win512_512.npz", spec_label="win512 (512専用)"),
]

SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "pf_h1_transfer.png")


def load_pts(npz_path):
    z = np.load(npz_path)
    g = z[f"greedy_{SEED}"]  # (NCMD,2)=[cost,avg_wait]
    rp = z[f"rp_{SEED}"]
    return g, rp


def metrics(ach, truepf, ref_nsga):
    """eval_done_criteria と同式。ach=(N,2) 到達点全体(非支配化前)。"""
    cheap = truepf[truepf[:, 0].argmin()]
    fast = truepf[truepf[:, 1].argmin()]
    cr = max(truepf[:, 0].max() - truepf[:, 0].min(), 1.0)
    wr = max(truepf[:, 1].max() - truepf[:, 1].min(), 1.0)

    def ndist(p):
        d = np.sqrt(((ach[:, 0] - p[0]) / cr) ** 2 + ((ach[:, 1] - p[1]) / wr) ** 2)
        return float(d.min())

    gaps = np.array([ndist(p) for p in ref_nsga])
    ref_pt = np.array([truepf[:, 0].max() * 1.02, truepf[:, 1].max() * 1.02])
    ach_pf = ach[nd(ach)]
    return dict(
        n_pf=len(ach_pf),
        d_cheap=ndist(cheap), d_fast=ndist(fast),
        gap_mean=float(gaps.mean()), gap_med=float(np.median(gaps)),
        hv_ratio=hv2d(ach_pf, ref_pt) / max(hv2d(truepf, ref_pt), 1e-9),
    )


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    print(f"{'case':<26} {'n_pf':>4} {'端cheap':>8} {'端fast':>8} {'gap平均':>8} {'gap中央':>8} {'HV比':>6}")
    for ax, cs in zip(axes, CASES):
        ep_cheap, _ = true_endpoints(cs["cfg"], cs["nj"])
        zc = np.load(cs["nsga"])
        ref = zc["pf"]
        ref = ref[nd(ref)]
        truepf = np.vstack([ref, ep_cheap[None]])
        truepf = truepf[nd(truepf)]
        tp = truepf[np.argsort(truepf[:, 0])]

        g_tr, rp_tr = load_pts(cs["transfer"])
        g_sp, _ = load_pts(cs["spec"])
        m_tr = metrics(g_tr, truepf, ref)
        m_sp = metrics(g_sp, truepf, ref)
        rows = [(cs["transfer_label"], m_tr), (cs["spec_label"], m_sp)]
        g_hb = m_hb = None
        if cs.get("h1b") and os.path.exists(cs["h1b"]):
            g_hb, _ = load_pts(cs["h1b"])
            m_hb = metrics(g_hb, truepf, ref)
            rows.insert(1, (cs["h1b_label"], m_hb))
        for name, m in rows:
            print(f"{name:<26} {m['n_pf']:>4} {m['d_cheap']:>8.3f} {m['d_fast']:>8.3f} "
                  f"{m['gap_mean']:>8.3f} {m['gap_med']:>8.3f} {m['hv_ratio']:>6.3f}")

        ax.plot(tp[:, 0], tp[:, 1], "--", color="#16a34a", lw=1.8, zorder=1,
                label="真PF (NSGA+厳密端点)")
        ax.scatter(rp_tr[:, 0], rp_tr[:, 1], s=10, c="#9aa0b0", marker="x", alpha=0.5,
                   zorder=2, label="random-p掃引")
        ax.scatter(g_sp[:, 0], g_sp[:, 1], s=46, marker="^", facecolors="none",
                   edgecolors="#2563eb", lw=1.6, zorder=3,
                   label=f"{cs['spec_label']}  HV比{m_sp['hv_ratio']:.2f}")
        ax.scatter(g_tr[:, 0], g_tr[:, 1], s=40, marker="o", facecolors="none",
                   edgecolors="#d94a4a", lw=1.6, zorder=4,
                   label=f"{cs['transfer_label']}  HV比{m_tr['hv_ratio']:.2f}")
        if g_hb is not None:
            ax.scatter(g_hb[:, 0], g_hb[:, 1], s=34, marker="o", c="#7c3aed",
                       alpha=0.85, zorder=5,
                       label=f"{cs['h1b_label']}  HV比{m_hb['hv_ratio']:.2f}")
        ax.set_title(f"n={cs['nj']} (trace, 同一CSV入れ子)")
        ax.set_xlabel("クラウドコスト (総額)")
        ax.set_ylabel("平均待ち時間")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("H1: 1024学習モデルのhorizon縮小転移 vs 各スケール専用学習 (greedy一発再現, seed0)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
