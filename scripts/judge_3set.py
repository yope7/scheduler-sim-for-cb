#!/usr/bin/env python
"""3点セット判定: 指標の守備範囲を重ねて盲点を消す。
  1) 絶対HV: NSGA-II込み共通真PF比のhypervolume比(%) — 壊滅を検出
  2) 追従超過: 生スイープ40点の自分の階段包絡線からの上方超過率(nd濾過なし) — 追従ノイズを検出(ユーザー目視相当)
  3) 実PF図: cost×待ちフロント + 真PF緑線 + 生スイープ散布 — 数値が見逃すものの最終防衛線

usage:
  SCALE=512 GROUPS="nup200512,hl512512,dens3512,qd512" \
  NSGA=results/eval_pf/nsga2_trace512_s0.npz OUTFIG=docs/figures/judge512.png \
  PYTHONPATH=. .venv/bin/python scripts/judge_3set.py
npz列順: col0=wait, col1=cost (eval_b2_compare.py 出力)
"""
import os
import sys

# env prefix がフック経由で落ちる環境向け: argv の KEY=VALUE を環境にマージ
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FP):
    fm.fontManager.addfont(_FP)
    matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_FP).get_name()

from src.agents.pcn_agent import get_non_dominated_inds_minimize

SCALE = os.environ.get("SCALE", "512")
GROUPS = [g.strip() for g in os.environ.get("GROUPS", "").split(",") if g.strip()]
NSGA = os.environ.get("NSGA", f"results/eval_pf/nsga2_trace{SCALE}_s0.npz")
OUTFIG = os.environ.get("OUTFIG", f"docs/figures/judge{SCALE}.png")
REPS = int(os.environ.get("REPS", "5"))


def hv2d(pts, ref):
    """plot_pf_summary_nsga2.py と同一規約(列順そのまま・ref=1.05x max)。"""
    if len(pts) == 0:
        return 0.0
    nd = pts[get_non_dominated_inds_minimize(pts)]
    front = [(a, b) for a, b in nd if a < ref[0] and b < ref[1]]
    if not front:
        return 0.0
    hv = 0.0
    prev = ref[0]
    for a, b in sorted(front, key=lambda x: -x[0]):
        hv += (prev - a) * (ref[1] - b)
        prev = a
    return hv


def follow_excess(raw):
    """生スイープ点の、自身のnd点が作る階段包絡線(cost→最小wait)からの上方超過率の平均(%)。
    nd濾過しないので「指令を追従できず上に散る」ノイズがそのまま出る。"""
    w, c = raw[:, 0], raw[:, 1]
    nd = raw[get_non_dominated_inds_minimize(raw)]
    nd = nd[np.argsort(nd[:, 1])]  # cost昇順
    ndc, ndw = nd[:, 1], nd[:, 0]
    # 階段: env(c) = min{ nd_w : nd_c <= c }(nd はcost昇順でwait降順なので累積minで単調化)
    env_w = np.minimum.accumulate(ndw)
    idx = np.searchsorted(ndc, c, side="right") - 1
    idx = np.clip(idx, 0, len(ndc) - 1)
    env = env_w[idx]
    exc = (w - env) / np.maximum(env, 1e-9)
    return float(np.mean(exc) * 100.0)


# ---- 読み込み ----
runs = {}  # grp -> {i: greedy(40,2)}
rp_all = []
for grp in GROUPS:
    g = {}
    for i in range(1, REPS + 1):
        fn = f"results/eval_pf/truepf_trace{SCALE}_{grp}_{i}_s0.npz"
        if os.path.exists(fn):
            d = np.load(fn)
            g[i] = d["greedy_0"]
            rp_all.append(d["rp_0"])
    if g:
        runs[grp] = g
    else:
        print(f"(skip {grp}: npz なし)")

nsga_pf = None
if os.path.exists(NSGA):
    nsga_pf = np.load(NSGA, allow_pickle=True)["pf"]
    print(f"NSGA-II 真PF: {NSGA} {len(nsga_pf)}点")
else:
    print(f"⚠️ NSGA npz なし ({NSGA}) — 絶対基準を欠く。PCN union のみ")

# ---- 共通真PF(絶対基準) ----
parts = [np.vstack(list(g.values())) for g in runs.values()] + rp_all
if nsga_pf is not None:
    parts.append(nsga_pf)
allpts = np.vstack(parts)
ref = np.array([allpts[:, 0].max(), allpts[:, 1].max()]) * 1.05
pf_true = allpts[get_non_dominated_inds_minimize(allpts)]
pf_true = pf_true[np.argsort(pf_true[:, 1])]
hvt = hv2d(pf_true, ref)
print(f"共通真PF {len(pf_true)}点 ref={ref}")

# ---- 集計 ----
print(f"\n{'group':<14} {'HV% mean±std':<16} {'min':<6} {'追従超過% mean':<14} per-run(HV%/exc%)")
summary = {}
for grp, g in runs.items():
    hvs, excs = [], []
    for i in sorted(g):
        hvs.append(hv2d(g[i], ref) / hvt * 100.0)
        excs.append(follow_excess(g[i]))
    summary[grp] = (hvs, excs)
    per = " ".join(f"r{i}:{h:.0f}/{e:.0f}" for i, h, e in zip(sorted(g), hvs, excs))
    print(f"{grp:<14} {np.mean(hvs):5.1f}±{np.std(hvs):4.1f}     {min(hvs):5.1f}  {np.mean(excs):8.1f}      {per}")

# ---- 実PF図 ----
n = len(runs)
if n:
    # 軸の意味(npz列順の一次データ確認済み): col0=コスト(0=全オンプレ角〜全クラウド角), col1=平均待ち時間(秒)。
    # 横軸=コスト/縦軸=平均待ち時間は本リポジトリのPF図の慣例([[always-show-pf-figure]])。
    # 縦軸が対数なのは待ち時間のレンジが~30×(1.6e4〜5.4e5s)に広がり、線形だと膝(低待ち側)が潰れて
    # 「真PFに乗っているか」が読めないため。掛け算的な差(同コストで待ち2倍)が等距離に見える利点もある。
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for ax, (grp, g) in zip(axes[0], runs.items()):
        pf_plot = pf_true[np.argsort(pf_true[:, 0])]
        ax.plot(pf_plot[:, 0], pf_plot[:, 1], "-", color="green", lw=2, label="真PF(NSGA-II込)", zorder=5)
        hvs, excs = summary[grp]
        order = np.argsort(hvs)[::-1]
        cmap = plt.cm.coolwarm_r(np.linspace(0, 1, len(order)))
        for rank, oi in enumerate(order):
            i = sorted(g)[oi]
            raw = g[i]
            nd = raw[get_non_dominated_inds_minimize(raw)]
            nd = nd[np.argsort(nd[:, 0])]
            bad = hvs[oi] < 60
            col = "crimson" if bad else cmap[rank]
            ax.step(nd[:, 0], nd[:, 1], where="post", color=col, lw=1.4,
                    label=f"r{i} HV{hvs[oi]:.0f}% exc{excs[oi]:.0f}%" + (" ←崩壊" if bad else ""))
            ax.plot(raw[:, 0], raw[:, 1], "x", color=col, ms=3, alpha=0.45)
        ax.set_title(f"{grp}  HV {np.mean(hvs):.1f}±{np.std(hvs):.1f}%  超過{np.mean(excs):.0f}%")
        ax.set_xlabel("コスト (クラウド使用量)")
        # JUDGE_YSCALE=linear で線形軸(既定log)。線形は突出の実寸・膝の絶対位置が見やすい。
        _yscale = os.environ.get("JUDGE_YSCALE", "log")
        if _yscale == "linear":
            ax.set_ylabel("平均待ち時間 (秒・線形)")
        else:
            ax.set_ylabel("平均待ち時間 (秒・対数)")
            ax.set_yscale("log")
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle(f"trace{SCALE} 3点セット判定 (緑=真PF / ×=生スイープ40点 / 線=greedy nd front)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)
    fig.savefig(OUTFIG, dpi=130)
    print(f"\nsaved {OUTFIG}")
