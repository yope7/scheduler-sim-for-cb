#!/usr/bin/env python3
"""「n^2.6 の壁」の根拠と、その正体を1枚にまとめる。

配置探索の総コストは 2つの因子の積で決まる:
   t  ≈  ΣR            ×   (1イベントあたりの単価)
        ~~~~~~~~~~~~       ~~~~~~~~~~~~~~~~~~~~~~~~
        仕事量:            当時env は候補ループが
        R(生存イベント数)   イベント列を何度も走査するため
        が刈れないと n²     n と共に単価が増える

Panel A: 1エピソードの env 実行時間 vs n (log-log, 冪指数フィット)
Panel B: 生存イベント数 R の平均 vs n  (刈り取れるか＝飽和するか)
Panel C: 1イベントあたりの単価 t/ΣR vs n (当時env だけ単価が増える)

usage: PYTHONPATH=. .venv/bin/python scripts/plot_alloc_scaling_summary.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
except Exception:
    pass
import matplotlib.pyplot as plt

B = "results/bench"
OUT = os.environ.get("OUT", "docs/figures/alloc_scaling_evidence.png")
POLICY = "mix"  # プローブ(P=0.5)と条件を揃える

# (label, era, workload, setting, time-json, color, linestyle)
SERIES = [
    ("当時env・実trace・過負荷",  "legacy",  "trace", "over",    f"{B}/alloc_scaling_trace_legacy.json", "#8e0b1e", "-o"),
    ("当時env・合成・過負荷",      "legacy",  "synth", "over",    f"{B}/alloc_scaling_synth_legacy.json", "#e0545c", "-o"),
    ("現在env・実trace・過負荷",  "current", "trace", "over",    f"{B}/alloc_scaling_trace_current.json", "#c96a12", "--s"),
    ("現在env・合成・過負荷",      "current", "synth", "over",    f"{B}/alloc_scaling.json",              "#f0b050", "--s"),
    ("現在env・実trace・正容量",  "current", "trace", "correct", f"{B}/alloc_scaling_trace_current.json", "#7c5cff", "-.^"),
    ("現在env・合成・正容量",      "current", "synth", "correct", f"{B}/alloc_scaling.json",              "#2563eb", "-.^"),
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fit(xs, ys, x_min=256):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = xs >= x_min
    if m.sum() < 2:
        m = np.ones_like(xs, bool)
    return float(np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)[0])


def main():
    growth = {}
    for era, p in (("current", f"{B}/event_growth_current.json"),
                   ("legacy", f"{B}/event_growth_legacy.json")):
        d = load(p)
        if d:
            for r in d["rows"]:
                growth[(era, r["workload"], r["setting"], r["n_jobs"])] = r

    data = []
    for lab, era, wl, st, path, col, ls in SERIES:
        d = load(path)
        if not d:
            continue
        rs = [r for r in d["rows"] if r["setting"] == st and r["policy"] == POLICY]
        if len(rs) < 2:
            continue
        data.append(dict(lab=lab, era=era, wl=wl, st=st, col=col, ls=ls,
                         ns=[r["n_jobs"] for r in rs],
                         ts=[r["sec_per_episode"] for r in rs]))

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.7))

    # ---- A: time vs n ----
    ax = axes[0]
    for d in data:
        b = fit(d["ns"], d["ts"])
        ax.loglog(d["ns"], d["ts"], d["ls"], color=d["col"], ms=5, lw=1.7,
                  label=f"{d['lab']}   n^{b:.2f}")
    nref = np.array([100.0, 9000.0])
    for expo, lab, col in ((1.0, "n¹ (線形)", "#888"), (2.6, "n^2.6 (当時の見積り)", "#c33")):
        ax.loglog(nref, 0.05 * (nref / 128.0) ** expo, ":", color=col, lw=1.3, label=lab)
    ax.set_xlabel("ジョブ数 n"); ax.set_ylabel("1エピソードの env 実行時間 [秒]")
    ax.set_title("A. 配置探索コスト\n(方策=混合・1コア)")
    ax.grid(alpha=.3, which="both", lw=.5); ax.legend(fontsize=7.4, loc="upper left")

    # ---- B: R_mean vs n ----
    ax = axes[1]
    for d in data:
        pts = [(n, growth[(d["era"], d["wl"], d["st"], n)]["R_mean"])
               for n in d["ns"] if (d["era"], d["wl"], d["st"], n) in growth]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        b = fit(xs, ys, x_min=128)
        ax.loglog(xs, ys, d["ls"], color=d["col"], ms=5, lw=1.7, label=f"{d['lab']}   R~n^{b:.2f}")
    ax.loglog(nref, 0.5 * nref, ":", color="#c33", lw=1.3, label="R = n/2 (刈り取り不能)")
    ax.set_xlabel("ジョブ数 n"); ax.set_ylabel("生存イベント数 R (エピソード平均)")
    ax.set_title("B. 仕事量の源: R は飽和するか?\n正容量×短いジョブ だけが刈り取れる")
    ax.grid(alpha=.3, which="both", lw=.5); ax.legend(fontsize=7.4, loc="upper left")

    # ---- C: 単価 t/ΣR vs n ----
    ax = axes[2]
    for d in data:
        pts = [(n, t / growth[(d["era"], d["wl"], d["st"], n)]["R_sum"] * 1e6)
               for n, t in zip(d["ns"], d["ts"]) if (d["era"], d["wl"], d["st"], n) in growth]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        b = fit(xs, ys, x_min=128)
        ax.loglog(xs, ys, d["ls"], color=d["col"], ms=5, lw=1.7, label=f"{d['lab']}   n^{b:+.2f}")
    ax.set_xlabel("ジョブ数 n"); ax.set_ylabel("1イベントあたりの単価  t / ΣR  [μs]")
    ax.set_title("C. 単価: 当時env だけ n と共に増える\n(候補ループがイベント列を何度も走査)")
    ax.grid(alpha=.3, which="both", lw=.5); ax.legend(fontsize=7.4, loc="lower left")

    fig.suptitle("「n^2.6 の壁」の根拠と正体 —  コスト = 仕事量 ΣR（R が刈れないと n²） × 単価（当時env は n と共に増加）",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
