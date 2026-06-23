#!/usr/bin/env python3
"""16セル(2^4)の実PF図を 4x4 グリッド(表の形)に並べる。
行=FW(フーリエ,重みサンプル) 列=ED(探索チューニング,後回し)。各マス=そのセルの代表seedのPF散布図。
未完セルは「測定中」。完了したマスから順次埋まる。出力 docs/figures/factorial_pf_grid.png
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COST_SCALE = 5.56e8


def best_front(tag):
    """そのセルの全seedから、最終eval PF(点+n_pf最大)を代表として返す。"""
    best = None; best_np = -1; best_cmd = None
    for i in [1, 2, 3]:
        subs = sorted(glob.glob(f"experiments/distributed_pcn/run_synth256_{tag}_{i}/2026*"))
        if not subs:
            continue
        sub = subs[-1]
        if not os.path.exists(f"{sub}/uniform_cmd_stats_iter_100.json"):
            continue
        try:
            d = json.load(open(f"{sub}/pcn_mo_hv.json"))
            pf = np.asarray(d["pareto_fronts_per_eval"][-1], float)
            npf = int([l for l in open(f"{sub}/uniform_cmd_stats_iter_100.json") if "n_pf" in l][0].split(":")[1].strip().rstrip(","))
            h = json.load(open(f"{sub}/cmd_track_history.json"))
            cmd = float(h[-1]["mse_cost"]) if h else None
            if npf > best_np:
                best_np = npf; best = pf; best_cmd = cmd
        except Exception:
            pass
    return best, best_np, best_cmd


FW = [(0, 0), (0, 1), (1, 0), (1, 1)]   # 行: フーリエ, 重みサンプル
ED = [(0, 0), (0, 1), (1, 0), (1, 1)]   # 列: 探索チューニング, 後回し
onoff = {0: "-", 1: "ON"}
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for r, (F, W) in enumerate(FW):
    for c, (E, D) in enumerate(ED):
        ax = axes[r][c]
        tag = f"fc{F}{W}{E}{D}"
        pf, npf, cmd = best_front(tag)
        # F=Fourier W=Weight-sampling E=Explore-tune D=Defer
        title = f"F:{onoff[F]} W:{onoff[W]} E:{onoff[E]} D:{onoff[D]}"
        if pf is not None and len(pf):
            # 非支配点のみ→コスト昇順にソート（待ちは降順になる）＝単調な綺麗なフロント線
            p = pf[(pf[:, 0] >= 0)]
            nd = []
            for k in range(len(p)):
                dominated = any((p[j, 0] <= p[k, 0]) and (p[j, 1] <= p[k, 1]) and (j != k) and
                                ((p[j, 0] < p[k, 0]) or (p[j, 1] < p[k, 1])) for j in range(len(p)))
                if not dominated:
                    nd.append(p[k])
            p = np.array(nd) if nd else p
            p = p[np.argsort(p[:, 0])]
            ax.plot(p[:, 0] / 1e8, p[:, 1] / 1e3, "-o", c="crimson", ms=3.5, lw=1.4)
            ax.set_xlim(0, 5.8); ax.set_ylim(0, 160)
            sub = f"cmd-dist={cmd:.2f}  n_pf={npf}" if cmd is not None else f"n_pf={npf}"
            ax.set_title(f"{title}\n{sub}", fontsize=11)
        else:
            ax.text(0.5, 0.5, "measuring...", ha="center", va="center", fontsize=13, color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=11, color="gray")
            ax.set_xlim(0, 5.8); ax.set_ylim(0, 160)
        ax.set_xlabel("Cost (x1e8)", fontsize=8); ax.set_ylabel("Wait (x1e3)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
fig.suptitle("2^4 ablation PF grid  rows=F(Fourier)/W(Weight-sampling), cols=E(Explore-tune)/D(Defer)   red=achieved Pareto front  (cmd-dist lower=better)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/factorial_pf_grid.png", dpi=85)
print("[SAVED] docs/figures/factorial_pf_grid.png")
ndone = sum(1 for F, W in FW for E, D in ED if best_front(f"fc{F}{W}{E}{D}")[0] is not None)
print(f"埋まったマス: {ndone}/16")
