#!/usr/bin/env python3
"""アブレーション各セルを真の品質メトリクスで集計(再学習なし, 保存データから)。
- cmd_dist: 指令→達成の正規化MSE距離(cmd_track_history.json 最終iter mse_cost)。小さいほど良=注文に従えてる。
- HV: 達成eval PF(pcn_mo_hv.json 最終front)の固定参照点に対するハイパーボリューム。大きいほど良(収束+広がり)。
- spread→Spacing(論文式18): min-max正規化目的空間の最近傍L1距離の標準偏差。小さいほど均等=良い。
- n_pf: uniform_cmd の非支配点数(従来の代理指標, 参考)。
trace256 固定参照: cost_ref=5.56e8, wait_ref=1.6e5。
"""
import json, glob, os
import numpy as np

COST_REF = 5.56e8
WAIT_REF = 1.6e5
COST_SCALE = 5.56e8


def spacing(front, cost_range=COST_SCALE, wait_range=WAIT_REF):
    """論文式(18) Spacing: min-max正規化目的空間で各点の最近傍L1距離 d_i の標準偏差。
    S=sqrt(1/(N-1) Σ(d_i-d̄)^2)。小さいほど均等=良い。点が少ない(<3)とNone。"""
    f = np.asarray(front, float)
    f = f[np.lexsort((f[:, 1], f[:, 0]))]
    f = np.unique(f, axis=0)
    if len(f) < 3:
        return None
    fn = np.column_stack([f[:, 0] / cost_range, f[:, 1] / wait_range])
    d = []
    for i in range(len(fn)):
        dd = [abs(fn[i, 0] - fn[j, 0]) + abs(fn[i, 1] - fn[j, 1]) for j in range(len(fn)) if j != i]
        d.append(min(dd))
    d = np.array(d); dbar = d.mean()
    return float(np.sqrt(np.sum((d - dbar) ** 2) / (len(d) - 1)))


def hv2d(front, ref=(COST_REF, WAIT_REF)):
    f = np.asarray(front, float)
    f = f[(f[:, 0] < ref[0]) & (f[:, 1] < ref[1])]
    if not len(f):
        return 0.0
    f = f[np.argsort(-f[:, 0])]
    hv = 0.0; prev = ref[0]
    for c, w in f.tolist():
        hv += (prev - c) * (ref[1] - w); prev = c
    return hv / (COST_REF * WAIT_REF)  # 正規化[0,1]


def metrics_for_run(run_glob):
    subs = sorted(glob.glob(run_glob + "/2026*"))
    if not subs:
        return None
    sub = subs[-1]
    # cmd_dist
    cmd_dist = None
    p = f"{sub}/cmd_track_history.json"
    if os.path.exists(p):
        try:
            h = json.load(open(p))
            if h:
                cmd_dist = float(h[-1].get("mse_cost", h[-1].get("mse_total")))
        except Exception:
            pass
    # HV + spacing(論文式18) from eval PF
    hv = spread = None
    p = f"{sub}/pcn_mo_hv.json"
    if os.path.exists(p):
        try:
            d = json.load(open(p)); pf = np.asarray(d["pareto_fronts_per_eval"][-1], float)
            hv = hv2d(pf)
            spread = spacing(pf)  # Spacing: 小さいほど均等=良い
        except Exception:
            pass
    # n_pf
    n_pf = None
    js = sorted(glob.glob(f"{sub}/uniform_cmd_stats_iter_*.json"))
    if js:
        try:
            n_pf = int([l for l in open(js[-1]) if "n_pf" in l][0].split(":")[1].strip().rstrip(","))
        except Exception:
            pass
    return dict(cmd_dist=cmd_dist, hv=hv, spread=spread, n_pf=n_pf)


def agg(vals):
    v = [x for x in vals if x is not None]
    return (np.mean(v) if v else None), v


print("=== アブレーション セル別メトリクス (trace256) ===")
print("F=フーリエ W=重みサンプル E=探索チューニング D=後回し")
print(f"{'F W E D':8s} | {'cmd距離↓':9s} {'HV↑':8s} {'Spacing↓':8s} {'n_pf':6s} | 完了")
cells = {}
for F in [0, 1]:
 for W in [0, 1]:
  for E in [0, 1]:
   for D in [0, 1]:
    tag = f"fc{F}{W}{E}{D}"
    cd, hv, sp, npf = [], [], [], []
    ndone = 0
    for i in [1, 2, 3]:
        m = metrics_for_run(f"experiments/distributed_pcn/run_synth256_{tag}_{i}")
        if not m:
            continue
        if m["n_pf"] is not None and os.path.exists(sorted(glob.glob(f"experiments/distributed_pcn/run_synth256_{tag}_{i}/2026*"))[-1] + "/uniform_cmd_stats_iter_100.json"):
            ndone += 1
        cd.append(m["cmd_dist"]); hv.append(m["hv"]); sp.append(m["spread"]); npf.append(m["n_pf"])
    mcd, _ = agg(cd); mhv, _ = agg(hv); msp, _ = agg(sp); mnp, _ = agg(npf)
    cells[(F, W, E, D)] = dict(cmd=mcd, hv=mhv, sp=msp, npf=mnp, ndone=ndone)
    def fmt(x, f): return (f % x) if x is not None else "  -  "
    print(f"{F} {W} {E} {D}   | {fmt(mcd,'%9.3f')} {fmt(mhv,'%8.4f')} {fmt(msp,'%8.2f')} {fmt(mnp,'%6.1f')} | {ndone}/3")

print("\n=== 各機能の主効果(ON平均 − OFF平均) ===")
names = ["フーリエ", "重みサンプル", "探索チューニング", "後回し"]
for idx in range(4):
    for met, lab, better in [("cmd", "cmd距離", "小"), ("hv", "HV", "大"), ("npf", "n_pf", "大")]:
        on = [c[met] for k, c in cells.items() if k[idx] == 1 and c[met] is not None]
        off = [c[met] for k, c in cells.items() if k[idx] == 0 and c[met] is not None]
        if on and off:
            d = np.mean(on) - np.mean(off)
            print(f"  {names[idx]:14s} {lab:7s}: ON {np.mean(on):.3f} vs OFF {np.mean(off):.3f}  差={d:+.3f} (良い向き={better})")
    print()
