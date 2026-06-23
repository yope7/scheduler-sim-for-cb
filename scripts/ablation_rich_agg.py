#!/usr/bin/env python3
"""リッチeval(rich48b)の3seed JSONから、平均だけでなく seed間のブレ(標準偏差)も集計する。
- セル別: hv / cmd_dist / spacing の mean±std(n=3, ddof=1)。stdが小さい=seed運に左右されず安定=良い。
- 主効果: 各機能ON8セル vs OFF8セルで、(a)平均の差 と (b)seed間stdの平均 を比較。
  (b)が「その機能が結果を安定させるか(ブレを減らすか)」を直接測る=ユーザー要望の「ブレも評価」。
入力: /tmp/rich48b_fc{FWED}_{1,2,3}.json の hv/cmd_dist/spacing。再評価なし。
"""
import json, os
import numpy as np

NAMES = ["フーリエ", "重みサンプル", "探索チューニング", "後回し"]


def cell_vals(tag):
    """そのセルの3seedの (hv, cmd_dist, spacing) を集める。"""
    hv, cd, sp = [], [], []
    for i in [1, 2, 3]:
        p = f"/tmp/rich48b_{tag}_{i}.json"
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
            if d.get("hv") is not None: hv.append(float(d["hv"]))
            if d.get("cmd_dist") is not None: cd.append(float(d["cmd_dist"]))
            if d.get("spacing") is not None: sp.append(float(d["spacing"]))
        except Exception:
            pass
    return hv, cd, sp


def ms(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0
    return m, s, len(v)


cells = {}
print("=== セル別 平均±seed間std (trace256, 3seed) ===")
print("F=フーリエ W=重みサンプル E=探索チューニング D=後回し   ±はseed間std(小=安定=良)")
print(f"{'F W E D':8s} | {'HV(平均±std)':18s} {'追従距離(平均±std)':20s} {'Spacing(平均±std)':18s}")
for F in [0, 1]:
 for W in [0, 1]:
  for E in [0, 1]:
   for D in [0, 1]:
    tag = f"fc{F}{W}{E}{D}"
    hv, cd, sp = cell_vals(tag)
    hm, hs, _ = ms(hv); cm, cs, _ = ms(cd); sm, ss, _ = ms(sp)
    cells[(F, W, E, D)] = dict(hv=hv, cd=cd, sp=sp, hm=hm, hs=hs, cm=cm, cs=cs, sm=sm, ss=ss)
    def f(m, s): return f"{m:.3f}±{s:.3f}" if m is not None else "   -   "
    print(f"{F} {W} {E} {D}   | {f(hm,hs):18s} {f(cm,cs):20s} {f(sm,ss):18s}")

print("\n=== 主効果: 平均 と seed間std の両方 (ON8セル vs OFF8セル) ===")
print("平均差=効きの大きさ / std差=安定性への効き(マイナス=ブレを減らす=良い)")
for idx in range(4):
    print(f"\n● {NAMES[idx]}")
    for key, lab, better in [("hv", "HV", "大"), ("cd", "追従距離", "小"), ("sp", "Spacing", "小")]:
        on_m = [cells[k][{'hv':'hm','cd':'cm','sp':'sm'}[key]] for k in cells if k[idx] == 1]
        off_m = [cells[k][{'hv':'hm','cd':'cm','sp':'sm'}[key]] for k in cells if k[idx] == 0]
        on_s = [cells[k][{'hv':'hs','cd':'cs','sp':'ss'}[key]] for k in cells if k[idx] == 1]
        off_s = [cells[k][{'hv':'hs','cd':'cs','sp':'ss'}[key]] for k in cells if k[idx] == 0]
        on_m = [x for x in on_m if x is not None]; off_m = [x for x in off_m if x is not None]
        on_s = [x for x in on_s if x is not None]; off_s = [x for x in off_s if x is not None]
        mdiff = np.mean(on_m) - np.mean(off_m)
        sdiff = np.mean(on_s) - np.mean(off_s)
        print(f"   {lab:9s} 平均 {np.mean(on_m):.3f}→{np.mean(off_m):.3f} (差{mdiff:+.3f}, 良={better}) | "
              f"seed間std {np.mean(off_s):.3f}→{np.mean(on_s):.3f} (差{sdiff:+.3f}, ブレ{'減' if sdiff<0 else '増'})")
