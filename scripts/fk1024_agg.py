#!/usr/bin/env python3
"""trace1024 完全16セルアブレーションの rich eval を集計(平均±seed間std)+主効果。
/tmp/ladrich_fdk{FWED}_{i}.json の最終JSON行を読む。各2seed。
"""
import json, os
import numpy as np

NAMES = ["フーリエ", "密度重み", "探索チューニング", "後回し(offset1)"]


def load(F, W, E, D, i):
    p = f"/tmp/ladrich_fdk{F}{W}{E}{D}_{i}.json"
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1]); return d if "hv" in d else None
    except Exception:
        return None


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0), len(v)


cells = {}
print("=== trace1024 完全16セルアブレーション(各2seed) ===")
print("F=フーリエ W=密度重み E=探索 D=後回し(offset1)")
print(f"{'F W E D':8s} | {'HV(平均±std)':16s} {'追従(平均±std)':16s} {'Spacing':12s} {'n_pf':5s} | eval")
for F in [0, 1]:
 for W in [0, 1]:
  for E in [0, 1]:
   for D in [0, 1]:
    hv = []; cd = []; sp = []; npf = []
    for i in [1, 2]:
        d = load(F, W, E, D, i)
        if d:
            hv.append(d.get("hv")); cd.append(d.get("cmd_dist")); sp.append(d.get("spacing")); npf.append(d.get("n_pf"))
    hm, hs, n = ms(hv); cm, cs, _ = ms(cd); sm, ss, _ = ms(sp); npm, _, _ = ms(npf)
    cells[(F, W, E, D)] = dict(hm=hm, hs=hs, cm=cm, cs=cs, sm=sm, ss=ss, npm=npm, n=n)
    def f(m, s): return f"{m:.3f}±{s:.3f}" if m is not None else "  -  "
    print(f"{F} {W} {E} {D}   | {f(hm,hs):16s} {f(cm,cs):16s} {f(sm,ss):12s} {('%.0f'%npm) if npm else '-':5s} | {n}/2")

full = {k: c for k, c in cells.items() if c["n"] >= 1 and c["hm"] is not None}
print(f"\n確定セル: {len(full)}/16")
if len(full) >= 14:
    print("\n=== 主効果(ON8 vs OFF8) 平均 と seed間std ===")
    for idx in range(4):
        print(f"● {NAMES[idx]}")
        for key, sk, lab, better in [("hm", "hs", "HV", "大"), ("cm", "cs", "追従", "小")]:
            on = [c[key] for k, c in full.items() if k[idx] == 1 and c[key] is not None]
            off = [c[key] for k, c in full.items() if k[idx] == 0 and c[key] is not None]
            ons = [c[sk] for k, c in full.items() if k[idx] == 1 and c[sk] is not None]
            offs = [c[sk] for k, c in full.items() if k[idx] == 0 and c[sk] is not None]
            print(f"   {lab:5s}(良={better}): ON {np.mean(on):.3f} / OFF {np.mean(off):.3f} 差={np.mean(on)-np.mean(off):+.3f} | std ON{np.mean(ons):.3f}/OFF{np.mean(offs):.3f}")
