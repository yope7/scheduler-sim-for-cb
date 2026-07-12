#!/usr/bin/env python3
"""探索E除外の8セル(F/W/D, E=0)アブレーション集計。trace/合成 共通(PFXで切替)。
/tmp/ladrich_{PFX}{F}{W}0{D}_{i}.json (i=1..3, "err"/欠損はスキップ) を読み、
HV/追従(cmd_dist)/Spacing/n_pf の mean±std(ddof=1) を8セル + 主効果(F/W/D ON4 vs OFF4)で集計。
結果は /tmp/lad8_{PFX}.json に保存(図・HTML生成が読む)。
usage: PFX=fdk uv run python scripts/lad8_agg.py   (trace1024)
       PFX=fsk uv run python scripts/lad8_agg.py   (合成1024)
"""
import json, os
import numpy as np

PFX = os.environ.get("PFX", "fdk")
NAMES = ["フーリエ(F)", "密度重み(W)", "後回しoffset1(D)"]


def load(F, W, D, i):
    p = f"/tmp/ladrich_{PFX}{F}{W}0{D}_{i}.json"
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1])
        return d if ("hv" in d and "err" not in d) else None
    except Exception:
        return None


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0), len(v)


cells = {}
print(f"=== {PFX} 探索E除外 8セル(F/W/D) アブレーション ===")
print(f"{'F W D':6s} | {'HV(平均±std)':16s} {'追従(平均±std)':16s} {'Spacing':10s} {'n_pf':5s} | n_seed")
for F in [0, 1]:
 for W in [0, 1]:
  for D in [0, 1]:
    hv = []; cd = []; sp = []; npf = []
    for i in [1, 2, 3]:
        d = load(F, W, D, i)
        if d:
            hv.append(d.get("hv")); cd.append(d.get("cmd_dist")); sp.append(d.get("spacing")); npf.append(d.get("n_pf"))
    hm, hs, n = ms(hv); cm, cs, _ = ms(cd); sm, ss, _ = ms(sp); npm, _, _ = ms(npf)
    cells[(F, W, D)] = dict(hm=hm, hs=hs, cm=cm, cs=cs, sm=sm, ss=ss, npm=npm, n=n)
    def f(m, s): return f"{m:.3f}±{s:.3f}" if m is not None else "  -  "
    print(f"{F} {W} {D}   | {f(hm,hs):16s} {f(cm,cs):16s} {(f'{sm:.3f}' if sm is not None else '-'):10s} {('%.0f'%npm) if npm else '-':5s} | {n}")

full = {k: c for k, c in cells.items() if c["n"] >= 1 and c["hm"] is not None}
print(f"\n確定セル: {len(full)}/8")
main = {}
if len(full) >= 6:
    print("\n=== 主効果(ON4 vs OFF4) 平均 と seed間std ===")
    for idx in range(3):
        row = {}
        print(f"● {NAMES[idx]}")
        for key, sk, lab, better in [("hm", "hs", "HV", "大"), ("cm", "cs", "追従", "小")]:
            on = [c[key] for k, c in full.items() if k[idx] == 1 and c[key] is not None]
            off = [c[key] for k, c in full.items() if k[idx] == 0 and c[key] is not None]
            ons = [c[sk] for k, c in full.items() if k[idx] == 1 and c[sk] is not None]
            offs = [c[sk] for k, c in full.items() if k[idx] == 0 and c[sk] is not None]
            on_m, off_m = float(np.mean(on)), float(np.mean(off))
            row[lab] = dict(on=on_m, off=off_m, diff=on_m - off_m,
                            std_on=float(np.mean(ons)), std_off=float(np.mean(offs)))
            print(f"   {lab:5s}(良={better}): ON {on_m:.3f} / OFF {off_m:.3f} 差={on_m-off_m:+.3f} | std ON{np.mean(ons):.3f}/OFF{np.mean(offs):.3f}")
        main[NAMES[idx]] = row

out = {"pfx": PFX,
       "cells": {f"{F}{W}{D}": cells[(F, W, D)] for F in [0, 1] for W in [0, 1] for D in [0, 1]},
       "main": main}
with open(f"/tmp/lad8_{PFX}.json", "w") as fo:
    json.dump(out, fo, ensure_ascii=False)
print(f"\n[SAVED] /tmp/lad8_{PFX}.json")
