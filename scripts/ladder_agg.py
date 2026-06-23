#!/usr/bin/env python3
"""ladder(段階的削減)アブレーションの rich eval を集計。mean±seed間std(ddof=1)。
env: PFX(=lad_s/lad_t) STAGES。/tmp/ladrich_{PFX}_{stage}_{i}.json の最終JSON行を読む。
"""
import json, os
import numpy as np

PFX = os.environ.get("PFX", "lad_s")
STAGES = os.environ.get("STAGES", "FWD FD F base").split()
NAMES = {"FWD": "F+W+D (最良)", "FD": "F+D (密度W OFF)", "F": "F のみ (defer OFF)", "base": "baseline (全OFF)"}


def load(tag, i):
    p = f"/tmp/ladrich_{tag}_{i}.json"
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


print(f"=== 段階的削減アブレーション集計  PFX={PFX} ===")
print(f"{'構成':24s} | {'HV(平均±std)':16s} {'追従(平均±std)':16s} {'Spacing':12s} {'n_pf':5s} | eval")
for st in STAGES:
    tag = f"{PFX}_{st}"; hv = []; cd = []; sp = []; npf = []
    for i in [1, 2, 3]:
        d = load(tag, i)
        if d:
            hv.append(d.get("hv")); cd.append(d.get("cmd_dist")); sp.append(d.get("spacing")); npf.append(d.get("n_pf"))
    hm, hs, n = ms(hv); cm, cs, _ = ms(cd); sm, ss, _ = ms(sp); npm, _, _ = ms(npf)
    def f(m, s): return f"{m:.3f}±{s:.3f}" if m is not None else "  -  "
    print(f"{NAMES.get(st, st):24s} | {f(hm,hs):16s} {f(cm,cs):16s} {f(sm,ss):12s} {('%.0f'%npm) if npm else '-':5s} | {n}/3")
