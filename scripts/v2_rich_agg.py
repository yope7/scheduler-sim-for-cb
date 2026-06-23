#!/usr/bin/env python3
"""Ver.2(fdタグ)の rich eval 結果を集計。平均±seed間std(ddof=1)。
重要: W=OFF かつ D=OFF のセルは Ver.1(fc)と完全同条件(重み無効・defer無し)なので
      旧 /tmp/rich48b_fc{...}.json を流用する(再eval不要・ユーザー指摘)。
      それ以外は新 /tmp/v2rich_fd{...}.json(stdoutにログ混在のため最終JSON行を読む)。
3/3 揃ったセルのみ確定表示。
"""
import os
import json
import numpy as np

NAMES = ["フーリエ", "密度重み", "探索チューニング", "後回し(offset1)"]


def _read_json_tail(p):
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p) if l.strip().startswith("{")]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1])
        return d if "hv" in d else None
    except Exception:
        return None


def load_cell(F, W, E, D, i):
    reuse = (W == 0 and D == 0)            # Ver.1と同条件 → 旧流用
    if reuse:
        return _read_json_tail(f"/tmp/rich48b_fc{F}{W}{E}{D}_{i}.json"), True
    return _read_json_tail(f"/tmp/v2rich_fd{F}{W}{E}{D}_{i}.json"), False


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0), len(v)


cells = {}
print("=== Ver.2(密度W + offset1) セル別 平均±seed間std (trace256) ===")
print("F=フーリエ W=密度重み E=探索 D=後回し(offset1)  [流用]=W,D共にOFF→Ver.1から流用")
print(f"{'F W E D':8s} | {'HV(平均±std)':16s} {'追従(平均±std)':16s} {'Spacing':14s} {'n_pf':5s} | eval")
for F in [0, 1]:
 for W in [0, 1]:
  for E in [0, 1]:
   for D in [0, 1]:
    hv, cd, sp, npf = [], [], [], []
    reuse = False
    for i in [1, 2, 3]:
        d, reuse = load_cell(F, W, E, D, i)
        if d:
            hv.append(d.get("hv")); cd.append(d.get("cmd_dist")); sp.append(d.get("spacing")); npf.append(d.get("n_pf"))
    hm, hs, n = ms(hv); cm, cs, _ = ms(cd); sm, ss, _ = ms(sp); npm, _, _ = ms(npf)
    cells[(F, W, E, D)] = dict(hm=hm, hs=hs, cm=cm, cs=cs, sm=sm, ss=ss, npm=npm, n=n, reuse=reuse)
    if n == 0:
        continue
    def f(m, s): return f"{m:.3f}±{s:.3f}" if m is not None else "   -   "
    tag_note = "[流用]" if reuse else ""
    print(f"{F} {W} {E} {D}   | {f(hm,hs):16s} {f(cm,cs):16s} {f(sm,ss):14s} {('%.0f'%npm) if npm else '-':5s} | {n}/3 {tag_note}")

ndone = sum(1 for c in cells.values() if c["n"] == 3)
print(f"\n3/3 確定セル: {ndone}/16 (うち流用={sum(1 for c in cells.values() if c['reuse'] and c['n']==3)})")

full = {k: c for k, c in cells.items() if c["n"] == 3}
if len(full) == 16:
    print("\n=== 主効果(Ver.2内, ON8 vs OFF8) 平均 と seed間std ===")
    for idx in range(4):
        print(f"● {NAMES[idx]}")
        for key, sk, lab, better in [("hm", "hs", "HV", "大"), ("cm", "cs", "追従", "小"), ("sm", "ss", "Spacing", "小")]:
            on = [c[key] for k, c in full.items() if k[idx] == 1 and c[key] is not None]
            off = [c[key] for k, c in full.items() if k[idx] == 0 and c[key] is not None]
            ons = [c[sk] for k, c in full.items() if k[idx] == 1 and c[sk] is not None]
            offs = [c[sk] for k, c in full.items() if k[idx] == 0 and c[sk] is not None]
            d = np.mean(on) - np.mean(off)
            sd = np.mean(ons) - np.mean(offs)
            print(f"   {lab:7s}(良={better}): ON {np.mean(on):.3f} / OFF {np.mean(off):.3f}  差={d:+.3f} | "
                  f"seed間std ON {np.mean(ons):.3f} / OFF {np.mean(offs):.3f} 差={sd:+.3f}")
