#!/usr/bin/env python
"""真PFのcost範囲に限定したHV（箱HV）= 全比較の標準物差し。

なぜ必要か
----------
生HVは「真PFのcost上限を超えて伸びた解」が有利になる。weekA4096 の例:
  真PF の cost 上限 = 1.050e9（そこで待ち 15.8 秒＝最速）
  全クラウド送り     = 2.375e9（待ち 351.5 秒）… cost も待ちも真PFに支配される劣解
にもかかわらず、pダイヤル単体のHVを「自分の点群から決めた nadir」で測ると
cost∈[1.05e9, 2.38e9] の帯にも体積が立ち、真PFが一度も踏まない領域で稼いでしまう。
その結果 HV基準の取り方だけで 20pt 動く（同じrunが 69.6% にも 86.5% にも見える）。

標準物差し（本モジュール）
--------------------------
真PF から箱 B = [0, cost_max] x [0, wait_max] を決め、
  * 箱の外の点は捨てる（cost > cost_max も wait > wait_max も）
  * 参照点は必ず (cost_max, wait_max) 固定
  * スコア = HV(点群) / HV(真PF)
とする。これで真PF=100% が定義上の天井になり、誰も箱の外で稼げない。
100% 超えが出たら「真PFが真でない（探索不足）」というシグナルとして読む。

使い方
------
  python scripts/hv_boxed.py --truepf results/eval_pf/regime_truepf/tp_weekA4096_cap.npz \
      --npz <eval.npz> [--npz ...] [--json out.json]
ライブラリとして:
  from scripts.hv_boxed import Box, hv_box, score
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 箱とHV
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """比較の箱。cost/wait ともに最小化。ref=(cost_max, wait_max)。"""

    cost_max: float
    wait_max: float
    truepf: np.ndarray  # (N,2) 非支配化済みの真PF（箱内）
    name: str = ""

    @property
    def ref(self) -> Tuple[float, float]:
        return (self.cost_max, self.wait_max)

    @property
    def hv_truepf(self) -> float:
        return hv_box(self.truepf, self)


def non_dominated_min(pts: np.ndarray) -> np.ndarray:
    """2目的最小化の非支配点のみ（cost昇順・wait狭義単調減）。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return p
    order = np.lexsort((p[:, 1], p[:, 0]))  # cost昇順、同costならwait昇順
    p = p[order]
    keep, best = [], np.inf
    for c, w in p:
        if w < best:
            keep.append((c, w))
            best = w
    return np.asarray(keep, dtype=np.float64).reshape(-1, 2)


def clip_to_box(pts: np.ndarray, box: Box) -> np.ndarray:
    """箱の外の点を落とす（切り詰めではなく除外＝箱外で稼がせない）。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return p
    m = np.isfinite(p).all(axis=1) & (p[:, 0] <= box.cost_max) & (p[:, 1] <= box.wait_max)
    return p[m]


def hv_box(pts: np.ndarray, box: Box) -> float:
    """箱内HV（参照点 = 箱の右上角で固定）。"""
    p = non_dominated_min(clip_to_box(pts, box))
    if len(p) == 0:
        return 0.0
    hv, prev_w = 0.0, float(box.wait_max)
    for c, w in p:  # cost昇順
        if w < prev_w:
            hv += (box.cost_max - c) * (prev_w - w)
            prev_w = w
    return float(hv)


def score(pts: np.ndarray, box: Box) -> float:
    """真PF比 [%]。"""
    base = box.hv_truepf
    return 100.0 * hv_box(pts, box) / base if base > 0 else float("nan")


def coverage(pts: np.ndarray, box: Box) -> Dict[str, float]:
    """箱に対する被覆の内訳（どこで負けているかを見る用）。"""
    p_all = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = non_dominated_min(clip_to_box(p_all, box))
    n_out_cost = int((p_all[:, 0] > box.cost_max).sum()) if len(p_all) else 0
    n_out_wait = int((p_all[:, 1] > box.wait_max).sum()) if len(p_all) else 0
    return {
        "n_pts": int(len(p_all)),
        "n_in_box_nd": int(len(p)),
        "n_dropped_cost_over": n_out_cost,   # 真PF上限より高costで伸ばした点＝生HVで得していた分
        "n_dropped_wait_over": n_out_wait,
        "cost_span_frac": float((p[:, 0].max() - p[:, 0].min()) / box.cost_max) if len(p) else 0.0,
        "wait_min": float(p[:, 1].min()) if len(p) else float("nan"),
        "cost_max_used": float(p[:, 0].max()) if len(p) else float("nan"),
    }


def eq_cost_wait(pts: np.ndarray, box: Box, n: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """箱内のcost格子上で「その予算以下で出せる最小待ち」を返す（階段関数）。
    HVより読みやすい第2の物差し＝「同じお金でどれだけ待たずに済むか」。"""
    p = non_dominated_min(clip_to_box(pts, box))
    grid = np.linspace(0.0, box.cost_max, n + 1)[1:]
    out = np.full(n, np.nan)
    for i, g in enumerate(grid):
        m = p[:, 0] <= g
        if m.any():
            out[i] = p[m, 1].min()
    return grid, out


def eq_cost_ratio(a: np.ndarray, b: np.ndarray, box: Box, n: int = 25) -> float:
    """等コストでの待ち比 a/b の幾何平均（<1 なら a が b より待たない＝勝ち）。"""
    _, wa = eq_cost_wait(a, box, n)
    _, wb = eq_cost_wait(b, box, n)
    m = np.isfinite(wa) & np.isfinite(wb) & (wa > 0) & (wb > 0)
    return float(np.exp(np.mean(np.log(wa[m] / wb[m])))) if m.any() else float("nan")


# --------------------------------------------------------------------------
# 真PF / 点群の読み込み
# --------------------------------------------------------------------------
_PF_KEYS = ("pf", "true_pf", "truepf", "front", "pareto_front")


def load_truepf(path: str, name: str = "") -> Box:
    """真PF npz -> Box。cost_max/wait_max は真PFの非支配点から取る。"""
    d = np.load(path, allow_pickle=True)
    pf = None
    for k in _PF_KEYS:
        if k in d.files:
            pf = np.asarray(d[k], dtype=np.float64).reshape(-1, 2)
            break
    if pf is None:
        raise KeyError(f"{path}: 真PFのキーが見つからない (files={d.files})")
    nd = non_dominated_min(pf)
    cost_max = float(nd[:, 0].max())
    wait_max = float(nd[:, 1].max())
    box = Box(cost_max=cost_max, wait_max=wait_max, truepf=nd,
              name=name or os.path.basename(path))
    return box


def extract_point_sets(path: str) -> Dict[str, np.ndarray]:
    """npz から (N,2) の点群らしきものを全部拾う。キー名はそのまま系列名に使う。"""
    out: Dict[str, np.ndarray] = {}
    d = np.load(path, allow_pickle=True)
    for k in d.files:
        try:
            a = np.asarray(d[k], dtype=np.float64)
        except Exception:
            continue
        if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 1:
            out[k] = a
        elif a.ndim == 2 and a.shape[0] == 2 and a.shape[1] > 8:
            out[k] = a.T
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="真PF箱に限定した標準HV")
    ap.add_argument("--truepf", required=True, help="真PF npz")
    ap.add_argument("--npz", action="append", default=[], help="評価npz（複数可・globも可）")
    ap.add_argument("--keys", default="", help="使うキーをカンマ区切りで限定")
    ap.add_argument("--json", default="", help="結果JSONの出力先")
    a = ap.parse_args(argv)

    box = load_truepf(a.truepf)
    only = {k.strip() for k in a.keys.split(",") if k.strip()}
    print(f"[箱] {box.name}  cost<= {box.cost_max:.6g}  wait<= {box.wait_max:.6g}  "
          f"真PF点数={len(box.truepf)}  HV(真PF)={box.hv_truepf:.6g}")
    print(f"{'file':<44} {'key':<22} {'箱HV%':>8} {'点数':>6} {'箱内nd':>7} "
          f"{'cost超':>7} {'最小待ち':>10} {'使ったcost上限':>14}")
    rows: List[dict] = []
    files: List[str] = []
    for pat in a.npz:
        files.extend(sorted(glob.glob(pat)) or [pat])
    for f in files:
        if not os.path.isfile(f):
            print(f"{os.path.basename(f):<44} (見つからない)")
            continue
        for k, pts in extract_point_sets(f).items():
            if only and k not in only:
                continue
            s = score(pts, box)
            cv = coverage(pts, box)
            rows.append({"file": f, "key": k, "hv_pct": s, **cv})
            print(f"{os.path.basename(f):<44} {k:<22} {s:>8.2f} {cv['n_pts']:>6} "
                  f"{cv['n_in_box_nd']:>7} {cv['n_dropped_cost_over']:>7} "
                  f"{cv['wait_min']:>10.1f} {cv['cost_max_used']:>14.6g}")
    if a.json:
        json.dump({"box": {"name": box.name, "cost_max": box.cost_max,
                           "wait_max": box.wait_max, "hv_truepf": box.hv_truepf},
                   "rows": rows}, open(a.json, "w"), indent=1)
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
