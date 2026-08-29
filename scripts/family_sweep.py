#!/usr/bin/env python3
"""多族掃引: 複数の「並べ方(族)」でクラウド送信対象を選び、送信割合を掃引して参照フロントを作る。

pダイヤル(ランダムに率pで送る)は「どのジョブを送るか」を選ばないので、同じコストでも
待ちが減らせる余地を捨てている。族=送る順番の決め方を変えると、同一コストで待ちが
大きく変わる。その非支配包絡を参照線に使う。

族(order):
  big    : 占有量 pt*nodes が大きい順に送る
  small  : 占有量が小さい順に送る
  long   : 実行時間 pt が長い順
  short  : 実行時間 pt が短い順
  wide   : 要求ノード(コア)数が多い順 (tie は pt 長い順)
  fcfs   : 到着が早い順(前半をクラウドへ)
  lifo   : 到着が遅い順(後半をクラウドへ)
  burst  : 到着が混んでいる時間帯のジョブ順(窓 BURST_W 秒の到着密度が高い順)
  random : ランダム順(件数固定版のpダイヤル)
比較用:
  pdial  : 各ジョブ独立に確率 p でクラウド(従来のpダイヤル)

出力 npz (hv_boxed.py 互換: `pf` に非支配包絡):
  pf          : 多族掃引の非支配包絡 (参照線)
  family_all  : 多族掃引の全点
  pdial       : pダイヤル掃引の全点
  fam_<name>  : 族ごとの点
usage:
  OMP_NUM_THREADS=1 CFG=... NJ=20000 NPROC=32 OUT=results/eval_pf/famsweep.npz \
  PYTHONPATH=. .venv/bin/python scripts/family_sweep.py
"""
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "0")
os.environ.setdefault("SCHEDULER_OBS_OCCUPANCY", "1")

import numpy as np
import yaml

from scripts.pcn_replay_snapshot import create_eval_env

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_weekB_head20000_cap48000_pcn.yml")
NJ = int(os.environ.get("NJ", "20000"))
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
NPROC = int(os.environ.get("NPROC", "32"))
OUT = os.environ.get("OUT", "results/eval_pf/famsweep_weekB20000_cap48000.npz")
# [2026-08-30 R4] 既存の参照線npz(08-17確定の20k参照線など)を黙って上書きしない。
if os.path.exists(OUT):
    raise SystemExit(f"[ABORT] {OUT} は既に存在します。別のOUTを明示するか、消すなら手で。")
BURST_W = float(os.environ.get("BURST_W", "3600"))
RAND_SEED = int(os.environ.get("RAND_SEED", "20260817"))

FAMILIES = os.environ.get(
    "FAMILIES", "big,small,long,short,wide,fcfs,lifo,burst,random"
).split(",")
# 送信割合の格子(端 0.0/1.0 は族に依らず同一なので1回だけ評価する)
FRACS = [float(x) for x in os.environ.get(
    "FRACS", "0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.65,0.80,0.90"
).split(",")]
# pダイヤル(従来ベースライン)の掃引点
PDIAL = [float(x) for x in os.environ.get(
    "PDIAL", ",".join(f"{v:.4f}" for v in np.linspace(0.0, 1.0, 40))
).split(",")]

_ENV = None
_JOBS = None


def _worker_init():
    global _ENV, _JOBS
    cfg = yaml.safe_load(open(CFG))
    cfg["param_job"]["job_trace_n_jobs"] = NJ
    cfg["param_env"]["n_jobs"] = NJ
    _ENV = create_eval_env(cfg, job_seed=JOB_SEED, n_jobs=NJ)
    _ENV.reset()
    _JOBS = np.asarray(_ENV.jobs, dtype=np.float64)


def build_order(fam: str, jobs: np.ndarray) -> np.ndarray:
    """族名 -> 「先にクラウドへ送る」順のインデックス列。"""
    st, pt, nd = jobs[:, 0], jobs[:, 1], jobs[:, 2]
    occ = pt * nd
    if fam == "big":
        key = -occ
    elif fam == "small":
        key = occ
    elif fam == "long":
        key = -pt
    elif fam == "short":
        key = pt
    elif fam == "wide":
        # nodes 降順 / 同 nodes 内は pt 降順(lexsort は最後のキーが主キー)
        return np.lexsort((-pt, -nd))
    elif fam == "fcfs":
        key = st
    elif fam == "lifo":
        key = -st
    elif fam == "burst":
        # 到着密度: 各ジョブの submit を中心とする幅 BURST_W の窓に入る到着件数(降順)
        s = np.sort(st)
        lo = np.searchsorted(s, st - BURST_W / 2.0, side="left")
        hi = np.searchsorted(s, st + BURST_W / 2.0, side="right")
        key = -(hi - lo).astype(np.float64)
    elif fam == "random":
        return np.random.default_rng(RAND_SEED).permutation(len(jobs))
    else:
        raise ValueError(f"unknown family: {fam}")
    return np.argsort(key, kind="stable")


def _run(action: np.ndarray) -> tuple:
    env = _ENV
    env.reset()
    done = False
    st = 0
    n = len(action)
    while not done and st < n + 16:
        a = int(action[st]) if st < n else 0
        _obs, _r, _sched, _wt, done = env.step(a)
        st += 1
    cost, mk, avgwt = env.calc_objective_values()
    return float(cost), float(avgwt), float(mk), st


def _task(arg) -> dict:
    kind, name, val = arg
    t0 = time.perf_counter()
    n = len(_JOBS)
    if kind == "fam":
        order = build_order(name, _JOBS)
        k = int(round(val * n))
        action = np.zeros(n, dtype=np.int8)
        action[order[:k]] = 1
    elif kind == "pdial":
        rng = np.random.default_rng(RAND_SEED + 1)
        action = (rng.random(n) < val).astype(np.int8)
    else:  # end points
        action = np.full(n, int(val), dtype=np.int8)
    cost, wait, mk, steps = _run(action)
    dt = time.perf_counter() - t0
    return dict(kind=kind, family=name, val=float(val), cost=cost, wait=wait,
                makespan=mk, steps=steps, sec=dt, n_cloud=int(action.sum()))


def non_dominated(pts: np.ndarray) -> np.ndarray:
    """min-min の非支配点(cost昇順→wait単調減の階段)。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return p
    idx = np.lexsort((p[:, 1], p[:, 0]))
    p = p[idx]
    keep, best = [], np.inf
    for i in range(len(p)):
        if p[i, 1] < best:
            keep.append(i)
            best = p[i, 1]
    return p[keep]


def main():
    tasks = [("end", "endpoint", 0.0), ("end", "endpoint", 1.0)]
    for fam in FAMILIES:
        for fr in FRACS:
            tasks.append(("fam", fam, fr))
    for p in PDIAL:
        if p <= 0.0 or p >= 1.0:
            continue  # 端は endpoint と同一
        tasks.append(("pdial", "pdial", p))

    print(f"多族掃引: CFG={CFG} NJ={NJ} 族={len(FAMILIES)} 点数={len(tasks)} NPROC={NPROC}",
          flush=True)
    t0 = time.perf_counter()
    with mp.Pool(NPROC, initializer=_worker_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)} 完了 ({time.perf_counter()-t0:.0f}s)", flush=True)
    wall = time.perf_counter() - t0
    print(f"掃引完了: {len(rows)}点 / {wall:.1f}秒 "
          f"(逐次換算 {sum(r['sec'] for r in rows):.0f}秒, 実効並列 "
          f"{sum(r['sec'] for r in rows)/max(wall,1e-9):.1f}x)", flush=True)

    ends = np.array([[r["cost"], r["wait"]] for r in rows if r["kind"] == "end"])
    fam_pts = {}
    for fam in FAMILIES:
        pts = [[r["cost"], r["wait"]] for r in rows if r["kind"] == "fam" and r["family"] == fam]
        fam_pts[fam] = np.vstack([np.array(pts), ends]) if pts else ends
    family_all = np.vstack([np.array([[r["cost"], r["wait"]] for r in rows
                                      if r["kind"] == "fam"]), ends])
    pdial_pts = np.array([[r["cost"], r["wait"]] for r in rows if r["kind"] == "pdial"])
    pdial_all = np.vstack([pdial_pts, ends]) if len(pdial_pts) else ends

    pf = non_dominated(family_all)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    save = dict(pf=pf, family_all=family_all, pdial=pdial_all,
                pdial_pf=non_dominated(pdial_all))
    for fam, pts in fam_pts.items():
        save[f"fam_{fam}"] = pts
    np.savez(OUT, **save)
    with open(OUT.replace(".npz", "_rows.json"), "w") as f:
        json.dump(dict(cfg=CFG, nj=NJ, wall_sec=wall, rows=rows), f, indent=1)
    print(f"saved {OUT}: 参照線{len(pf)}点 / 族全点{len(family_all)} / pダイヤル{len(pdial_all)}")

    # 参照線 vs pダイヤル の等コスト待ち比
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hv_boxed import Box, eq_cost_ratio, non_dominated_min
    both = np.vstack([family_all, pdial_all])
    nd = non_dominated_min(both)
    box = Box(cost_max=float(nd[:, 0].max()), wait_max=float(nd[:, 1].max()),
              truepf=nd, name="famsweep-box")
    r = eq_cost_ratio(pf, non_dominated(pdial_all), box)
    print(f"[判定] 等コスト待ち比 参照線/pダイヤル = {r:.4f} (1未満=参照線が待たない)")
    for fam in FAMILIES:
        rf = eq_cost_ratio(non_dominated(fam_pts[fam]), non_dominated(pdial_all), box)
        print(f"    族 {fam:<7} vs pダイヤル = {rf:.4f}")


if __name__ == "__main__":
    main()
