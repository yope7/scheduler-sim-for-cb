#!/usr/bin/env python3
"""配置探索が重くなる「本当の変数」を直接観測する。

env._resource_events = まだ生きている配置イベントの一覧。配置探索のコストはこの長さ R に比例する
(sweep-line は O(R log R + R·H))。よって 1エピソードの総コスト ≈ Σ_step R(step)。
R が n と共に伸びれば超線形、R が飽和すれば線形。

R が伸びる条件:
  (1) ジョブが待たされる(過負荷) -> end が未来に伸びて刈り取れない
  (2) ジョブの実行時間が長い(実traceの重い裾) -> 待ち0 でも end が遠く刈り取れない
  (3) そもそも刈り取りが無い(legacy env, PCN_FAST_ENV=0)

usage:
  PYTHONPATH=. .venv/bin/python scripts/probe_event_growth.py
"""
import os
import json

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")

import numpy as np

from scripts.pcn_replay_snapshot import create_eval_env, load_config

NJ_LIST = [int(x) for x in os.environ.get("NJ_LIST", "128,256,512,1024,2048").split(",")]
P_CLOUD = float(os.environ.get("P", "0.5"))
OUT = os.environ.get("OUT", "results/bench/event_growth.json")
ERA = os.environ.get("ERA", "current")

CFGS = {"synth": "experiments/distributed_pcn/job_synthetic_pcn.yml",
        "trace": "experiments/distributed_pcn/job_trace_5120_pcn.yml"}
_ALL_CASES = [
    # (workload, setting, onprem, cloud, synth_max_nodes)
    ("synth", "over", 256, 1024, 256),
    ("synth", "correct", 1520, 65536, 16),
    ("trace", "over", 256, 1024, None),
    ("trace", "correct", 115520, 924160, None),
]
# legacy env(刈り取り無し・sweep無し)は巨大ノード軸で非現実的に遅いので over だけに絞れるように。
_ONLY = os.environ.get("ONLY_SETTINGS", "over,correct").split(",")
CASES = [c for c in _ALL_CASES if c[1] in _ONLY]


def make_cfg(base, onprem, cloud, max_nodes, nj):
    import copy
    cfg = copy.deepcopy(base)
    cfg["param_env"]["n_on_premise_node"] = onprem
    cfg["param_env"]["n_cloud_node"] = cloud
    pj = cfg.setdefault("param_job", {})
    if max_nodes is not None:
        pj["synth_max_required_nodes"] = max_nodes
    if int(pj.get("job_type", 1)) == 2:
        pj["job_trace_n_jobs"] = nj
    return cfg


def probe(env, nj):
    rng = np.random.default_rng(1000)
    env.reset()
    done = False
    st = 0
    sizes = []
    while not done and st < nj + 16:
        a = 1 if rng.random() < P_CLOUD else 0
        _obs, _r, _s, _wt, done = env.step(a)
        r = len(env._resource_events[False]) + len(env._resource_events[True])
        sizes.append(r)
        st += 1
    sizes = np.asarray(sizes, dtype=float)
    return dict(steps=st, R_mean=float(sizes.mean()), R_max=float(sizes.max()),
                R_final=float(sizes[-1]), R_sum=float(sizes.sum()))


def main():
    rows = []
    for wl, setting, onprem, cloud, mx in CASES:
        base = load_config(CFGS[wl])
        for nj in NJ_LIST:
            env = create_eval_env(make_cfg(base, onprem, cloud, mx, nj), job_seed=0, n_jobs=nj)
            r = probe(env, nj)
            r.update(workload=wl, setting=setting, n_jobs=nj, era=ERA)
            rows.append(r)
            print(f"[{wl:5s}/{setting:7s}] n={nj:5d}  R_mean={r['R_mean']:9.1f}  "
                  f"R_max={r['R_max']:8.0f}  R_final={r['R_final']:8.0f}  "
                  f"ΣR={r['R_sum']:12.0f}  (R_mean/n={r['R_mean']/nj:.3f})", flush=True)

    # R_mean の伸び(冪指数)
    fits = {}
    for wl, setting, *_ in CASES:
        key = f"{wl}/{setting}"
        if key in fits:
            continue
        rs = [r for r in rows if r["workload"] == wl and r["setting"] == setting]
        ns = np.log([r["n_jobs"] for r in rs])
        rm = np.log([max(r["R_mean"], 1e-9) for r in rs])
        b = float(np.polyfit(ns, rm, 1)[0]) if len(rs) >= 2 else float("nan")
        # ΣR = 総探索コストの理論値。その指数も出す。
        sm = np.log([max(r["R_sum"], 1e-9) for r in rs])
        bs = float(np.polyfit(ns, sm, 1)[0]) if len(rs) >= 2 else float("nan")
        fits[key] = dict(exp_R_mean=b, exp_R_sum=bs)
        print(f"FIT {key}: R_mean ~ n^{b:.2f}   ΣR(総探索コスト) ~ n^{bs:.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(rows=rows, fits=fits, era=ERA, p_cloud=P_CLOUD), f, indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
