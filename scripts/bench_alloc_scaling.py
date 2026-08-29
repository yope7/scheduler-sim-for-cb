#!/usr/bin/env python3
"""env(配置探索)の計算コストがジョブ数 n でどう伸びるかを実測し、冪指数を推定する。

「n^2.6 の壁」の根拠データ、および「それは容量の単位ミスマッチ(過負荷)のアーティファクトで
正しい容量ならほぼ線形」の反証データを、同一スクリプト・同一環境で同時に取る。

系列:
  over    : 旧設定(過負荷)   オンプレ256 / クラウド1024 / 合成ジョブ最大256ノード要求
  correct : 正容量           オンプレ1520 / クラウド65536 / 合成ジョブ最大16ノード要求
方策:
  P=0.5 (混合) と P=0.0 (全オンプレ=最悪混雑)

出力: results/bench/alloc_scaling.json (+ .csv), docs/figures/alloc_scaling.png

usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/bench_alloc_scaling.py
env:
  NJ_LIST=64,128,256,512,1024,2048  MAX_SEC=90  OUT=...
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")

import numpy as np

from scripts.pcn_replay_snapshot import create_eval_env, load_config

NJ_LIST = [int(x) for x in os.environ.get("NJ_LIST", "64,128,256,512,1024,2048").split(",")]
MAX_SEC = float(os.environ.get("MAX_SEC", "90"))  # 1エピソードがこれを超えたらその系列は打ち切り
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
WORKLOAD = os.environ.get("WORKLOAD", "synth")  # synth | trace
_DEF_CFG = {"synth": "experiments/distributed_pcn/job_synthetic_pcn.yml",
            "trace": "experiments/distributed_pcn/job_trace_5120_pcn.yml"}
CFG = os.environ.get("CFG", _DEF_CFG[WORKLOAD])
OUT_JSON = os.environ.get("OUT", f"results/bench/alloc_scaling_{WORKLOAD}.json")
OUT_PNG = os.environ.get("OUT_PNG", f"docs/figures/alloc_scaling_{WORKLOAD}.png")

# 「env 世代」はモジュール import 時に確定するため、プロセスの環境変数で切り替える。
ERA = os.environ.get("ERA", "current")  # ラベルのみ(実体は PCN_FAST_ENV / PCN_SWEEP_C)

# name: (n_on_premise_node, n_cloud_node, synth_max_required_nodes)
#  over    = これまでの全実験の設定(容量256)。required_nodes を「ノード数」と誤読した過負荷設定。
#  correct = 正しい容量。synth はノード粒度(1520ノード/小ジョブ)、trace は required_nodes が
#            実質コア数なのでコア粒度(115,520コア = 1520ノード×76コア)で受ける。
_SETTINGS = {
    "synth": {"over": (256, 1024, 256), "correct": (1520, 65536, 16)},
    "trace": {"over": (256, 1024, None), "correct": (115520, 924160, None)},
}
SETTINGS = {k: v for k, v in _SETTINGS[WORKLOAD].items()
            if k in os.environ.get("SETTINGS", "over,correct").split(",")}
POLICIES = {k: v for k, v in {"mix": 0.5, "onprem": 0.0}.items()
            if k in os.environ.get("POLICIES", "mix,onprem").split(",")}


def make_cfg(base: dict, onprem: int, cloud: int, max_nodes, nj: int) -> dict:
    import copy

    cfg = copy.deepcopy(base)
    cfg["param_env"]["n_on_premise_node"] = onprem
    cfg["param_env"]["n_cloud_node"] = cloud
    pj = cfg.setdefault("param_job", {})
    if max_nodes is not None:
        pj["synth_max_required_nodes"] = max_nodes
    if int(pj.get("job_type", 1)) == 2:
        pj["job_trace_n_jobs"] = nj  # trace は CSV 先頭 nj 行を使う
    return cfg


def run_episode(env, nj: int, p_cloud: float, action_seed: int):
    rng = np.random.default_rng(action_seed)
    env.reset()
    done = False
    st = 0
    while not done and st < nj + 16:
        a = 1 if rng.random() < p_cloud else 0
        _obs, _r, _sched, _wt, done = env.step(a)
        st += 1
    cost, _mk, avgwt = env.calc_objective_values()
    return st, float(cost), float(avgwt)


def measure(cfg_base, setting: str, policy: str) -> list:
    onprem, cloud, max_nodes = SETTINGS[setting]
    p_cloud = POLICIES[policy]
    rows = []
    for nj in NJ_LIST:
        cfg = make_cfg(cfg_base, onprem, cloud, max_nodes, nj)
        t_build = time.perf_counter()
        env = create_eval_env(cfg, job_seed=JOB_SEED, n_jobs=nj)
        t_build = time.perf_counter() - t_build

        t0 = time.perf_counter()
        steps, cost, avgwt = run_episode(env, nj, p_cloud, 1)  # warm(計測外)
        warm = time.perf_counter() - t0

        reps = 4 if warm < 1.0 else (2 if warm < 5.0 else 1)
        t0 = time.perf_counter()
        for e in range(reps):
            steps, cost, avgwt = run_episode(env, nj, p_cloud, 1000 + e)
        dt = (time.perf_counter() - t0) / reps

        rows.append(
            dict(setting=setting, policy=policy, n_jobs=nj, sec_per_episode=dt,
                 steps=steps, cost=cost, avg_wait=avgwt, reps=reps,
                 env_build_sec=t_build, onprem=onprem, cloud=cloud, max_nodes=max_nodes,
                 era=ERA, workload=WORKLOAD)
        )
        print(f"[{setting:7s}/{policy:6s}] n={nj:6d}  {dt:9.4f} s/ep  "
              f"({dt/nj*1e6:8.1f} us/step)  avg_wait={avgwt:10.2f}  reps={reps}", flush=True)
        if dt > MAX_SEC:
            print(f"  -> {dt:.1f}s > MAX_SEC={MAX_SEC}: この系列はここで打ち切り", flush=True)
            break
    return rows


def fit_exponent(ns, ts, n_min: int = 0):
    """log-log 最小二乗で t ≈ a * n^b の b を推定。R^2 も返す。"""
    ns = np.asarray(ns, dtype=float)
    ts = np.asarray(ts, dtype=float)
    m = ns >= n_min
    if m.sum() < 2:
        return None, None
    x = np.log(ns[m])
    y = np.log(ts[m])
    b, loga = np.polyfit(x, y, 1)
    yhat = b * x + loga
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), float(r2)


def main():
    cfg_base = load_config(CFG)
    all_rows = []
    for setting in SETTINGS:
        for policy in POLICIES:
            all_rows += measure(cfg_base, setting, policy)

    fits = {}
    for setting in SETTINGS:
        for policy in POLICIES:
            rs = [r for r in all_rows if r["setting"] == setting and r["policy"] == policy]
            ns = [r["n_jobs"] for r in rs]
            ts = [r["sec_per_episode"] for r in rs]
            b_all, r2_all = fit_exponent(ns, ts)
            b_tail, r2_tail = fit_exponent(ns, ts, n_min=256)
            fits[f"{setting}/{policy}"] = dict(exp_all=b_all, r2_all=r2_all,
                                               exp_n_ge_256=b_tail, r2_n_ge_256=r2_tail,
                                               n_points=len(rs))
            print(f"FIT {setting}/{policy}: b(all)={b_all}  b(n>=256)={b_tail} (R2={r2_tail})")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = dict(rows=all_rows, fits=fits, settings=SETTINGS, policies=POLICIES,
                   cfg=CFG, job_seed=JOB_SEED, nj_list=NJ_LIST,
                   workload=WORKLOAD, era=ERA,
                   env=dict(fast_env=os.environ.get("PCN_FAST_ENV", "1"),
                            fast_env_sweep=os.environ.get("PCN_FAST_ENV_SWEEP", "1"),
                            sweep_c=os.environ.get("PCN_SWEEP_C", "1"),
                            omp=os.environ.get("OMP_NUM_THREADS")))
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    csv = OUT_JSON.replace(".json", ".csv")
    with open(csv, "w") as f:
        keys = ["setting", "policy", "n_jobs", "sec_per_episode", "avg_wait", "cost", "steps", "reps"]
        f.write(",".join(keys) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"saved {OUT_JSON} / {csv}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    except Exception:
        pass
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    style = {("over", "onprem"): ("#d94a4a", "-o"), ("over", "mix"): ("#e88", "--o"),
             ("correct", "onprem"): ("#2563eb", "-s"), ("correct", "mix"): ("#7ab", "--s")}
    _cap = {"over": "過負荷(容量256)",
            "correct": "正容量(1520ノード)" if WORKLOAD == "synth" else "正容量(115,520コア)"}
    _pol = {"onprem": "全オンプレ", "mix": "混合"}
    label = {(s, p): f"{_cap[s]} {_pol[p]}" for s in ("over", "correct") for p in ("onprem", "mix")}
    ax = axes[0]
    for key, (c, ls) in style.items():
        rs = [r for r in all_rows if (r["setting"], r["policy"]) == key]
        if not rs:
            continue
        ns = np.array([r["n_jobs"] for r in rs], dtype=float)
        ts = np.array([r["sec_per_episode"] for r in rs], dtype=float)
        b = fits[f"{key[0]}/{key[1]}"]["exp_n_ge_256"] or fits[f"{key[0]}/{key[1]}"]["exp_all"]
        ax.loglog(ns, ts, ls, color=c, ms=5, lw=1.6, label=f"{label[key]}  n^{b:.2f}")
    # 参照傾き
    nref = np.array([NJ_LIST[0], NJ_LIST[-1]], dtype=float)
    for expo, lab, col in ((1.0, "n^1 (線形)", "#888"), (2.6, "n^2.6", "#c33")):
        rs = [r for r in all_rows if r["setting"] == "over" and r["policy"] == "onprem"]
        if rs:
            anchor_t = rs[0]["sec_per_episode"]
            anchor_n = rs[0]["n_jobs"]
            ax.loglog(nref, anchor_t * (nref / anchor_n) ** expo, ":", color=col, lw=1.2, label=lab)
    ax.set_xlabel("ジョブ数 n"); ax.set_ylabel("1エピソードの env 実行時間 [秒]")
    ax.set_title("配置探索コストのスケーリング(log-log)")
    ax.grid(alpha=0.3, which="both", lw=0.5); ax.legend(fontsize=8)

    ax = axes[1]
    for key, (c, ls) in style.items():
        rs = [r for r in all_rows if (r["setting"], r["policy"]) == key]
        if not rs:
            continue
        ns = np.array([r["n_jobs"] for r in rs], dtype=float)
        ws = np.array([r["avg_wait"] for r in rs], dtype=float)
        ax.loglog(ns, np.maximum(ws, 1e-3), ls, color=c, ms=5, lw=1.6, label=label[key])
    ax.set_xlabel("ジョブ数 n"); ax.set_ylabel("平均待ち時間 (混雑の指標)")
    ax.set_title("なぜ伸びるか: 過負荷は待ち行列が n と共に膨張")
    ax.grid(alpha=0.3, which="both", lw=0.5); ax.legend(fontsize=8)

    _wl = {"synth": "合成ジョブ", "trace": "実trace"}[WORKLOAD]
    fig.suptitle(f"env 配置探索の計算量 [{_wl} / env={ERA}]: "
                 "「n^2.6 の壁」は容量の単位ミスマッチ(過負荷)のアーティファクト")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    sys.exit(main())
