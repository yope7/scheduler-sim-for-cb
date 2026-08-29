"""raw_rollout_kernel の検証器(設計書 Task7)。

既定(引数なし): N=128ジョブ・B=4(行動列 Bernoulli p=0.3/0.5/0.7/1.0 各1本)で、
event_native_env(CPU, 既定挙動=PCN_FAST_ENV=1のまま)と GPUカーネルの
(start_time, waiting_time, cost) を全ジョブ突き合わせる(Task7-1)。

--big: 実トレース weekB 先頭5万ジョブ(cap 48000/192000)で検証(Task7-2)。
GPU側 B=2(p=0.5/0.9)、CPU側は p=0.5 の1本だけ step で回して全件照合。
E_MAX=16384 で試し ovf なら 65536 に上げて再実行。CPU側が4-5分かかる。

usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
    scripts/proto_gpu_sweep/verify_raw_rollout.py [--big]
注: numba CUDA はドライバ(CUDA 12.2)対応の nvvm が必要。/usr/local/cuda が 12.5 の
このマシンはドライバが CUDA 12.2 / システムが 12.5 で PTX が不整合になるため、12.2 の
nvcc 断片(tools/nvcc122)へ CUDA_HOME を向ける。未構築なら tools/setup_nvcc122.sh を実行。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raw_rollout_kernel import run_raw_rollout  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative  # noqa: E402

N_JOBS = 128
B = 4
N_ON = 64
N_CLOUD = 256
SEED = 12345
P_LIST = [0.3, 0.5, 0.7, 1.0]


def gen_jobs(n_jobs: int, seed: int) -> np.ndarray:
    """raw形式 [submit,pt,nodes,can_cloud,user,id,-1,0] のジョブ列を乱数生成する。

    n_on_premise_node=64 が両資源のうち小さい方なので、height はそれより十分小さく
    抑える(K=16断片上限の余裕と、両資源どちらに送っても有効なジョブにするため)。
    到着間隔を小さめにして混雑(=断片化・雲フォールバック)を誘発する。
    """
    rng = np.random.default_rng(seed)
    submit = 0
    rows = []
    for jid in range(n_jobs):
        submit += int(rng.integers(0, 2))   # 到着間隔0-1(密集させて資源競合を強く作る)
        pt = int(rng.integers(1, 21))       # 処理時間 1-20
        nodes = int(rng.integers(1, 13))    # 要求ノード数 1-12 (<=64, K=16に十分な余裕)
        can_cloud = 1
        user = int(rng.integers(1, 6))
        rows.append([submit, pt, nodes, can_cloud, user, jid, -1, 0])
    return np.asarray(rows, dtype=np.float64)


def gen_actions(n_jobs: int, p_list: list[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for p in p_list:
        rows.append((rng.random(n_jobs) < p).astype(np.int8))
    return np.stack(rows, axis=0)


def run_cpu_episode(jobs_set: dict, actions_row: np.ndarray, n_jobs: int):
    env = SchedulingEnvEventNative(
        np.inf,      # max_step
        16,          # n_window (event-native では配置探索に不使用)
        N_ON,        # n_on_premise_node
        N_CLOUD,     # n_cloud_node
        5,           # n_job_queue_obs
        5,           # n_job_queue_bck
        1.0,         # weight_wt
        1.0,         # weight_cost
        0.0,         # penalty_not_allocate
        0.0,         # penalty_invalid_action
        jobs_set,    # jobs_set
        None,        # job_type
        0,           # flag
    )
    env.reset()
    starts = np.zeros(n_jobs, dtype=np.int64)
    waits = np.zeros(n_jobs, dtype=np.int64)
    costs = np.zeros(n_jobs, dtype=np.int64)
    for j in range(n_jobs):
        action = int(actions_row[j])
        _, rewards, scheduled, waiting_time, done = env.step(action)
        assert scheduled, f"job {j} not scheduled (defer must be off)"
        rec = env._absolute_schedule_records[-1]
        starts[j] = int(rec["start"])
        waits[j] = int(waiting_time)
        costs[j] = int(round(-float(rewards[1])))
    assert done
    return starts, waits, costs


BIG_CFG = "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml"
BIG_N_JOBS = 50000
BIG_P_LIST = [0.5, 0.9]


def run_cpu_episode_env(env, actions_row: np.ndarray, n_jobs: int, progress: int = 0):
    """構築済み env で1エピソード回し (start, wait, cost) を記録する(--big 用)。"""
    env.reset()
    starts = np.zeros(n_jobs, dtype=np.int64)
    waits = np.zeros(n_jobs, dtype=np.int64)
    costs = np.zeros(n_jobs, dtype=np.int64)
    for j in range(n_jobs):
        _, rewards, scheduled, waiting_time, done = env.step(int(actions_row[j]))
        assert scheduled, f"job {j} not scheduled"
        rec = env._absolute_schedule_records[-1]
        starts[j] = int(rec["start"])
        waits[j] = int(waiting_time)
        costs[j] = int(round(-float(rewards[1])))
        if progress and (j + 1) % progress == 0:
            print(f"  [cpu] {j+1}/{n_jobs} steps...", flush=True)
    assert done
    return starts, waits, costs


def main_big() -> int:
    import time

    from scripts.pcn_replay_snapshot import load_config, create_eval_env

    cfg = load_config(BIG_CFG)
    env = create_eval_env(cfg, job_seed=0, n_jobs=BIG_N_JOBS)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    n_on = int(env.n_on_premise_node)
    n_cl = int(env.n_cloud_node)
    n_jobs = jobs.shape[0]
    print(f"[big] trace={BIG_CFG}")
    print(f"[big] n_jobs={n_jobs} n_on={n_on} n_cl={n_cl} p_list={BIG_P_LIST}")

    actions = gen_actions(n_jobs, BIG_P_LIST, SEED + 1)

    # GPU側: E_MAX=16384/K=128 で試し、ovf=1(E_MAX溢れ)なら E_MAX を、
    # ovf=2(断片>K)なら K を上げて再実行(K=16 は trace の断片化で溢れることを実測済み)。
    gpu = None
    e_max, kk = 16384, 128
    for _ in range(4):
        t0 = time.time()
        gpu = run_raw_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=kk)
        dt = time.time() - t0
        print(f"[big gpu] E_MAX={e_max} K={kk}: {dt:.1f}s ovf={gpu['ovf'].tolist()} "
              f"peak_ev_on={gpu['peak_ev_on'].tolist()} peak_ev_cl={gpu['peak_ev_cl'].tolist()}",
              flush=True)
        if not gpu["ovf"].any():
            break
        if (gpu["ovf"] == 1).any():
            e_max = 65536
        if (gpu["ovf"] == 2).any():
            kk *= 4
    if gpu["ovf"].any():
        print(f"[big] FAIL: ovf still set (E_MAX={e_max}, K={kk})")
        return 1
    for b in range(len(BIG_P_LIST)):
        print(f"[big gpu] row={b} p={BIG_P_LIST[b]}: total_cost={gpu['costs'][b].sum()} "
              f"mean_wait={gpu['waits'][b].mean():.3f} max_wait={gpu['waits'][b].max()}")

    # CPU側: p=0.5 の1本だけ全件照合
    print("[big] running CPU reference episode (p=0.5, this takes minutes)...", flush=True)
    t0 = time.time()
    cs, cw, cc = run_cpu_episode_env(env, actions[0], n_jobs, progress=10000)
    print(f"[big cpu] p=0.5: {time.time()-t0:.1f}s total_cost={cc.sum()} "
          f"mean_wait={cw.mean():.3f} max_wait={cw.max()}")

    mism = (cs != gpu["start_times"][0]) | (cw != gpu["waits"][0]) | (cc != gpu["costs"][0])
    n_mism = int(mism.sum())
    if n_mism:
        print(f"[big] FAIL: {n_mism} job mismatches")
        for j in np.flatnonzero(mism)[:10]:
            print(f"  job={j}: cpu=({cs[j]},{cw[j]},{cc[j]}) "
                  f"gpu=({gpu['start_times'][0,j]},{gpu['waits'][0,j]},{gpu['costs'][0,j]})")
        return 1
    print(f"[big] PASS: {n_jobs}/{n_jobs} jobs match on (start, wait, cost) "
          f"[row p=0.5 vs CPU]; row p=0.9 GPU-only (ovf=0)")
    return 0


def main() -> int:
    jobs = gen_jobs(N_JOBS, SEED)
    actions = gen_actions(N_JOBS, P_LIST, SEED + 1)
    jobs_set = {0: jobs.tolist()}

    print(f"[verify] N_JOBS={N_JOBS} B={B} n_on={N_ON} n_cloud={N_CLOUD} p_list={P_LIST}")

    cpu_start = np.zeros((B, N_JOBS), dtype=np.int64)
    cpu_wait = np.zeros((B, N_JOBS), dtype=np.int64)
    cpu_cost = np.zeros((B, N_JOBS), dtype=np.int64)
    for b in range(B):
        s, w, c = run_cpu_episode(jobs_set, actions[b], N_JOBS)
        cpu_start[b], cpu_wait[b], cpu_cost[b] = s, w, c
        print(f"[cpu] row={b} p={P_LIST[b]}: total_cost={c.sum()} mean_wait={w.mean():.3f} max_wait={w.max()}")

    print("[verify] launching GPU kernel...")
    gpu = run_raw_rollout(jobs, actions, N_ON, N_CLOUD)
    print(f"[verify] ovf={gpu['ovf'].tolist()} "
          f"peak_ev_on={gpu['peak_ev_on'].tolist()} peak_ev_cl={gpu['peak_ev_cl'].tolist()}")
    for b in range(B):
        print(
            f"[gpu] row={b} p={P_LIST[b]}: total_cost={gpu['costs'][b].sum()} "
            f"mean_wait={gpu['waits'][b].mean():.3f}"
        )

    assert np.all(gpu["ovf"] == 0), f"ovf flag set: {gpu['ovf'].tolist()}"

    mism_start = cpu_start != gpu["start_times"]
    mism_wait = cpu_wait != gpu["waits"]
    mism_cost = cpu_cost != gpu["costs"]
    n_mismatch = int(mism_start.sum() + mism_wait.sum() + mism_cost.sum())

    if n_mismatch:
        print(f"[verify] FAIL: {n_mismatch} field mismatches")
        for b in range(B):
            idx = np.flatnonzero(mism_start[b] | mism_wait[b] | mism_cost[b])
            for j in idx[:10]:
                print(
                    f"  row={b} job={j}: "
                    f"cpu=(start={cpu_start[b,j]},wait={cpu_wait[b,j]},cost={cpu_cost[b,j]}) "
                    f"gpu=(start={gpu['start_times'][b,j]},wait={gpu['waits'][b,j]},cost={gpu['costs'][b,j]})"
                )
        return 1

    total_checks = B * N_JOBS * 3
    print(f"[verify] PASS: {total_checks}/{total_checks} fields match (B={B} x N_JOBS={N_JOBS} x 3 fields)")
    return 0


if __name__ == "__main__":
    if "--big" in sys.argv[1:]:
        raise SystemExit(main_big())
    raise SystemExit(main())
