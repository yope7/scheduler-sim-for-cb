"""決定論リプレイ(replay_obs_builder)の検証器。

128ジョブ・B=4(verify_raw_rollout.py と同じ生成条件・同じ行動列)で、
  (a) 通常経路: env.step を回して各ステップの (obs, reward, start) を記録
  (b) リプレイ経路: raw_rollout_kernel で start_times を求め、replay_obs_builder でリプレイ
の全ステップの obs / reward / start が完全一致(np.array_equal)することを確認する。

観測環境変数は学習と同じ構成に揃える(スクリプト冒頭で setdefault。既に設定済みなら
上書きしない)。SchedulingEnvEventObs.__init__ が os.environ を読んで観測次元/内容を
決めるため、env を1つでも構築する前にこの設定が済んでいる必要がある。

usage: CUDA_HOME=$PWD/tools/nvcc122/nvidia/cuda_nvcc CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
    scripts/proto_gpu_sweep/verify_replay_obs.py
"""
from __future__ import annotations

import os

# 観測環境変数: 学習と同じ構成(スクリプト冒頭で setdefault。既存の指定があれば尊重する)。
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
os.environ.setdefault("SCHEDULER_OBS_EFFICIENCY", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raw_rollout_kernel import run_raw_rollout  # noqa: E402
from replay_obs_builder import build_env, build_replay_dataset  # noqa: E402
from verify_raw_rollout import (  # noqa: E402
    B, N_CLOUD, N_JOBS, N_ON, P_LIST, SEED, gen_actions, gen_jobs,
)


def run_cpu_normal(jobs_set: dict, actions_row: np.ndarray, n_jobs: int):
    """通常経路(step)を回し、各ステップの (obs, reward) と各ジョブの start を記録する。"""
    env = build_env(jobs_set[0], N_ON, N_CLOUD)
    obs = env.reset()
    obs_dim = obs.shape[0]
    obs_arr = np.zeros((n_jobs, obs_dim), dtype=np.float32)
    rewards_arr = np.zeros((n_jobs, 2), dtype=np.float32)
    starts = np.zeros(n_jobs, dtype=np.int64)
    for t in range(n_jobs):
        obs_arr[t] = obs
        action = int(actions_row[t])
        n_obs, reward, scheduled, waiting_time, done = env.step(action)
        assert scheduled, f"job {t} not scheduled (defer must be off)"
        rec = env._absolute_schedule_records[-1]
        starts[t] = int(rec["start"])
        rewards_arr[t] = reward
        obs = n_obs
    assert done
    return obs_arr, rewards_arr, starts


def main() -> int:
    jobs = gen_jobs(N_JOBS, SEED)
    actions = gen_actions(N_JOBS, P_LIST, SEED + 1)
    jobs_set = {0: jobs.tolist()}

    print(f"[verify_replay] N_JOBS={N_JOBS} B={B} n_on={N_ON} n_cloud={N_CLOUD} p_list={P_LIST}")
    print(
        "[verify_replay] obs flags: "
        f"SCHEDULER_OBS_URGENCY={os.environ.get('SCHEDULER_OBS_URGENCY')} "
        f"SCHEDULER_OBS_EFFICIENCY={os.environ.get('SCHEDULER_OBS_EFFICIENCY')} "
        f"DISTRIBUTED_PCN_USE_EVENT_OBS={os.environ.get('DISTRIBUTED_PCN_USE_EVENT_OBS')} "
        f"SCHEDULER_LEARNER_BITMAP={os.environ.get('SCHEDULER_LEARNER_BITMAP')}"
    )

    # (a) 通常経路
    cpu_obs_list = []
    cpu_rew_list = []
    cpu_start = np.zeros((B, N_JOBS), dtype=np.int64)
    t0 = time.time()
    for b in range(B):
        o, r, s = run_cpu_normal(jobs_set, actions[b], N_JOBS)
        cpu_obs_list.append(o)
        cpu_rew_list.append(r)
        cpu_start[b] = s
    dt_normal = time.time() - t0
    print(f"[verify_replay] normal path: {dt_normal:.2f}s for B={B} episodes ({dt_normal/B:.3f}s/episode)")

    # (b) GPUカーネルで start_times を求める
    print("[verify_replay] launching GPU kernel...")
    t0 = time.time()
    gpu = run_raw_rollout(jobs, actions, N_ON, N_CLOUD)
    dt_gpu = time.time() - t0
    print(f"[verify_replay] gpu kernel: {dt_gpu:.3f}s ovf={gpu['ovf'].tolist()}")
    assert np.all(gpu["ovf"] == 0), f"ovf flag set: {gpu['ovf'].tolist()}"

    mism_start_gpu = cpu_start != gpu["start_times"]
    if mism_start_gpu.any():
        print(
            f"[verify_replay] WARN: start_times already mismatch vs raw GPU kernel "
            f"before replay ({int(mism_start_gpu.sum())} cells)"
        )

    # (c) リプレイ経路
    t0 = time.time()
    replay = build_replay_dataset(
        jobs, actions, gpu["start_times"], N_ON, N_CLOUD, nproc=min(B, os.cpu_count() or 1)
    )
    dt_replay = time.time() - t0
    print(
        f"[verify_replay] replay: {dt_replay:.3f}s for B={B} episodes "
        f"({dt_replay/B:.3f}s/episode, {dt_replay/(B*N_JOBS)*1000:.3f}ms/job)"
    )

    # (d) 全ステップ一致判定
    n_obs_checked = 0
    n_rew_checked = 0
    n_start_checked = 0
    fail = False
    for b in range(B):
        ep = replay[b]
        ok_obs = np.array_equal(cpu_obs_list[b], ep["obs"])
        ok_rew = np.array_equal(cpu_rew_list[b], ep["rewards"])
        ok_start = np.array_equal(cpu_start[b], ep["start_times"])
        n_obs_checked += cpu_obs_list[b].size
        n_rew_checked += cpu_rew_list[b].size
        n_start_checked += cpu_start[b].size
        status = "OK" if (ok_obs and ok_rew and ok_start) else "MISMATCH"
        print(
            f"[verify_replay] row={b} p={P_LIST[b]}: "
            f"obs={'OK' if ok_obs else 'FAIL'} reward={'OK' if ok_rew else 'FAIL'} "
            f"start={'OK' if ok_start else 'FAIL'} -> {status}"
        )
        if not (ok_obs and ok_rew and ok_start):
            fail = True
            if not ok_obs:
                diff = np.flatnonzero(np.any(cpu_obs_list[b] != ep["obs"], axis=1))
                print(f"  obs mismatch at steps: {diff[:10].tolist()} (total {len(diff)})")
            if not ok_rew:
                diff = np.flatnonzero(np.any(cpu_rew_list[b] != ep["rewards"], axis=1))
                print(f"  reward mismatch at steps: {diff[:10].tolist()} (total {len(diff)})")
            if not ok_start:
                diff = np.flatnonzero(cpu_start[b] != ep["start_times"])
                print(f"  start mismatch at steps: {diff[:10].tolist()} (total {len(diff)})")

    if fail:
        print("[verify_replay] FAIL")
        return 1
    print(
        f"[verify_replay] PASS: obs {n_obs_checked} cells, reward {n_rew_checked} cells, "
        f"start {n_start_checked} cells all match (B={B} x N_JOBS={N_JOBS})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
