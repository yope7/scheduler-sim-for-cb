"""raw_rollout_kernel の出力を「観測つきエピソード」に変換する決定論リプレイビルダー。

GPU完結カーネル(raw_rollout_kernel.run_raw_rollout)は各エピソードの
(actions[T], start_times[T], waits[T], costs[T]) を返すが観測(obs)は返さない。
ここでは event_native_env.SchedulingEnvEventNative.step_with_start_hint を使い、
「開始時刻が既知の1点についてノード選択を1回だけやる」ことで配置探索(候補時刻
sweep)を丸ごと省きつつ、通常の step() と完全に同一の (観測, 報酬, イベント表) を
再現する(scripts/proto_gpu_sweep/verify_replay_obs.py で全ステップ一致を検証済み)。

観測に何を含めるか(SCHEDULER_OBS_URGENCY 等)は env の __init__ が os.environ を
読んで決めるため、呼び出し側が事前に環境変数で指定する(このモジュールは一切設定
しない)。multiprocessing worker は fork で起動するため、親プロセスで設定済みの
環境変数はそのまま子プロセスへ引き継がれる。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative  # noqa: E402


def build_env(
    jobs: np.ndarray,
    n_on: int,
    n_cl: int,
    *,
    n_window: int = 16,
    n_job_queue_obs: int = 5,
    n_job_queue_bck: int = 5,
) -> SchedulingEnvEventNative:
    """1エピソード分の event-native env を構築する(verify_raw_rollout.py と同一シグネチャ)。

    n_window は event-native では配置探索に使わないが、観測の時間正規化
    (SchedulingEnvEventObs._norm_time = max(1, n_window)) に効くため、学習と同じ観測を
    再現するには学習 env と同じ値を渡すこと(gpu_factory 統合経路は config の n_window を渡す)。
    """
    jobs_set = {0: np.asarray(jobs, dtype=np.float64).tolist()}
    return SchedulingEnvEventNative(
        np.inf,            # max_step
        n_window,          # n_window
        n_on,              # n_on_premise_node
        n_cl,              # n_cloud_node
        n_job_queue_obs,   # n_job_queue_obs
        n_job_queue_bck,   # n_job_queue_bck
        1.0,               # weight_wt
        1.0,               # weight_cost
        0.0,               # penalty_not_allocate
        0.0,               # penalty_invalid_action
        jobs_set,          # jobs_set
        None,              # job_type
        0,                 # flag
    )


def replay_episode(
    jobs: np.ndarray,
    n_on: int,
    n_cl: int,
    actions_row: np.ndarray,
    start_times_row: np.ndarray,
    env_kwargs: Optional[dict] = None,
) -> dict:
    """1エピソード分を step_with_start_hint でリプレイし、教材dictを返す。

    Returns:
        dict(obs: (T,obs_dim) float32, actions: (T,) int8,
             rewards: (T,2) float32, start_times: (T,) int64,
             objective_values: [cost, makespan, avg_wait])
    objective_values は従来 actor のエピソード終端処理
    (env.finalize_window_history() → env.calc_objective_values()) と同一手順で計算する。
    """
    env = build_env(jobs, n_on, n_cl, **(env_kwargs or {}))
    t_jobs = int(np.asarray(jobs).shape[0])
    obs = env.reset()
    obs_dim = int(obs.shape[0])
    obs_arr = np.zeros((t_jobs, obs_dim), dtype=np.float32)
    rewards_arr = np.zeros((t_jobs, 2), dtype=np.float32)
    for t in range(t_jobs):
        obs_arr[t] = obs
        action = int(actions_row[t])
        hint = int(start_times_row[t])
        n_obs, reward, scheduled, waiting_time, done = env.step_with_start_hint(action, hint)
        if not scheduled:
            raise RuntimeError(f"job {t} not scheduled during replay (defer must be off)")
        rewards_arr[t] = reward
        obs = n_obs
    env.finalize_window_history()
    cost, makespan, avg_wait = env.calc_objective_values()
    return dict(
        obs=obs_arr,
        actions=np.asarray(actions_row[:t_jobs], dtype=np.int8),
        rewards=rewards_arr,
        start_times=np.asarray(start_times_row[:t_jobs], dtype=np.int64),
        objective_values=[cost, makespan, avg_wait],
    )


# --- multiprocessing worker -------------------------------------------------
# fork した子プロセスで参照するグローバル(jobs は Copy-on-Write で共有され、
# 子プロセスに渡す際の pickle コピーを避けられる)。
_WORKER_JOBS: Optional[np.ndarray] = None
_WORKER_N_ON: int = 0
_WORKER_N_CL: int = 0
_WORKER_ENV_KWARGS: Optional[dict] = None


def _worker_init(jobs: np.ndarray, n_on: int, n_cl: int, env_kwargs: Optional[dict]) -> None:
    global _WORKER_JOBS, _WORKER_N_ON, _WORKER_N_CL, _WORKER_ENV_KWARGS
    _WORKER_JOBS = jobs
    _WORKER_N_ON = n_on
    _WORKER_N_CL = n_cl
    _WORKER_ENV_KWARGS = env_kwargs


def _worker_replay(args):
    b, actions_row, start_times_row = args
    result = replay_episode(
        _WORKER_JOBS, _WORKER_N_ON, _WORKER_N_CL, actions_row, start_times_row,
        env_kwargs=_WORKER_ENV_KWARGS,
    )
    return b, result


def build_replay_dataset(
    jobs: np.ndarray,
    actions: np.ndarray,
    start_times: np.ndarray,
    n_on: int,
    n_cl: int,
    nproc: int = 1,
    env_kwargs: Optional[dict] = None,
) -> list[dict]:
    """B本のエピソードをリプレイし、エピソードごとの教材dictのリストを返す(入力順を保持)。

    Args:
        jobs: (T,8) 配列(raw形式、run_raw_rollout と同一)。
        actions: (B,T) 配列。
        start_times: (B,T) 配列(raw_rollout_kernel.run_raw_rollout の出力)。
        n_on, n_cl: オンプレ/クラウドのノード数。
        nproc: 並列ワーカー数。1以下なら逐次実行(fork オーバーヘッドを避ける)。
        env_kwargs: build_env への追加引数(n_window 等。学習 env と観測正規化を揃える)。
    """
    jobs = np.ascontiguousarray(jobs, dtype=np.float64)
    actions = np.ascontiguousarray(actions)
    start_times = np.ascontiguousarray(start_times)
    b_count = int(actions.shape[0])
    if start_times.shape[0] != b_count:
        raise ValueError(f"actions.shape[0]={b_count} != start_times.shape[0]={start_times.shape[0]}")

    if nproc <= 1 or b_count <= 1:
        return [
            replay_episode(jobs, n_on, n_cl, actions[b], start_times[b], env_kwargs=env_kwargs)
            for b in range(b_count)
        ]

    ctx = mp.get_context("fork")
    tasks = [(b, actions[b], start_times[b]) for b in range(b_count)]
    results: list[Optional[dict]] = [None] * b_count
    with ctx.Pool(
        processes=min(nproc, b_count),
        initializer=_worker_init,
        initargs=(jobs, n_on, n_cl, env_kwargs),
    ) as pool:
        for b, res in pool.imap_unordered(_worker_replay, tasks):
            results[b] = res
    return results
