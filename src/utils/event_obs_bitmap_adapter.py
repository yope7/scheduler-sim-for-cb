"""
イベント観測ベクトルを旧ビットマップ互換のフラット観測へ復元する（ラーナー／学習器側）。

分散PCN・DQN など、NN に入れる直前で get_observation を差し替える用途。
環境変数:
  SCHEDULER_LEARNER_BITMAP … 既定 '1'（ON）。'0' で生イベントベクトルのまま。
  DISTRIBUTED_PCN_EVENT_TO_BITMAP … 未設定時は SCHEDULER_LEARNER_BITMAP にフォールバックしないが、
    両方ある場合は SCHEDULER_LEARNER_BITMAP を優先し、無ければこちらを参照（後方互換）。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from gym import spaces

from src.envs.scheduling_env_event_obs import (
    SchedulingEnvEventObs,
    N_EVENTS_OBS,
    EVENT_FEATURES,
    JOB_QUEUE_SIZE,
)

if TYPE_CHECKING:
    pass


def learner_bitmap_enabled() -> bool:
    if "SCHEDULER_LEARNER_BITMAP" in os.environ:
        return os.environ.get("SCHEDULER_LEARNER_BITMAP", "1") == "1"
    return os.environ.get("DISTRIBUTED_PCN_EVENT_TO_BITMAP", "1") == "1"


def event_obs_to_bitmap_observation(
    obs: np.ndarray,
    n_window: int,
    n_on_premise_node: int,
    n_cloud_node: int,
    obs_window_size: int,
) -> np.ndarray:
    """イベント観測を旧ビットマップ互換の観測へ復元する。"""
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    events_size = N_EVENTS_OBS * EVENT_FEATURES
    if x.size < events_size:
        return x.astype(np.float32, copy=False)

    events = x[:events_size].reshape(N_EVENTS_OBS, EVENT_FEATURES)
    job_queue = x[events_size : events_size + JOB_QUEUE_SIZE]
    if job_queue.size < JOB_QUEUE_SIZE:
        job_queue = np.pad(job_queue, (0, JOB_QUEUE_SIZE - job_queue.size))
    elif job_queue.size > JOB_QUEUE_SIZE:
        job_queue = job_queue[:JOB_QUEUE_SIZE]

    onpre = np.zeros((n_on_premise_node, obs_window_size), dtype=np.float32)
    cloud = np.zeros((n_cloud_node, obs_window_size), dtype=np.float32)

    max_nodes = max(1, n_on_premise_node, n_cloud_node)
    window_start = max(0, n_window - obs_window_size)

    for ev in events:
        start_n, end_n, _, use_cloud_n, start_node_n, job_height_n = ev
        if end_n <= 0.0 and start_n <= 0.0:
            continue

        start_t = int(np.floor(np.clip(start_n, 0.0, 1.0) * n_window))
        end_t = int(np.ceil(np.clip(end_n, 0.0, 1.0) * n_window))
        if end_t <= start_t:
            continue

        use_cloud = bool(use_cloud_n >= 0.5)
        mat = cloud if use_cloud else onpre
        n_nodes = n_cloud_node if use_cloud else n_on_premise_node
        if n_nodes <= 0:
            continue

        start_node = int(np.floor(np.clip(start_node_n, 0.0, 1.0) * max_nodes))
        start_node = int(np.clip(start_node, 0, n_nodes - 1))

        job_height = int(np.ceil(np.clip(job_height_n, 0.0, 1.0) * max_nodes))
        job_height = max(1, min(job_height, n_nodes))
        end_node = min(n_nodes, start_node + job_height)
        if end_node <= start_node:
            continue

        t0 = max(start_t, window_start)
        t1 = min(end_t, n_window)
        if t1 <= t0:
            continue
        col0 = t0 - window_start
        col1 = t1 - window_start
        if col1 <= col0:
            continue
        mat[start_node:end_node, col0:col1] = 1.0

    return np.concatenate([onpre.reshape(-1), cloud.reshape(-1), job_queue]).astype(np.float32, copy=False)


def apply_learner_bitmap_to_event_env(env: SchedulingEnvEventObs) -> SchedulingEnvEventObs:
    """SchedulingEnvEventObs の get_observation をビットマップ復元版へ差し替える（既定ON）。"""
    if not isinstance(env, SchedulingEnvEventObs):
        return env
    if not learner_bitmap_enabled():
        return env
    if getattr(env, "_event_bitmap_adapter_enabled", False):
        return env

    obs_window_size = int(getattr(env, "obs_window_size", 10))
    n_window = int(env.n_window)
    n_onprem = int(env.n_on_premise_node)
    n_cloud = int(env.n_cloud_node)
    original_get_observation = env.get_observation

    def wrapped_get_observation():
        raw = original_get_observation()
        return event_obs_to_bitmap_observation(
            raw,
            n_window=n_window,
            n_on_premise_node=n_onprem,
            n_cloud_node=n_cloud,
            obs_window_size=obs_window_size,
        )

    env._event_obs_get_observation_raw = original_get_observation
    env.get_observation = wrapped_get_observation
    sample = wrapped_get_observation()
    env.observation_space = spaces.Box(low=0, high=1, shape=(sample.shape[0],), dtype=np.float32)
    env._event_bitmap_adapter_enabled = True
    print(f"[ENV] イベント観測をビットマップへ復元してNN入力に使用: dim={sample.shape[0]}")
    return env
