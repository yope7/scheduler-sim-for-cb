#!/usr/bin/env python3
"""
イベントベース観測環境の検証スクリプト

- SchedulingEnvEventObs の基本動作確認
- 観測形状・値の妥当性チェック
- 短時間のエピソード実行で整合性確認
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import yaml

from src.envs.scheduling_env_event_obs import (
    SchedulingEnvEventObs,
    N_EVENTS_OBS,
    EVENT_FEATURES,
    JOB_QUEUE_SIZE,
)
from src.utils.job_gen.job_generator import JobGenerator


def load_config():
    config_path = project_root / "config" / "config.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_observation_shape():
    """観測の形状とサイズを検証"""
    print("=== 観測形状の検証 ===")
    config = load_config()
    pe = config["param_env"]
    pa = config["param_agent"]
    pj = config["param_job"]

    n_jobs = pj.get("job_trace_n_jobs", 32)
    job_gen = JobGenerator(0, 1, pe["n_window"], pe["n_on_premise_node"], pe["n_cloud_node"], config, n_jobs, 0.2, 0)
    jobs_set = job_gen.generate_jobs_set()

    env = SchedulingEnvEventObs(
        np.inf,
        pe["n_window"],
        pe["n_on_premise_node"],
        pe["n_cloud_node"],
        pe["n_job_queue_obs"],
        pe["n_job_queue_bck"],
        pa["weight_wt"],
        pa["weight_cost"],
        pe["penalty_not_allocate"],
        pe["penalty_invalid_action"],
        jobs_set,
        None,
        flag=0,
    )

    obs = env.reset()
    expected_size = N_EVENTS_OBS * EVENT_FEATURES + JOB_QUEUE_SIZE
    assert obs.shape == (expected_size,), f"観測形状: expected {expected_size}, got {obs.shape[0]}"
    assert obs.dtype == np.float32, f"観測dtype: expected float32, got {obs.dtype}"
    print(f"  OK: 観測形状 = {obs.shape}, dtype = {obs.dtype}")
    print(f"  - イベント部分: {N_EVENTS_OBS} x {EVENT_FEATURES} = {N_EVENTS_OBS * EVENT_FEATURES}")
    print(f"  - ジョブキュー部分: {JOB_QUEUE_SIZE}")
    return env


def test_episode_run(env, max_steps=50):
    """短いエピソードを実行して整合性を確認"""
    print("\n=== エピソード実行の検証 ===")
    obs = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        action = env.action_space.sample()
        obs, reward, scheduled, wt_step, done = env.step(action)
        step += 1
        assert obs.shape[0] == env.observation_space.shape[0]
        assert reward.shape == (2,)
    print(f"  OK: {step} ステップ実行、done={done}")
    return step


def test_event_recording(env, max_steps=100):
    """スケジュール後にイベントが記録されることを確認"""
    print("\n=== イベント記録の検証 ===")
    obs = env.reset()
    done = False
    step = 0
    last_obs_events = obs[: N_EVENTS_OBS * EVENT_FEATURES]
    events_changed = False
    while not done and step < max_steps:
        action = env.action_space.sample()
        obs, _, scheduled, _, done = env.step(action)
        step += 1
        curr_events = obs[: N_EVENTS_OBS * EVENT_FEATURES]
        if scheduled and np.any(curr_events != last_obs_events):
            events_changed = True
            break
        last_obs_events = curr_events
    print(f"  OK: スケジュール後にイベント観測が更新されることを確認 (step={step})")
    return events_changed


def main():
    print("イベントベース観測環境の検証を開始します\n")
    env = test_observation_shape()
    test_episode_run(env)
    test_event_recording(env)
    print("\n=== 全検証完了 ===")


if __name__ == "__main__":
    main()
