#!/usr/bin/env python3
"""find_allocation_position のプロファイリング"""
import sys
import os
import time
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
from src.utils.job_gen.job_generator import JobGenerator
from src.agents.heuristic_agent import HeuristicAgent


def load_config(path: str = "config/config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    nb_jobs = 2000
    job_gen = JobGenerator(
        42, 1,
        config["param_env"]["n_window"],
        config["param_env"]["n_on_premise_node"],
        config["param_env"]["n_cloud_node"],
        config, nb_jobs, 0.2, 0
    )
    jobs_set = job_gen.generate_jobs_set()

    env = SchedulingEnvCacheOptimized(
        float("inf"),
        config["param_env"]["n_window"],
        config["param_env"]["n_on_premise_node"],
        config["param_env"]["n_cloud_node"],
        config["param_env"]["n_job_queue_obs"],
        config["param_env"]["n_job_queue_bck"],
        config["param_agent"]["weight_wt"],
        config["param_agent"]["weight_cost"],
        config["param_env"]["penalty_not_allocate"],
        config["param_env"]["penalty_invalid_action"],
        jobs_set, None, flag=0
    )
    agent = HeuristicAgent(base_wait_time_threshold=5, width_factor=0.3, use_cloud_fallback=True)

    env.reset()
    max_steps = min(nb_jobs * 10, 50000)
    step_count = 0
    done = False

    # ウォームアップ
    for _ in range(100):
        action_raw, is_valid = agent.select_action(env)
        obs, rewards, scheduled, wt_step, done = env.step(action_raw if is_valid else 0)
        step_count += 1
        if done:
            env.reset()
            done = False

    # プロファイリング: find_allocation_position の時間
    env.reset()
    step_count = 0
    done = False
    t_find_total = 0.0
    t_step_total = 0.0
    n_find_calls = 0
    n_steps = 0

    while not env.check_is_done() and step_count < max_steps:
        action_raw, is_valid = agent.select_action(env)
        act = env.get_converted_action(action_raw if is_valid else 0)

        t0 = time.perf_counter()
        position, wt_real = env.find_allocation_position(act)
        t_find = time.perf_counter() - t0
        t_find_total += t_find
        n_find_calls += 1

        t0 = time.perf_counter()
        obs, rewards, scheduled, wt_step, done = env.step(action_raw if is_valid else 0)
        t_step_total += time.perf_counter() - t0
        n_steps += 1
        step_count += 1
        if done:
            break

    print(f"n_steps={n_steps}, n_find_calls={n_find_calls}")
    print(f"find_allocation_position: {t_find_total:.4f}s total, {t_find_total/n_find_calls*1000:.3f}ms/call")
    print(f"env.step (total):          {t_step_total:.4f}s total, {t_step_total/n_steps*1000:.3f}ms/step")
    print(f"find_allocation_position の割合: {t_find_total/t_step_total*100:.1f}%")


if __name__ == "__main__":
    main()
