#!/usr/bin/env python3
"""
動作確認スクリプト（C拡張 + 環境の簡易検証）

使い方:
  uv run python scripts/verify.py
  # または
  uv run verify
"""
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    errors = []

    # 1. scheduling_env_core
    try:
        from scheduling_env_core import WindowCache, find_allocation_position
        import numpy as np
        H, W = 5, 20
        window = np.zeros((H, W), dtype=np.int32)
        cache = WindowCache(window, H, W)
        pos, wt = find_allocation_position(cache, 2, 2, 0, 0)
        print("✓ scheduling_env_core")
    except Exception as e:
        errors.append(f"scheduling_env_core: {e}")
        print("✗ scheduling_env_core")

    # 2. nsga2_core
    try:
        import nsga2_core
        import numpy as np
        obj = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]], dtype=np.float64)
        ranks = nsga2_core.non_dominated_sort(obj)
        dist = nsga2_core.calculate_crowding_distance(obj)
        print("✓ nsga2_core")
    except Exception as e:
        errors.append(f"nsga2_core: {e}")
        print("✗ nsga2_core")

    # 3. SchedulingEnvCacheOptimized（数ステップ実行）
    try:
        import numpy as np
        import yaml
        from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
        from src.utils.job_gen.job_generator import JobGenerator

        with open("config/config.yml") as f:
            config = yaml.safe_load(f)
        job_gen = JobGenerator(
            0, 1, config["param_env"]["n_window"],
            config["param_env"]["n_on_premise_node"],
            config["param_env"]["n_cloud_node"],
            config, 5, 0.2, 0
        )
        jobs_set = job_gen.generate_jobs_set()

        env = SchedulingEnvCacheOptimized(
            float("inf"), config["param_env"]["n_window"],
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
        obs = env.reset()
        for _ in range(3):
            action = env.action_space.sample()
            obs, rewards, scheduled, wt_step, done = env.step(action)
            if done:
                break
        print("✓ SchedulingEnvCacheOptimized")
    except Exception as e:
        errors.append(f"SchedulingEnvCacheOptimized: {e}")
        print("✗ SchedulingEnvCacheOptimized")

    if errors:
        print("\n--- エラー詳細 ---")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("\n✓ 動作確認完了")


if __name__ == "__main__":
    main()
