#!/usr/bin/env python3
"""
ビットマップ版とイベント版の結果一致検証

同一のジョブセット・同一の行動列で両環境を実行し、
目的関数値（コスト、makespan、平均待ち時間）が一致することを確認する。
観測は異なるが、スケジューリングロジックは同一のため結果は一致するはず。
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import yaml

from src.envs.scheduling_variants.bitmap_c_env import SchedulingEnvCacheOptimized
from src.envs.scheduling_variants.event_c_env import SchedulingEnvEventObs
from src.utils.job_gen.job_generator import JobGenerator


def load_config():
    config_path = project_root / "config" / "config.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_episode_with_actions(env, actions, seed=42):
    """指定した行動列でエピソードを実行し、目的関数値を返す"""
    np.random.seed(seed)
    obs = env.reset()
    done = False
    step = 0
    while not done and step < len(actions):
        action = actions[step]
        obs, reward, scheduled, wt_step, done = env.step(action)
        step += 1
    env.finalize_window_history()
    cost, makespan, avg_wt = env.calc_objective_values()
    return cost, makespan, avg_wt, step


def main():
    print("=== ビットマップ版 vs イベント版 結果一致検証 ===\n")
    config = load_config()
    pe = config["param_env"]
    pa = config["param_agent"]
    pj = config["param_job"]

    # 第1引数でジョブ数を指定可能（例: python3 scripts/verify_event_vs_bitmap_equivalence.py 128）
    n_jobs = pj.get("job_trace_n_jobs", 32)
    if len(sys.argv) > 1:
        n_jobs = int(sys.argv[1])
    print(f"n_jobs = {n_jobs}\n")
    seed = 42
    np.random.seed(seed)
    nb_episodes = 2  # episode 0, 1 の検証用
    job_gen = JobGenerator(0, 1, pe["n_window"], pe["n_on_premise_node"], pe["n_cloud_node"], config, n_jobs, 0.2, nb_episodes)
    jobs_set = job_gen.generate_jobs_set()

    # 両環境を同じジョブセットで作成
    def make_env(EnvClass):
        return EnvClass(
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

    env_bitmap = make_env(SchedulingEnvCacheOptimized)
    env_event = make_env(SchedulingEnvEventObs)

    # 同じシードで行動列を事前生成（両環境で同じ行動を使う）
    np.random.seed(seed)
    actions_bitmap = []
    obs = env_bitmap.reset()
    done = False
    while not done:
        a = env_bitmap.action_space.sample()
        actions_bitmap.append(a)
        obs, _, _, _, done = env_bitmap.step(a)

    # イベント版でも同じ行動列を取得（ステップ数は同じになるはず）
    np.random.seed(seed)
    actions_event = []
    obs = env_event.reset()
    done = False
    while not done:
        a = env_event.action_space.sample()
        actions_event.append(a)
        obs, _, _, _, done = env_event.step(a)

    # 行動列が同じ長さか確認（同じジョブセットなら同じ）
    assert len(actions_bitmap) == len(actions_event), f"行動数不一致: {len(actions_bitmap)} vs {len(actions_event)}"
    actions = actions_bitmap
    print(f"エピソード長: {len(actions)} ステップ\n")

    # 固定の行動列で両環境を実行（リセットしてから同じ行動を適用）
    def run_with_fixed_actions(env, actions):
        obs = env.reset()
        done = False
        for a in actions:
            if done:
                break
            obs, _, _, _, done = env.step(a)
        env.finalize_window_history()
        return env.calc_objective_values()

    env_b = make_env(SchedulingEnvCacheOptimized)
    env_e = make_env(SchedulingEnvEventObs)
    cost_b, mks_b, wt_b = run_with_fixed_actions(env_b, actions)
    cost_e, mks_e, wt_e = run_with_fixed_actions(env_e, actions)

    print("目的関数値の比較:")
    print(f"  ビットマップ版: cost={cost_b}, makespan={mks_b}, avg_waiting_time={wt_b:.6f}")
    print(f"  イベント版:     cost={cost_e}, makespan={mks_e}, avg_waiting_time={wt_e:.6f}")

    tol = 1e-5
    ok_cost = abs(cost_b - cost_e) < tol
    ok_mks = abs(mks_b - mks_e) < tol
    ok_wt = abs(wt_b - wt_e) < tol

    print()
    if ok_cost and ok_mks and ok_wt:
        print("✓ 結果は一致しました（ビットマップ版とイベント版で同一）")
    else:
        print("✗ 結果に差異があります:")
        if not ok_cost:
            print(f"  - cost: 差={abs(cost_b - cost_e)}")
        if not ok_mks:
            print(f"  - makespan: 差={abs(mks_b - mks_e)}")
        if not ok_wt:
            print(f"  - avg_waiting_time: 差={abs(wt_b - wt_e)}")
        sys.exit(1)

    # 複数エピソードで検証（episode=1でもう1回）
    print("\n--- 複数エピソード検証 (episode=1) ---")
    env_b2 = make_env(SchedulingEnvCacheOptimized)
    env_e2 = make_env(SchedulingEnvEventObs)
    env_b2.episode = 1
    env_e2.episode = 1
    np.random.seed(seed + 1)
    acts = []
    obs = env_b2.reset()
    done = False
    while not done:
        a = env_b2.action_space.sample()
        acts.append(a)
        obs, _, _, _, done = env_b2.step(a)
    cost_b2, mks_b2, wt_b2 = run_with_fixed_actions(env_b2, acts)
    cost_e2, mks_e2, wt_e2 = run_with_fixed_actions(env_e2, acts)
    print(f"  episode=1: bitmap=({cost_b2},{mks_b2},{wt_b2:.4f}) event=({cost_e2},{mks_e2},{wt_e2:.4f})")
    if abs(cost_b2 - cost_e2) < tol and abs(mks_b2 - mks_e2) < tol and abs(wt_b2 - wt_e2) < tol:
        print("  ✓ episode=1 も一致")
    else:
        print("  ✗ episode=1 で不一致")
        sys.exit(1)

    print("\n=== 検証完了: 結果は一致しています ===")


if __name__ == "__main__":
    main()
