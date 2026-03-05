#!/usr/bin/env python3
"""
8000ジョブ環境でのメモリ使用量プロファイリングスクリプト

各Actorが約29GB使用する原因を特定するため:
- 1エピソードあたりのメモリ使用量
- 観測空間のサイズ
- collected_episodesの蓄積によるメモリ増加
"""
import os
import sys
import yaml
import numpy as np
import psutil

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_memory_mb():
    """現在プロセスのメモリ使用量をMBで取得"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    with open("config/config.yml") as f:
        config = yaml.safe_load(f)
    
    # distributed_pcnと同じ設定
    N_JOBS = 8000
    config['param_env'] = config.get('param_env', {})
    config['param_env']['n_jobs'] = N_JOBS
    
    n_window = config['param_env']['n_window']
    n_on_premise = config['param_env']['n_on_premise_node']
    n_cloud = config['param_env']['n_cloud_node']
    obs_window_size = 30  # scheduling_env.py より
    
    # 観測空間サイズ
    obs_size = n_on_premise * obs_window_size + n_cloud * obs_window_size + 8 * 5
    obs_bytes = obs_size * 4  # float32
    
    print("=" * 60)
    print("8000ジョブ環境 メモリプロファイリング")
    print("=" * 60)
    print(f"観測空間: {obs_size} floats = {obs_bytes/1024:.1f} KB per observation")
    print(f"1 Transition (obs + next_obs): ~{obs_bytes*2/1024:.1f} KB")
    print(f"1エピソード (8000 steps): ~{obs_bytes*2*8000/(1024*1024):.1f} MB")
    print(f"50エピソード (collected_episodes): ~{obs_bytes*2*8000*50/(1024*1024*1024):.1f} GB")
    print()
    
    # 実際に環境を作成してメモリを計測
    from src.utils.job_gen.job_generator import JobGenerator
    from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
    
    mem_before = get_memory_mb()
    
    job_gen = JobGenerator(0, 1, n_window, n_on_premise, n_cloud, config, N_JOBS, 0.2, 0)
    jobs_set = job_gen.generate_jobs_set()
    
    mem_after_jobs = get_memory_mb()
    print(f"JobGenerator + jobs_set: {mem_after_jobs - mem_before:.1f} MB")
    
    env = SchedulingEnvCacheOptimized(
        np.inf, n_window, n_on_premise, n_cloud,
        config['param_env']['n_job_queue_obs'],
        config['param_env']['n_job_queue_bck'],
        config['param_agent']['weight_wt'],
        config['param_agent']['weight_cost'],
        config['param_env']['penalty_not_allocate'],
        config['param_env']['penalty_invalid_action'],
        jobs_set, None, flag=0
    )
    
    mem_after_env = get_memory_mb()
    print(f"環境 (SchedulingEnvCacheOptimized): {mem_after_env - mem_after_jobs:.1f} MB")
    
    # 1エピソード分のTransitionをシミュレート
    from src.agents.pcn_agent import Transition
    
    obs = env.reset()
    transitions = []
    step_count = 0
    max_steps = min(100, 8000)  # 100ステップでサンプル（8000は時間がかかる）
    
    for _ in range(max_steps):
        action = env.action_space.sample()
        n_obs, reward, _, _, done = env.step(action)
        obs_f32 = np.array(obs, dtype=np.float32, copy=True)
        n_obs_f32 = np.array(n_obs, dtype=np.float32, copy=True)
        transitions.append(Transition(obs_f32, action, np.float32(reward).copy(), n_obs_f32, done))
        obs = n_obs
        if done:
            break
    
    mem_after_episode = get_memory_mb()
    episode_mem = mem_after_episode - mem_after_env
    print(f"\n{len(transitions)} ステップのエピソード: {episode_mem:.1f} MB")
    print(f"  1ステップあたり: {episode_mem/len(transitions)*1024:.0f} KB")
    
    # 8000ステップに換算
    estimated_8000 = episode_mem * (8000 / len(transitions))
    print(f"  8000ステップ換算: ~{estimated_8000:.1f} MB per episode")
    print(f"  50エピソード (Actorのcollected_episodes): ~{estimated_8000*50/1024:.1f} GB")
    print()
    print("結論: Actorが全50エピソードをメモリに保持してから送信する設計が")
    print("      8000ジョブ×16ActorでOOMの主要原因。ストリーミング送信が必要。")

if __name__ == "__main__":
    main()
