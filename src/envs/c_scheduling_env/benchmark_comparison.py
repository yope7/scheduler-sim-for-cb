#!/usr/bin/env python3
"""
既存のPython実装とC言語実装の性能比較
"""
import numpy as np
import time
import sys
import os

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from scheduling_env_core import WindowCache, find_allocation_position as c_find_allocation_position
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    print("警告: C言語実装が利用できません。Python実装のみで比較します。")

# 既存のPython実装をインポート
from src.envs.scheduling_env import SchedulingEnv


def benchmark_python_implementation(env, n_iterations=100):
    """Python実装のベンチマーク"""
    print("\n=== Python実装のベンチマーク ===")
    
    # キャッシュを構築
    cache_onpre = env._rebuild_cache_if_needed(use_cloud=False)
    cache_cloud = env._rebuild_cache_if_needed(use_cloud=True)
    
    # テスト用のジョブを準備
    job = np.array([5, 3, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    action = [0, 0]  # オンプレミス
    
    start_time = time.time()
    for _ in range(n_iterations):
        position, waiting_time = env.find_allocation_position(
            action, cache_onpre=cache_onpre, cache_cloud=cache_cloud
        )
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / n_iterations * 1000  # ms
    
    print(f"✓ {n_iterations}回の探索を {elapsed:.3f}秒で完了")
    print(f"  平均時間: {avg_time:.3f}ms/回")
    
    return elapsed, avg_time


def benchmark_c_implementation(n_iterations=100):
    """C言語実装のベンチマーク"""
    if not C_AVAILABLE:
        return None, None
    
    print("\n=== C言語実装のベンチマーク ===")
    
    H, W = 50, 500
    window_status = np.zeros((H, W), dtype=np.int32)
    
    # ランダムにセルを占有（既存実装と同様の状態を作成）
    np.random.seed(42)
    occupied_mask = np.random.random((H, W)) < 0.3
    window_status[occupied_mask] = 1
    
    cache = WindowCache(window_status, H, W)
    
    job_width, job_height = 5, 3
    when_submitted, current_time = 0, 10
    
    start_time = time.time()
    for _ in range(n_iterations):
        position, waiting_time = c_find_allocation_position(
            cache, job_width, job_height,
            when_submitted, current_time
        )
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / n_iterations * 1000  # ms
    
    print(f"✓ {n_iterations}回の探索を {elapsed:.3f}秒で完了")
    print(f"  平均時間: {avg_time:.3f}ms/回")
    
    return elapsed, avg_time


def main():
    """メイン関数"""
    print("=" * 60)
    print("Python実装 vs C言語実装の性能比較")
    print("=" * 60)
    
    # 環境を初期化
    config = {
        'param_env': {
            'n_window': 500,
            'n_on_premise_node': 50,
            'n_cloud_node': 50,
            'n_job_queue_obs': 5,
            'n_job_queue_bck': 10,
            'penalty_not_allocate': 0,
            'penalty_invalid_action': 0
        },
        'param_agent': {
            'weight_wt': 1.0,
            'weight_cost': 1.0
        }
    }
    
    # ダミーのジョブセットを作成
    jobs_set = {0: np.array([[0, 5, 3, 0, 0, 1, 0, 0]], dtype=np.float32)}
    
    env = SchedulingEnv(
        max_step=1000,
        n_window=config['param_env']['n_window'],
        n_on_premise_node=config['param_env']['n_on_premise_node'],
        n_cloud_node=config['param_env']['n_cloud_node'],
        n_job_queue_obs=config['param_env']['n_job_queue_obs'],
        n_job_queue_bck=config['param_env']['n_job_queue_bck'],
        weight_wt=config['param_agent']['weight_wt'],
        weight_cost=config['param_agent']['weight_cost'],
        penalty_not_allocate=config['param_env']['penalty_not_allocate'],
        penalty_invalid_action=config['param_env']['penalty_invalid_action'],
        jobs_set=jobs_set,
        flag=0
    )
    
    env.reset()
    
    # ベンチマーク実行
    n_iterations = 1000
    
    python_elapsed, python_avg = benchmark_python_implementation(env, n_iterations)
    c_elapsed, c_avg = benchmark_c_implementation(n_iterations)
    
    # 結果の比較
    print("\n" + "=" * 60)
    print("性能比較結果")
    print("=" * 60)
    
    print(f"Python実装:")
    print(f"  総時間: {python_elapsed:.3f}秒")
    print(f"  平均時間: {python_avg:.3f}ms/回")
    
    if C_AVAILABLE:
        print(f"\nC言語実装:")
        print(f"  総時間: {c_elapsed:.3f}秒")
        print(f"  平均時間: {c_avg:.3f}ms/回")
        
        speedup = python_elapsed / c_elapsed
        print(f"\n✓ 高速化率: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  C言語実装が {speedup:.2f}倍高速です")
        else:
            print(f"  Python実装が {1.0/speedup:.2f}倍高速です（予期しない結果）")
    else:
        print("\nC言語実装は利用できませんでした")


if __name__ == "__main__":
    main()

