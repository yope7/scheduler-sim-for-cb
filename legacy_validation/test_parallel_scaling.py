#!/usr/bin/env python3
"""
並列数のスケーラビリティテスト
異なる並列数で実行時間を計測
"""
import sys
import os
import time
import numpy as np
import yaml

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.agents.nsga2_agent import NSGA2Agent
from src.envs.scheduling_variants.bitmap_c_env import SchedulingEnvCacheOptimized
from src.utils.job_gen.job_generator import JobGenerator

def test_parallel_scaling(nb_jobs=20, pop_size=50, num_generations=5, parallel_counts=None):
    """異なる並列数で実行時間を計測"""
    
    # デフォルトの並列数リスト
    if parallel_counts is None:
        cpu_count = os.cpu_count()
        # 1, 2, 4, 8, 16, 32, 64, 96, 128までテスト
        parallel_counts = [1, 2, 4, 8, 16, 32, min(64, cpu_count), min(96, cpu_count * 2), min(128, cpu_count * 2)]
        # CPUコア数を超えないように調整
        parallel_counts = [p for p in parallel_counts if p <= cpu_count * 2]
    
    # 設定ファイルを読み込み
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # ジョブセットを生成
    job_gen = JobGenerator(
        0, 1,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, nb_jobs, 0.2, 0
    )
    jobs_set = job_gen.generate_jobs_set()
    
    results = []
    
    print("=" * 80)
    print(f"並列数スケーラビリティテスト")
    print(f"  ジョブ数: {nb_jobs}")
    print(f"  集団サイズ: {pop_size}")
    print(f"  世代数: {num_generations}")
    print(f"  CPUコア数: {os.cpu_count()}")
    print("=" * 80)
    print()
    
    for n_workers in parallel_counts:
        print(f"並列数: {n_workers:3d} でテスト中...", end=" ", flush=True)
        
        # 環境を作成
        env = SchedulingEnvCacheOptimized(
            max_step=np.inf,
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
        
        # エージェントを作成
        agent = NSGA2Agent(
            pop_size=pop_size,
            num_generations=num_generations,
            crossover_prob=0.8,
            mutation_prob=0.1
        )
        
        # 実行時間を計測
        start_time = time.perf_counter()
        try:
            result = agent.run(env, nb_jobs, n_jobs=n_workers)
            elapsed_time = time.perf_counter() - start_time
            
            # 結果を記録
            results.append({
                'n_workers': n_workers,
                'elapsed_time': elapsed_time,
                'success': True,
                'pareto_size': len(result) if result else 0
            })
            
            print(f"✓ 完了: {elapsed_time:.2f}秒 (パレート解: {len(result) if result else 0}個)")
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            results.append({
                'n_workers': n_workers,
                'elapsed_time': elapsed_time,
                'success': False,
                'error': str(e)
            })
            print(f"✗ エラー: {e}")
        
        # クリーンアップ
        if hasattr(agent, 'pool') and agent.pool is not None:
            agent.pool.close()
            agent.pool.join()
            agent.pool = None
    
    # 結果を表示
    print()
    print("=" * 80)
    print("結果サマリー")
    print("=" * 80)
    print(f"{'並列数':>8} | {'実行時間(秒)':>12} | {'スピードアップ':>12} | {'効率':>8} | {'状態':>10}")
    print("-" * 80)
    
    baseline_time = None
    for r in results:
        if r['success']:
            if baseline_time is None:
                baseline_time = r['elapsed_time']
                speedup = 1.0
                efficiency = 1.0
            else:
                speedup = baseline_time / r['elapsed_time']
                efficiency = speedup / r['n_workers'] * 100
            
            status = "成功"
            print(f"{r['n_workers']:>8} | {r['elapsed_time']:>12.2f} | {speedup:>12.2f}x | {efficiency:>7.1f}% | {status:>10}")
        else:
            print(f"{r['n_workers']:>8} | {'N/A':>12} | {'N/A':>12} | {'N/A':>8} | {'エラー':>10}")
    
    print("=" * 80)
    
    # 最適な並列数を推奨
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['elapsed_time'])
        print(f"\n最速実行時間: {best_result['elapsed_time']:.2f}秒 (並列数: {best_result['n_workers']})")
        
        # 効率が高い並列数を推奨（効率80%以上）
        efficient_results = [r for r in successful_results if r['n_workers'] > 1]
        if efficient_results:
            baseline = min([r for r in successful_results if r['n_workers'] == 1], key=lambda x: x['elapsed_time'], default=None)
            if baseline:
                for r in efficient_results:
                    speedup = baseline['elapsed_time'] / r['elapsed_time']
                    efficiency = speedup / r['n_workers'] * 100
                    if efficiency >= 80:
                        print(f"推奨並列数: {r['n_workers']} (効率: {efficiency:.1f}%)")
                        break
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='並列数のスケーラビリティテスト')
    parser.add_argument('--nb_jobs', type=int, default=20, help='ジョブ数')
    parser.add_argument('--pop_size', type=int, default=50, help='集団サイズ')
    parser.add_argument('--num_generations', type=int, default=5, help='世代数')
    parser.add_argument('--parallel_counts', type=int, nargs='+', default=None, help='テストする並列数のリスト')
    
    args = parser.parse_args()
    
    test_parallel_scaling(
        nb_jobs=args.nb_jobs,
        pop_size=args.pop_size,
        num_generations=args.num_generations,
        parallel_counts=args.parallel_counts
    )

