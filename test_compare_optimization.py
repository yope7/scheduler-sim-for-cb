#!/usr/bin/env python3
"""
最適化前後のパフォーマンスを比較するスクリプト
"""
import sys
import os
import numpy as np
import time
import yaml
from typing import Dict, Any, List

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.job_gen.job_generator import JobGenerator

def load_config():
    """設定ファイルを読み込み"""
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    return config

def create_large_scale_jobs(nb_jobs: int, config: Dict[str, Any], *, seed: int = 0, nb_steps: int = 1, nb_episodes: int = 0) -> Dict[int, List]:
    """JobGeneratorを用いてジョブセットを作成"""
    lam = config['param_job'].get('lam', 0.2)
    job_generator = JobGenerator(
        seed, nb_steps,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, nb_jobs, lam, nb_episodes
    )
    return job_generator.generate_jobs_set()

def compare_implementations(nb_jobs: int, *, seed: int = 0, nb_steps: int = 1, nb_episodes: int = 0):
    """最適化前後のパフォーマンスを比較"""
    print(f"\n{'='*80}")
    print(f"最適化前後のパフォーマンス比較: {nb_jobs}ジョブ")
    print(f"{'='*80}")
    
    # 設定を読み込み
    config = load_config()
    
    # 同じジョブセットを使用（公平な比較のため）
    jobs_set = create_large_scale_jobs(nb_jobs, config, seed=seed, nb_steps=nb_steps, nb_episodes=nb_episodes)
    
    # 環境パラメータを設定
    max_step = np.inf
    n_window = config['param_env']['n_window']
    n_on_premise_node = config['param_env']['n_on_premise_node']
    n_cloud_node = config['param_env']['n_cloud_node']
    n_job_queue_obs = config['param_env']['n_job_queue_obs']
    n_job_queue_bck = config['param_env']['n_job_queue_bck']
    weight_wt = config['param_agent']['weight_wt']
    weight_cost = config['param_agent']['weight_cost']
    penalty_not_allocate = config['param_env']['penalty_not_allocate']
    penalty_invalid_action = config['param_env']['penalty_invalid_action']
    
    print("\n" + "="*80)
    print("1. C言語実装版（最適化前）")
    print("="*80)
    
    # C言語実装版のテスト（test_large_scale_timing_c.pyから関数をインポート）
    from test_large_scale_timing_c import run_environment_timing_test as run_c_test
    results_c = run_c_test(
        nb_jobs,
        use_heuristic=True,
        seed=seed,
        nb_steps=nb_steps,
        nb_episodes=nb_episodes
    )
    
    print("\n" + "="*80)
    print("2. C言語実装最適化版（最適化後）")
    print("="*80)
    
    # 最適化版のテスト（test_large_scale_timing_optimized.pyから関数をインポート）
    from test_large_scale_timing_optimized import run_environment_timing_test as run_optimized_test
    results_optimized = run_optimized_test(
        nb_jobs,
        use_heuristic=True,
        seed=seed,
        nb_steps=nb_steps,
        nb_episodes=nb_episodes
    )
    
    # 結果を比較
    print("\n" + "="*80)
    print("パフォーマンス比較結果")
    print("="*80)
    
    print(f"\n{'項目':<30} {'C実装版':<20} {'最適化版':<20} {'改善率':<15}")
    print("-" * 85)
    
    # 総実行時間
    total_time_c = results_c['total_time']
    total_time_opt = results_optimized['total_time']
    improvement = (total_time_c - total_time_opt) / total_time_c * 100
    print(f"{'総実行時間 (秒)':<30} {total_time_c:<20.3f} {total_time_opt:<20.3f} {improvement:>14.1f}%")
    
    # メイン実行時間
    main_time_c = results_c['main_execution_time']
    main_time_opt = results_optimized['main_execution_time']
    improvement = (main_time_c - main_time_opt) / main_time_c * 100
    print(f"{'メイン実行時間 (秒)':<30} {main_time_c:<20.3f} {main_time_opt:<20.3f} {improvement:>14.1f}%")
    
    # 平均ステップ時間
    avg_step_c = results_c['avg_step_time']
    avg_step_opt = results_optimized['avg_step_time']
    improvement = (avg_step_c - avg_step_opt) / avg_step_c * 100
    print(f"{'平均ステップ時間 (ms)':<30} {avg_step_c*1000:<20.2f} {avg_step_opt*1000:<20.2f} {improvement:>14.1f}%")
    
    # 平均スケジュール時間
    avg_schedule_c = results_c['avg_schedule_time']
    avg_schedule_opt = results_optimized['avg_schedule_time']
    improvement = (avg_schedule_c - avg_schedule_opt) / avg_schedule_c * 100
    print(f"{'平均スケジュール時間 (ms)':<30} {avg_schedule_c*1000:<20.2f} {avg_schedule_opt*1000:<20.2f} {improvement:>14.1f}%")
    
    # ジョブあたり平均時間
    if results_c['scheduled_jobs'] > 0 and results_optimized['scheduled_jobs'] > 0:
        job_time_c = results_c['main_execution_time'] / results_c['scheduled_jobs']
        job_time_opt = results_optimized['main_execution_time'] / results_optimized['scheduled_jobs']
        improvement = (job_time_c - job_time_opt) / job_time_c * 100
        print(f"{'ジョブあたり平均時間 (ms)':<30} {job_time_c*1000:<20.2f} {job_time_opt*1000:<20.2f} {improvement:>14.1f}%")
    
    # スケジュール率
    schedule_rate_c = results_c['scheduled_jobs'] / results_c['nb_jobs'] * 100
    schedule_rate_opt = results_optimized['scheduled_jobs'] / results_optimized['nb_jobs'] * 100
    print(f"{'スケジュール率 (%)':<30} {schedule_rate_c:<20.1f} {schedule_rate_opt:<20.1f} {'-':>15}")
    
    # 目的関数値の比較
    print(f"\n{'目的関数値の比較':<30}")
    print("-" * 85)
    print(f"{'総コスト':<30} {results_c['total_cost']:<20.2f} {results_optimized['total_cost']:<20.2f}")
    print(f"{'メイクスパン':<30} {results_c['makespan']:<20.2f} {results_optimized['makespan']:<20.2f}")
    print(f"{'平均待ち時間':<30} {results_c['avg_waiting_time']:<20.2f} {results_optimized['avg_waiting_time']:<20.2f}")
    
    # 結果の要約
    print("\n" + "="*80)
    print("結果の要約")
    print("="*80)
    
    if total_time_opt < total_time_c:
        speedup = total_time_c / total_time_opt
        print(f"✓ 最適化版は {speedup:.2f}倍高速です")
    else:
        slowdown = total_time_opt / total_time_c
        print(f"⚠ 最適化版は {slowdown:.2f}倍低速です（予期しない結果）")
    
    if abs(results_c['total_cost'] - results_optimized['total_cost']) < 1e-6:
        print("✓ 目的関数値は完全に一致しています（最適化の影響なし）")
    else:
        print(f"⚠ 目的関数値が異なります（C実装版: {results_c['total_cost']}, 最適化版: {results_optimized['total_cost']}）")
    
    return {
        'c_implementation': results_c,
        'optimized': results_optimized,
        'speedup': total_time_c / total_time_opt if total_time_opt > 0 else 0
    }

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='最適化前後のパフォーマンス比較')
    parser.add_argument('--nb_jobs', type=int, default=100, help='ジョブ数')
    parser.add_argument('--seed', type=int, default=0, help='ジョブ生成の乱数シード')
    parser.add_argument('--nb_steps', type=int, default=1, help='ジョブ生成ステップ')
    parser.add_argument('--episodes', type=int, default=0, help='生成するエピソード数')
    
    args = parser.parse_args()
    
    compare_implementations(
        args.nb_jobs,
        seed=args.seed,
        nb_steps=args.nb_steps,
        nb_episodes=args.episodes
    )

if __name__ == "__main__":
    main()

