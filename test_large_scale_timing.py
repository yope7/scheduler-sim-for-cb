#!/usr/bin/env python3
"""
数千ジョブでの割り当て時間計測スクリプト
main.pyの挙動を真似て、環境自体の時間を測定する
"""

import sys
import os
import numpy as np
import time
import yaml
import argparse
import cProfile
import pstats
from typing import List, Dict, Any

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# C実装環境を優先して使用（main.py準拠）
try:
    from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
    C_AVAILABLE = True
except ImportError:
    SchedulingEnvCacheOptimized = None
    C_AVAILABLE = False
    print("警告: C言語実装が利用できません。Python実装を使用します。")

from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator

def load_config():
    """設定ファイルを読み込み"""
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    return config

def create_large_scale_jobs(nb_jobs: int, config: Dict[str, Any], *, seed: int = 0, nb_steps: int = 1, nb_episodes: int = 0) -> Dict[int, List]:
    """JobGeneratorを用いてジョブセットを作成（main.py準拠）"""
    print(f"=== JobGeneratorで{nb_jobs}ジョブのサンプルデータを作成中 ===")

    lam = config['param_job'].get('lam', 0.2)

    job_generator = JobGenerator(
        seed,
        nb_steps,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config,
        nb_jobs,
        lam,
        nb_episodes
    )

    jobs_set = job_generator.generate_jobs_set()

    if isinstance(jobs_set, dict) and jobs_set:
        first_episode = list(jobs_set.keys())[0]
        print(f"生成: {len(jobs_set)}エピソード / エピソード{first_episode}のジョブ数: {len(jobs_set[first_episode]) if jobs_set[first_episode] is not None else 0}")
    else:
        print("警告: jobs_setが空です")

    return jobs_set


def run_environment_timing_test(nb_jobs: int,
                               *,
                               use_heuristic: bool = True,
                               seed: int = 0,
                               nb_steps: int = 1,
                               nb_episodes: int = 0) -> Dict[str, Any]:
    """環境の時間計測テストを実行"""
    print(f"\n{'='*60}")
    print(f"環境時間計測テスト開始: {nb_jobs}ジョブ")
    print(f"開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 設定を読み込み
    config = load_config()
    
    # ジョブセットを作成（JobGeneratorを使用）
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
    
    # 環境を初期化（C実装があればそちらを優先）
    env_cls = SchedulingEnvCacheOptimized if C_AVAILABLE else SchedulingEnv
    env_label = "C実装" if C_AVAILABLE else "Python実装"
    print(f"\n環境を初期化中...（{env_label}）")
    env_start_time = time.time()
    
    env = env_cls(
        max_step, n_window, n_on_premise_node, n_cloud_node, 
        n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, 
        jobs_set, None, flag=0
    )
    
    env_init_time = time.time() - env_start_time
    print(f"環境初期化時間: {env_init_time:.3f}秒")
    
    # 環境をリセット
    reset_start_time = time.time()
    observation = env.reset()
    reset_time = time.time() - reset_start_time
    print(f"環境リセット時間: {reset_time:.3f}秒")
    
    # ヒューリスティックエージェントを初期化（オプション）
    agent = None
    if use_heuristic:
        from src.agents.heuristic_agent import HeuristicAgent
        agent = HeuristicAgent(
            base_wait_time_threshold=5,
            width_factor=0.3,
            use_cloud_fallback=True
        )
        print("ヒューリスティックエージェントを初期化しました")
    
    # メインの実行ループ（SchedulingEnvの標準的な使用方法）
    print(f"\nジョブ割り当て処理を開始...")
    main_start_time = time.time()
    
    step_count = 0
    scheduled_jobs = 0
    total_wait_time = 0
    total_cost = 0
    on_premise_allocations = 0
    cloud_allocations = 0
    
    # 時間計測用の変数
    step_times = []
    schedule_times = []
    
    # エピソードの実行（SchedulingEnvの標準的なstep()メソッドを使用）
    max_steps_limit = min(env.max_step, nb_jobs * 10) if env.max_step != np.inf else nb_jobs * 10
    done = False
    
    while not done and step_count < max_steps_limit:
        step_start_time = time.time()
        
        # SchedulingEnvの標準的なstep()メソッドを使用
        if use_heuristic and agent:
            # ヒューリスティックエージェントを使用
            action, is_valid = agent.select_action(env)
            if is_valid:
                schedule_start_time = time.time()
                observation, rewards, scheduled, wt_step, done = env.step(action)
                schedule_time = time.time() - schedule_start_time
                schedule_times.append(schedule_time)
                
                if scheduled:
                    scheduled_jobs += 1
                    total_wait_time += wt_step
                    total_cost += abs(rewards[1])  # コストの報酬（絶対値）
                    if action == 0:
                        on_premise_allocations += 1
                    elif action == 1:
                        cloud_allocations += 1
            else:
                # 無効なアクションの場合、デフォルトアクションを実行
                schedule_start_time = time.time()
                observation, rewards, scheduled, wt_step, done = env.step(0)  # オンプレミス
                schedule_time = time.time() - schedule_start_time
                schedule_times.append(schedule_time)
                
                if scheduled:
                    scheduled_jobs += 1
                    total_wait_time += wt_step
                    total_cost += abs(rewards[1])
                    on_premise_allocations += 1
        else:
            # シンプルなポリシー: オンプレミスを優先
            action = 0  # オンプレミス
            schedule_start_time = time.time()
            
            observation, rewards, scheduled, wt_step, done = env.step(action)
            schedule_time = time.time() - schedule_start_time
            schedule_times.append(schedule_time)
            
            if scheduled:
                scheduled_jobs += 1
                total_wait_time += wt_step
                total_cost += abs(rewards[1])
                on_premise_allocations += 1
        
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        step_count += 1
        
        # 進捗表示
        if step_count % max(1, nb_jobs // 10) == 0:
            elapsed = time.time() - main_start_time
            print(f"進捗: {step_count}ステップ, {scheduled_jobs}ジョブスケジュール済み, "
                  f"経過時間: {elapsed:.1f}秒")
    
    main_execution_time = time.time() - main_start_time
    
    # 環境の最終化
    finalize_start_time = time.time()
    env.finalize_window_history()
    finalize_time = time.time() - finalize_start_time
    
    # 目的関数値を計算
    calc_start_time = time.time()
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    calc_time = time.time() - calc_start_time
    
    # 結果をまとめる
    results = {
        'nb_jobs': nb_jobs,
        'env_init_time': env_init_time,
        'reset_time': reset_time,
        'main_execution_time': main_execution_time,
        'finalize_time': finalize_time,
        'calc_objective_time': calc_time,
        'total_time': env_init_time + reset_time + main_execution_time + finalize_time + calc_time,
        'step_count': step_count,
        'scheduled_jobs': scheduled_jobs,
        'total_wait_time': total_wait_time,
        'total_cost': total_cost,
        'avg_waiting_time': avg_waiting_time,
        'makespan': makespan,
        'step_times': step_times,
        'schedule_times': schedule_times,
        'avg_step_time': np.mean(step_times) if step_times else 0,
        'avg_schedule_time': np.mean(schedule_times) if schedule_times else 0,
        'max_step_time': np.max(step_times) if step_times else 0,
        'max_schedule_time': np.max(schedule_times) if schedule_times else 0,
        'use_heuristic': use_heuristic,
        'on_premise_allocations': on_premise_allocations,
        'cloud_allocations': cloud_allocations
    }
    
    return results

def print_timing_results(results: Dict[str, Any]):
    """時間計測結果を表示"""
    print(f"\n{'='*60}")
    print("時間計測結果")
    print(f"{'='*60}")
    
    print(f"ジョブ数: {results['nb_jobs']:,}")
    print(f"使用したポリシー: {'ヒューリスティック' if results['use_heuristic'] else 'シンプル'}")
    print(f"ジョブ生成方法: JobGenerator")
    
    print(f"\n--- 時間計測 ---")
    print(f"環境初期化時間: {results['env_init_time']:.3f}秒")
    print(f"環境リセット時間: {results['reset_time']:.3f}秒")
    print(f"メイン実行時間: {results['main_execution_time']:.3f}秒")
    print(f"環境最終化時間: {results['finalize_time']:.3f}秒")
    print(f"目的関数計算時間: {results['calc_objective_time']:.3f}秒")
    print(f"総実行時間: {results['total_time']:.3f}秒")
    
    print(f"\n--- 実行統計 ---")
    print(f"総ステップ数: {results['step_count']:,}")
    print(f"スケジュール済みジョブ数: {results['scheduled_jobs']:,}")
    print(f"スケジュール率: {results['scheduled_jobs']/results['nb_jobs']*100:.1f}%")
    print(f"オンプレ割り当て数: {results['on_premise_allocations']:,}")
    print(f"クラウド割り当て数: {results['cloud_allocations']:,}")
    
    print(f"\n--- パフォーマンス統計 ---")
    print(f"平均ステップ時間: {results['avg_step_time']*1000:.2f}ms")
    print(f"平均スケジュール時間: {results['avg_schedule_time']*1000:.2f}ms")
    print(f"最大ステップ時間: {results['max_step_time']*1000:.2f}ms")
    print(f"最大スケジュール時間: {results['max_schedule_time']*1000:.2f}ms")
    
    if results['scheduled_jobs'] > 0:
        print(f"ジョブあたり平均時間: {results['main_execution_time']/results['scheduled_jobs']*1000:.2f}ms")
        print(f"ステップあたり平均時間: {results['main_execution_time']/results['step_count']*1000:.2f}ms")
    
    print(f"\n--- 目的関数値 ---")
    print(f"総コスト: {results['total_cost']:.2f}")
    print(f"メイクスパン: {results['makespan']:.2f}")
    print(f"平均待ち時間: {results['avg_waiting_time']:.2f}")

def run_scaling_test(job_counts: List[int], use_heuristic: bool = True, *, seed: int = 0, nb_steps: int = 1, nb_episodes: int = 0):
    """複数のジョブ数でスケーリングテストを実行"""
    print(f"\n{'='*80}")
    print("スケーリングテスト開始")
    print(f"{'='*80}")
    
    all_results = []
    
    for nb_jobs in job_counts:
        print(f"\n{'='*40}")
        print(f"テスト: {nb_jobs}ジョブ")
        print(f"{'='*40}")
        
        try:
            results = run_environment_timing_test(
                nb_jobs,
                use_heuristic=use_heuristic,
                seed=seed,
                nb_steps=nb_steps,
                nb_episodes=nb_episodes
            )
            all_results.append(results)
            print_timing_results(results)
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # スケーリング結果の要約
    print(f"\n{'='*80}")
    print("スケーリングテスト結果要約")
    print(f"{'='*80}")
    
    print(f"{'ジョブ数':<10} {'総時間(秒)':<12} {'ジョブ/秒':<12} {'ステップ/秒':<12} {'スケジュール率(%)':<15}")
    print(f"{'-'*70}")
    
    for results in all_results:
        jobs_per_sec = results['scheduled_jobs'] / results['total_time'] if results['total_time'] > 0 else 0
        steps_per_sec = results['step_count'] / results['total_time'] if results['total_time'] > 0 else 0
        schedule_rate = results['scheduled_jobs'] / results['nb_jobs'] * 100 if results['nb_jobs'] > 0 else 0
        
        print(f"{results['nb_jobs']:<10} {results['total_time']:<12.3f} "
              f"{jobs_per_sec:<12.1f} {steps_per_sec:<12.1f} {schedule_rate:<15.1f}")
    
    return all_results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='数千ジョブでの割り当て時間計測')
    parser.add_argument('--nb_jobs', type=int, default=1000, help='ジョブ数')
    parser.add_argument('--seed', type=int, default=0, help='ジョブ生成の乱数シード')
    parser.add_argument('--nb_steps', type=int, default=1, help='ジョブ生成ステップ（JobGenerator）')
    parser.add_argument('--episodes', type=int, default=0, help='生成するエピソード数（JobGenerator）')
    parser.add_argument('--scaling_test', action='store_true', help='スケーリングテストを実行')
    parser.add_argument('--job_counts', nargs='+', type=int, default=[100, 500, 1000, 2000], 
                       help='スケーリングテストのジョブ数リスト')
    parser.add_argument('--use_heuristic', action='store_true', default=True, 
                       help='ヒューリスティックエージェントを使用')
    parser.add_argument('--profile', action='store_true', help='プロファイリングを実行（cProfile）')
    parser.add_argument('--profile_output', type=str, default='profile_output.prof',
                       help='プロファイリング結果の出力ファイル名')
    
    args = parser.parse_args()
    
    # プロファイリング実行
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            if args.scaling_test:
                # スケーリングテスト
                results = run_scaling_test(
                    args.job_counts,
                    args.use_heuristic,
                    seed=args.seed,
                    nb_steps=args.nb_steps,
                    nb_episodes=args.episodes
                )
            else:
                # 単一テスト
                results = run_environment_timing_test(
                    args.nb_jobs, 
                    use_heuristic=args.use_heuristic,
                    seed=args.seed,
                    nb_steps=args.nb_steps,
                    nb_episodes=args.episodes
                )
                print_timing_results(results)
        finally:
            profiler.disable()
            profiler.dump_stats(args.profile_output)
            print(f"\nプロファイリング結果を {args.profile_output} に保存しました")
            print(f"可視化するには: snakeviz {args.profile_output}")
            
            # フィルタリングされたプロファイルも作成（SnakeViz用に最適化）
            filtered_output = args.profile_output.replace('.prof', '_filtered.prof')
            stats = pstats.Stats(profiler)
            # プロジェクト内のファイルのみに絞る（外部ライブラリを除外）
            stats.strip_dirs()
            # 最小累積時間でフィルタリング（1ms未満を除外）
            stats.dump_stats(filtered_output)
            print(f"フィルタリング済みプロファイルを {filtered_output} に保存しました")
            
            # 簡単な統計情報を表示
            stats.sort_stats('cumulative')
            print("\n=== 累積時間順 Top 20 ===")
            stats.print_stats(20)
    else:
        # 通常実行
        if args.scaling_test:
            # スケーリングテスト
            results = run_scaling_test(
                args.job_counts,
                args.use_heuristic,
                seed=args.seed,
                nb_steps=args.nb_steps,
                nb_episodes=args.episodes
            )
        else:
            # 単一テスト
            results = run_environment_timing_test(
                args.nb_jobs, 
                use_heuristic=args.use_heuristic,
                seed=args.seed,
                nb_steps=args.nb_steps,
                nb_episodes=args.episodes
            )
            print_timing_results(results)

if __name__ == "__main__":
    main()
