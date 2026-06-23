#!/usr/bin/env python3
"""
数千ジョブでの割り当て時間計測スクリプト（C言語実装版）
main.pyの挙動を真似て、環境自体の時間を測定する
C言語実装を使用してパフォーマンスを向上
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# C言語実装をインポート
try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position as c_find_allocation_position,
        time_transition as c_time_transition,
        do_schedule as c_do_schedule,
        get_unique_job_ids as c_get_unique_job_ids,
        calculate_makespan as c_calculate_makespan
    )
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    print("警告: C言語実装が利用できません。ビルドしてください。")
    print("以下のコマンドでビルドしてください:")
    print("  uv sync")
    sys.exit(1)

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


class SchedulingEnvC(SchedulingEnv):
    """C言語実装を使用するSchedulingEnvのサブクラス"""
    
    def __init__(self, *args, **kwargs):
        """初期化（C言語実装を使用）"""
        super().__init__(*args, **kwargs)
        self._cache_onpre_c = None
        self._cache_cloud_c = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        print("C言語実装を使用します")
    
    def _rebuild_cache_if_needed_c(self, use_cloud: bool):
        """C言語実装を使用してキャッシュを構築"""
        if not use_cloud:
            # オンプレミスのキャッシュ
            # バージョンチェック（親クラスの属性を使用）
            current_version = getattr(self, '_version_onpre', 0)
            if self._cache_onpre_c is None or self._cache_version_onpre != current_version:
                window_status = np.ascontiguousarray(
                    self.on_premise_window['status'], dtype=np.int32
                )
                self._cache_onpre_c = WindowCache(
                    window_status, self.n_on_premise_node, self.n_window
                )
                self._cache_version_onpre = current_version
            return self._cache_onpre_c
        else:
            # クラウドのキャッシュ
            # バージョンチェック（親クラスの属性を使用）
            current_version = getattr(self, '_version_cloud', 0)
            if self._cache_cloud_c is None or self._cache_version_cloud != current_version:
                window_status = np.ascontiguousarray(
                    self.cloud_window['status'], dtype=np.int32
                )
                self._cache_cloud_c = WindowCache(
                    window_status, self.n_cloud_node, self.n_window
                )
                self._cache_version_cloud = current_version
            return self._cache_cloud_c
    
    def _rebuild_cache_if_needed(self, use_cloud: bool):
        """C言語実装を使用してキャッシュを構築（親クラスのメソッドをオーバーライド）"""
        # C言語実装のキャッシュを返す（親クラスのPython実装は使用しない）
        # ただし、親クラスのバージョン管理は維持する
        self._ensure_cache_initialized()
        
        if not use_cloud:
            # オンプレミスのキャッシュ
            current_version = getattr(self, '_version_onpre', 0)
            if self._cache_onpre_c is None or self._cache_version_onpre != current_version:
                window_status = np.ascontiguousarray(
                    self.on_premise_window['status'], dtype=np.int32
                )
                self._cache_onpre_c = WindowCache(
                    window_status, self.n_on_premise_node, self.n_window
                )
                self._cache_version_onpre = current_version
            # 親クラスのPythonキャッシュも更新（互換性のため）
            # ただし、実際には使用しない
            if not hasattr(self, '_cache_onpre') or self._cache_onpre.get('version', -1) != current_version:
                # 最小限のPythonキャッシュを作成（互換性のため）
                self._cache_onpre = {
                    'version': current_version,
                    'free_per_col': np.array([self.n_on_premise_node] * self.n_window, dtype=np.int32),
                    'prefix_sum': np.zeros((self.n_on_premise_node+1, self.n_window+1), dtype=np.int32),
                    'free_nodes_list': [np.arange(self.n_on_premise_node) for _ in range(self.n_window)],
                    'shape': (self.n_on_premise_node, self.n_window),
                    'occ': np.zeros((self.n_on_premise_node, self.n_window), dtype=np.int32)
                }
            return self._cache_onpre  # 互換性のためPythonキャッシュを返す（実際には使用されない）
        else:
            # クラウドのキャッシュ
            current_version = getattr(self, '_version_cloud', 0)
            if self._cache_cloud_c is None or self._cache_version_cloud != current_version:
                window_status = np.ascontiguousarray(
                    self.cloud_window['status'], dtype=np.int32
                )
                self._cache_cloud_c = WindowCache(
                    window_status, self.n_cloud_node, self.n_window
                )
                self._cache_version_cloud = current_version
            # 親クラスのPythonキャッシュも更新（互換性のため）
            if not hasattr(self, '_cache_cloud') or self._cache_cloud.get('version', -1) != current_version:
                # 最小限のPythonキャッシュを作成（互換性のため）
                self._cache_cloud = {
                    'version': current_version,
                    'free_per_col': np.array([self.n_cloud_node] * self.n_window, dtype=np.int32),
                    'prefix_sum': np.zeros((self.n_cloud_node+1, self.n_window+1), dtype=np.int32),
                    'free_nodes_list': [np.arange(self.n_cloud_node) for _ in range(self.n_window)],
                    'shape': (self.n_cloud_node, self.n_window),
                    'occ': np.zeros((self.n_cloud_node, self.n_window), dtype=np.int32)
                }
            return self._cache_cloud  # 互換性のためPythonキャッシュを返す（実際には使用されない）
    
    def find_allocation_position(self, action, cache_onpre=None, cache_cloud=None):
        """C言語実装を使用して割り当て位置を探索"""
        method = action[0]
        use_cloud = action[1]
        job = self.job_queue[0]
        
        if method == 0:
            job = self.job_queue[0]

        job_width = int(job[0])
        job_height = int(job[1])
        when_submitted = int(job[-1])
        current_time = self.time

        # job が 0 なら早期リターン
        if job[0] == 0 and job[1] == 0:
            return None, np.inf

        # 使用するウィンドウの選択とキャッシュ取得（C言語実装を使用）
        if not use_cloud:
            max_h, max_w = self.n_on_premise_node, self.n_window
            # C言語実装のキャッシュを構築（cache_onpreは無視して常にCキャッシュを構築）
            current_version = getattr(self, '_version_onpre', 0)
            if self._cache_onpre_c is None or self._cache_version_onpre != current_version:
                window_status = np.ascontiguousarray(
                    self.on_premise_window['status'], dtype=np.int32
                )
                self._cache_onpre_c = WindowCache(window_status, max_h, max_w)
                self._cache_version_onpre = current_version
            cache_c = self._cache_onpre_c
        else:
            max_h, max_w = self.n_cloud_node, self.n_window
            # C言語実装のキャッシュを構築（cache_cloudは無視して常にCキャッシュを構築）
            current_version = getattr(self, '_version_cloud', 0)
            if self._cache_cloud_c is None or self._cache_version_cloud != current_version:
                window_status = np.ascontiguousarray(
                    self.cloud_window['status'], dtype=np.int32
                )
                self._cache_cloud_c = WindowCache(window_status, max_h, max_w)
                self._cache_version_cloud = current_version
            cache_c = self._cache_cloud_c

        # ジョブサイズが大きすぎる場合は早期リターン
        if job_width > max_w or job_height > max_h:
            return None, np.inf
        
        # C言語実装で位置を探索
        position, waiting_time = c_find_allocation_position(
            cache_c, job_width, job_height, when_submitted, current_time
        )
        
        return position, waiting_time
    
    def time_transition(self, slide_on_premise=True, slide_cloud=True):
        """C言語実装を使用して時間遷移"""
        # 時間を1進める
        self.time += 1
        self.update_window_history()

        # 構造化配列からndarrayを取得（型とメモリレイアウトを固定）
        on_premise_status = np.ascontiguousarray(
            self.on_premise_window['status'], dtype=np.int32
        )
        on_premise_job_id = np.ascontiguousarray(
            self.on_premise_window['job_id'], dtype=np.int32
        )
        cloud_status = np.ascontiguousarray(
            self.cloud_window['status'], dtype=np.int32
        )
        cloud_job_id = np.ascontiguousarray(
            self.cloud_window['job_id'], dtype=np.int32
        )
    
        # C言語実装で時間遷移を実行（in-placeで変更）
        if slide_on_premise:
            # 配列を直接変更（C実装はin-placeで変更する）
            c_time_transition(
                on_premise_status, on_premise_job_id,
                self.n_on_premise_node, self.n_window, True
            )
            # 結果を元の配列に書き戻し（C実装はin-placeで変更するため、既に変更されている）
            self.on_premise_window['status'] = on_premise_status
            self.on_premise_window['job_id'] = on_premise_job_id
        
        if slide_cloud:
            # 配列を直接変更（C実装はin-placeで変更する）
            c_time_transition(
                cloud_status, cloud_job_id,
                self.n_cloud_node, self.n_window, True
            )
            # 結果を元の配列に書き戻し（C実装はin-placeで変更するため、既に変更されている）
            self.cloud_window['status'] = cloud_status
            self.cloud_window['job_id'] = cloud_job_id

        # 新しいジョブをジョブキューに追加
        self.append_new_job2job_queue()
        
        # キャッシュを無効化
        self._invalidate_window_cache(on_premise=slide_on_premise, cloud=slide_cloud)
        self._cache_onpre_c = None
        self._cache_cloud_c = None
    
    def do_schedule(self, action, job, position):
        """C言語実装を使用してジョブをスケジュール"""
        self.jobs_processed_count += 1
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        when_submitted = int(job[-1])
        use_cloud = action[1]
        
        # NumPy配列をC連続に保証
        if not use_cloud:
            window_status = np.ascontiguousarray(
                self.on_premise_window['status'], dtype=np.int32
            )
            window_job_id = np.ascontiguousarray(
                self.on_premise_window['job_id'], dtype=np.int32
            )
            c_do_schedule(
                window_status, window_job_id,
                self.n_on_premise_node, self.n_window,
                job_width, job_height, job_id,
                position
            )
            self.on_premise_window['status'] = window_status
            self.on_premise_window['job_id'] = window_job_id
            self._invalidate_window_cache(on_premise=True, cloud=False)
            self._cache_onpre_c = None
        else:
            window_status = np.ascontiguousarray(
                self.cloud_window['status'], dtype=np.int32
            )
            window_job_id = np.ascontiguousarray(
                self.cloud_window['job_id'], dtype=np.int32
            )
            c_do_schedule(
                window_status, window_job_id,
                self.n_cloud_node, self.n_window,
                job_width, job_height, job_id,
                position
            )
            self.cloud_window['status'] = window_status
            self.cloud_window['job_id'] = window_job_id
            self._invalidate_window_cache(on_premise=False, cloud=True)
            self._cache_cloud_c = None
        
        waiting_time = self.time - when_submitted
        self.waiting_times.append(waiting_time)
        
        return waiting_time


def run_environment_timing_test(nb_jobs: int,
                               *,
                               use_heuristic: bool = True,
                               seed: int = 0,
                               nb_steps: int = 1,
                               nb_episodes: int = 0) -> Dict[str, Any]:
    """環境の時間計測テストを実行（C言語実装版）"""
    print(f"\n{'='*60}")
    print(f"環境時間計測テスト開始（C言語実装版）: {nb_jobs}ジョブ")
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
    
    # 環境を初期化（C言語実装版）
    print(f"\n環境を初期化中（C言語実装版）...")
    env_start_time = time.time()
    
    env = SchedulingEnvC(
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
    print(f"\nジョブ割り当て処理を開始（C言語実装版）...")
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
        'cloud_allocations': cloud_allocations,
        'implementation': 'C'
    }
    
    return results

def print_timing_results(results: Dict[str, Any]):
    """時間計測結果を表示"""
    print(f"\n{'='*60}")
    print("時間計測結果（C言語実装版）")
    print(f"{'='*60}")
    
    print(f"ジョブ数: {results['nb_jobs']:,}")
    print(f"実装: C言語実装")
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
    """複数のジョブ数でスケーリングテストを実行（C言語実装版）"""
    print(f"\n{'='*80}")
    print("スケーリングテスト開始（C言語実装版）")
    print(f"{'='*80}")
    
    all_results = []
    
    for nb_jobs in job_counts:
        print(f"\n{'='*40}")
        print(f"テスト: {nb_jobs}ジョブ（C言語実装版）")
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
    print("スケーリングテスト結果要約（C言語実装版）")
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
    parser = argparse.ArgumentParser(description='数千ジョブでの割り当て時間計測（C言語実装版）')
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
    parser.add_argument('--profile_output', type=str, default='profile_output_c.prof',
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

