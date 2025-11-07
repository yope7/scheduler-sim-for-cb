import numpy as np
import itertools
import ray
import time
from typing import List, Dict
from morl_baselines.common.pareto import get_non_dominated_inds

# パレートフロント計算用の関数（最小化問題用）
def get_non_dominated_inds_minimize(costs):
    """最小化問題用の非支配解インデックスを取得"""
    # コストをマイナスにして最大化問題として扱う
    return get_non_dominated_inds(-costs)

@ray.remote
def evaluate_action_set_pure_step_time(action_sets: List[List[int]], env_config: Dict) -> List[Dict]:
    """純粋なstep処理時間のみを計測する関数（環境作成・初期化時間除外）"""
    import numpy as np
    from src.envs.scheduling_env import SchedulingEnv
    import time
    
    results = []
    
    # 環境インスタンスを1つ作成して再利用
    env = SchedulingEnv(
        env_config['params']['max_step'],
        env_config['params']['n_window'],
        env_config['params']['n_on_premise_node'],
        env_config['params']['n_cloud_node'],
        env_config['params']['n_job_queue_obs'],
        env_config['params']['n_job_queue_bck'],
        env_config['params']['weight_wt'],
        env_config['params']['weight_cost'],
        env_config['params']['penalty_not_allocate'],
        env_config['params']['penalty_invalid_action'],
        env_config['params']['jobs_set'],
        None,
        flag=env_config['params']['flag']
    )
    
    for i, action_set in enumerate(action_sets):
        # 環境をリセット（初期化時間は除外）
        obs = env.reset()
        
        # NSGA-IIと同じ戦略的実行方式に変更
        done = False
        nb_jobs = len(action_set)
        processed_jobs = 0
        total_reward = [0, 0]
        
        # 純粋なstep処理時間のみを計測
        step_start_time = time.time()
        
        # 各ジョブに対してアクションを実行
        action_index = 0
        while not done and processed_jobs < nb_jobs and action_index < len(action_set):
            # アクションセットから次のアクションを取得
            action = action_set[action_index]
            
            obs, reward, scheduled, wt_step, done = env.step(action)
            
            # スケジュールされた場合のみジョブカウントを増加
            if scheduled:
                processed_jobs += 1
            
            # 累積報酬を計算
            if isinstance(reward, (list, np.ndarray)) and len(reward) >= 2:
                total_reward[0] += reward[0]
                total_reward[1] += reward[1]
            else:
                # スカラー報酬の場合は適切に分配
                total_reward[0] += float(reward)
                total_reward[1] += 0
            
            # 次のアクションインデックスを増加
            action_index += 1
            
            if done:
                break
        
        # 環境の最終化
        if hasattr(env, 'finalize_window_history'):
            env.finalize_window_history()
        
        # NSGA-IIと同じ集計値を使用
        cost, _, avg_waiting_time = env.calc_objective_values()

        # 純粋なstep処理時間
        pure_step_time = time.time() - step_start_time
        print(f"アクションセット {i+1} の純粋step時間: {pure_step_time:.4f}秒")
    
        # NSGA-IIと全く同じ形式で結果を保存
        results.append({
            'results': [cost, avg_waiting_time],  # [コスト, 平均待ち時間] - NSGA-IIと同じ
            'reward_summary': total_reward,  # 累積報酬を保存
            'epi_summary': [processed_jobs, action_index],  # 処理済みジョブ数とアクション数
            'execution_time': pure_step_time  # 純粋なstep処理時間
        })
    
    return results

@ray.remote
def evaluate_action_set_batch(action_sets: List[List[int]], env_config: Dict) -> List[Dict]:
    """バッチでアクションセットを評価する関数（Ray remote function）"""
    import numpy as np
    from src.envs.scheduling_env import SchedulingEnv
    import time
    
    results = []
    
    for i, action_set in enumerate(action_sets):
        
        # 環境の初期化
        env = SchedulingEnv(
            env_config['params']['max_step'],
            env_config['params']['n_window'],
            env_config['params']['n_on_premise_node'],
            env_config['params']['n_cloud_node'],
            env_config['params']['n_job_queue_obs'],
            env_config['params']['n_job_queue_bck'],
            env_config['params']['weight_wt'],
            env_config['params']['weight_cost'],
            env_config['params']['penalty_not_allocate'],
            env_config['params']['penalty_invalid_action'],
            env_config['params']['jobs_set'],
            None,
            flag=env_config['params']['flag']
        )
        
        # 環境をリセット
        obs = env.reset()
        
        # NSGA-IIと同じ戦略的実行方式に変更
        done = False
        nb_jobs = len(action_set)
        processed_jobs = 0
        total_reward = [0, 0]
        
        # 各ジョブに対してアクションを実行
        action_index = 0
        start_time = time.time()
        while not done and processed_jobs < nb_jobs and action_index < len(action_set):
            # アクションセットから次のアクションを取得
            action = action_set[action_index]
            
            
            obs, reward, scheduled, wt_step, done = env.step(action)
            
            # スケジュールされた場合のみジョブカウントを増加
            if scheduled:
                processed_jobs += 1
            
            # 累積報酬を計算
            if isinstance(reward, (list, np.ndarray)) and len(reward) >= 2:
                total_reward[0] += reward[0]
                total_reward[1] += reward[1]
            else:
                # スカラー報酬の場合は適切に分配
                total_reward[0] += float(reward)
                total_reward[1] += 0
            
            # 次のアクションインデックスを増加
            action_index += 1
            
            if done:
                break
        
        # 環境の最終化
        if hasattr(env, 'finalize_window_history'):
            env.finalize_window_history()
        
        # NSGA-IIと同じ集計値を使用
        cost, _, avg_waiting_time = env.calc_objective_values()

        execution_time = time.time() - start_time
        print(f"評価時間: {execution_time:.4f}秒")
    
        
        # NSGA-IIと全く同じ形式で結果を保存
        results.append({
            'results': [cost, avg_waiting_time],  # [コスト, 平均待ち時間] - NSGA-IIと同じ
            'reward_summary': total_reward,  # 累積報酬を保存
            'epi_summary': [processed_jobs, action_index],  # 処理済みジョブ数とアクション数
            'execution_time': execution_time
        })
    
    return results

class ExhaustiveSearchAgentDistributed:
    def __init__(self, num_workers: int = None):
        """
        Args:
            num_workers: 分散ワーカー数。Noneの場合は利用可能な全CPUコアを使用
        """
        self.num_workers = num_workers
        
        # Rayの初期化（シンプルな設定）
        if not ray.is_initialized():
            # 既存クラスターに接続する場合は num_cpus を指定しない
            ray.init(
                ignore_reinit_error=True,
                local_mode=False
            )
        
        # メトリクスの初期化
        self.total_tasks = 0
        self.completed_tasks = 0
        self.execution_times = []
        
        # サンプリング設定
        self.max_jobs_full_search = 20
        self.max_samples = 2 ** 16  # 32,768 samples
    
    def generate_uniform_samples(self, nb_jobs: int, num_samples: int):
        """指定されたジョブ数とサンプル数で多様性のあるサンプリングを実行"""
        print(f"Generating {num_samples} diverse samples from {2**nb_jobs} total combinations")
        
        # 総数から均等にサンプルするためのインデックスを生成
        total_combinations = 2 ** nb_jobs
        
        action_sets = []
        
        # 1. 境界値を確実に含める（全0、全1、および中間値）
        action_sets.append([0] * nb_jobs)  # 全オンプレミス
        action_sets.append([1] * nb_jobs)  # 全クラウド
        
        # 2. クラウド使用率が段階的に増える解を追加
        cloud_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for ratio in cloud_ratios:
            if len(action_sets) < num_samples:
                # 指定された比率でクラウドを使用
                num_cloud_jobs = int(nb_jobs * ratio)
                action_set = [0] * nb_jobs
                # ランダムにクラウド使用ジョブを選択
                np.random.seed(int(ratio * 1000))  # 再現性のため
                cloud_indices = np.random.choice(nb_jobs, num_cloud_jobs, replace=False)
                for idx in cloud_indices:
                    action_set[idx] = 1
                action_sets.append(action_set)
        
        # 3. パターンベースのサンプリング
        patterns = [
            # 交互パターン
            [i % 2 for i in range(nb_jobs)],
            [(i + 1) % 2 for i in range(nb_jobs)],
            # ブロックパターン
            [1 if i < nb_jobs // 2 else 0 for i in range(nb_jobs)],
            [0 if i < nb_jobs // 2 else 1 for i in range(nb_jobs)],
            # 4分割パターン
            [1 if i < nb_jobs // 4 or (nb_jobs // 2 <= i < 3 * nb_jobs // 4) else 0 for i in range(nb_jobs)],
        ]
        
        for pattern in patterns:
            if len(action_sets) < num_samples:
                action_sets.append(pattern[:nb_jobs])  # nb_jobsに切り詰め
        
        # 4. 残りをランダムサンプリングで補完
        remaining_samples = num_samples - len(action_sets)
        if remaining_samples > 0:
            # より多様性のあるランダムサンプリング
            np.random.seed(42)  # 再現性のため
            
            # 異なる確率分布を使用してランダムサンプリング
            prob_distributions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            for i in range(remaining_samples):
                if i < len(prob_distributions):
                    # 指定確率でクラウドを選択
                    prob = prob_distributions[i % len(prob_distributions)]
                    action_set = (np.random.random(nb_jobs) < prob).astype(int).tolist()
                else:
                    # 完全ランダム
                    action_set = np.random.randint(0, 2, nb_jobs).tolist()
                
                action_sets.append(action_set)
        
        # 重複除去
        unique_action_sets = []
        seen = set()
        for action_set in action_sets:
            action_tuple = tuple(action_set)
            if action_tuple not in seen:
                seen.add(action_tuple)
                unique_action_sets.append(action_set)
        
        action_sets = unique_action_sets[:num_samples]
        
        print(f"Generated {len(action_sets)} unique action sets using diverse sampling")
        
        # 多様性の統計を表示
        if action_sets:
            print(f"=== サンプリング多様性の分析 ===")
            cloud_usage_ratios = [sum(action_set) / len(action_set) for action_set in action_sets]
            print(f"クラウド使用率の範囲: {min(cloud_usage_ratios):.3f} - {max(cloud_usage_ratios):.3f}")
            print(f"クラウド使用率の平均: {np.mean(cloud_usage_ratios):.3f}")
            print(f"クラウド使用率の標準偏差: {np.std(cloud_usage_ratios):.3f}")
            
            # 期待されるジョブ数との整合性確認
            all_correct_length = all(len(action_set) == nb_jobs for action_set in action_sets)
            print(f"全アクションセットが正しい長さ: {all_correct_length}")
            
            if not all_correct_length:
                wrong_lengths = [(i, len(action_set)) for i, action_set in enumerate(action_sets) if len(action_set) != nb_jobs]
                print(f"間違った長さのサンプル: {wrong_lengths[:10]}")
        
        return action_sets
    
    def run_exhaustive_search_pure_step_time(self, env, nb_jobs: int):
        """
        純粋なstep処理時間のみを計測する全探索（環境作成・初期化時間除外）
        
        Args:
            env: 初期化済みのSchedulingEnv環境
            nb_jobs: ジョブの数
        """
        # サンプリング方式の決定
        if nb_jobs >= self.max_jobs_full_search:
            print(f"Job count ({nb_jobs}) >= {self.max_jobs_full_search}. Using uniform sampling.")
            all_action_sets = self.generate_uniform_samples(nb_jobs, self.max_samples)
            total_sets = len(all_action_sets)
            is_sampling = True
        else:
            print(f"Job count ({nb_jobs}) < {self.max_jobs_full_search}. Performing full exhaustive search.")
            all_action_sets = list(itertools.product([0, 1], repeat=nb_jobs))
            total_sets = len(all_action_sets)
            is_sampling = False
        
        print(f"Total action sets to evaluate: {total_sets}")
        if is_sampling:
            print(f"Sampling rate: {total_sets / (2**nb_jobs) * 100:.4f}% of all possible combinations")

        # 環境の設定を準備
        env_config = {
            'params': {
                'max_step': env.max_step,
                'n_window': env.n_window,
                'n_on_premise_node': env.n_on_premise_node,
                'n_cloud_node': env.n_cloud_node,
                'n_job_queue_obs': env.n_job_queue_obs,
                'n_job_queue_bck': env.n_job_queue_bck,
                'weight_wt': env.weight_wt,
                'weight_cost': env.weight_cost,
                'penalty_not_allocate': env.penalty_not_allocate,
                'penalty_invalid_action': env.penalty_invalid_action,
                'jobs_set': env.jobs_set,
                'flag': 0
            }
        }

        # メトリクスの更新
        self.total_tasks = total_sets

        # ワーカー数の設定
        if self.num_workers is not None:
            num_workers = self.num_workers
            print(f"Using specified number of workers: {num_workers}")
        else:
            num_workers = int(ray.available_resources()['CPU'])
            print(f"Using available CPU cores: {num_workers}")
        
        # バッチサイズの設定
        batch_size = max(100, min(10000, total_sets // (num_workers * 10)))
        
        print(f"Using {num_workers} workers with batch size: {batch_size}")
        print(f"Estimated total batches: {(total_sets + batch_size - 1) // batch_size}")

        # 結果の収集と進捗表示
        all_results = []
        all_reward_summary = []
        all_epi_summary = []
        all_pure_step_times = []
        
        # バッチごとに処理
        futures = []
        completed = 0
        start_time = time.time()
        
        # アクションセットをバッチに分割して並列実行（純粋step時間計測版）
        for i in range(0, total_sets, batch_size):
            batch_end = min(i + batch_size, total_sets)
            batch_action_sets = all_action_sets[i:batch_end]
            
            # バッチを並列実行（純粋step時間計測版）
            future = evaluate_action_set_pure_step_time.remote(batch_action_sets, env_config)
            futures.append(future)
            
            # 一定数のバッチが溜まったら結果を取得（最適化版：複数の完了タスクを一度に取得）
            if len(futures) >= num_workers * 2:
                # 複数の完了タスクを一度に待つ
                num_returns = min(num_workers * 2, len(futures))
                done_id, remaining_futures = ray.wait(futures, num_returns=num_returns, timeout=30.0)
                
                if done_id:
                    # 完了した結果を並列に取得
                    batch_results_list = ray.get(done_id)
                    
                    # 各バッチの結果を処理
                    for batch_results in batch_results_list:
                        for result in batch_results:
                            # メトリクスの更新
                            self.completed_tasks += 1
                            self.execution_times.append(result['execution_time'])
                            
                            # 結果の保存
                            all_results.append(result['results'])
                            all_reward_summary.append(result['reward_summary'])
                            all_epi_summary.append(result['epi_summary'])
                            all_pure_step_times.append(result['execution_time'])
                            
                            # 進捗表示
                            completed += 1
                            if completed % 100 == 0 or completed == total_sets:
                                elapsed_time = time.time() - start_time
                                progress = (completed / total_sets) * 100
                                estimated_remaining = (elapsed_time / completed) * (total_sets - completed) if completed > 0 else 0
                                print(f"\rProgress: {progress:.1f}% ({completed}/{total_sets}) "
                                      f"Elapsed: {elapsed_time:.1f}s "
                                      f"Remaining: {estimated_remaining:.1f}s "
                                      f"pure_step_time:{result['execution_time']:.4f}s", end="")
                    
                    # futuresリストを更新
                    futures = remaining_futures
        
        # 残りのバッチを処理（最適化版：複数の完了タスクを一度に取得）
        print(f"\nProcessing remaining {len(futures)} batches...")
        max_concurrent = min(num_workers * 2, len(futures)) if len(futures) > 0 else 1
        
        while futures:
            # 複数の完了タスクを一度に待つ
            num_returns = min(max_concurrent, len(futures))
            done_id, remaining_futures = ray.wait(futures, num_returns=num_returns, timeout=30.0)
            
            if done_id:
                # 完了した結果を並列に取得
                batch_results_list = ray.get(done_id)
                
                # 各バッチの結果を処理
                for batch_results in batch_results_list:
                    for result in batch_results:
                        # メトリクスの更新
                        self.completed_tasks += 1
                        self.execution_times.append(result['execution_time'])
                        
                        # 結果の保存
                        all_results.append(result['results'])
                        all_reward_summary.append(result['reward_summary'])
                        all_epi_summary.append(result['epi_summary'])
                        all_pure_step_times.append(result['execution_time'])
                        
                        # 進捗表示
                        completed += 1
                        if completed % 100 == 0 or completed == total_sets:
                            elapsed_time = time.time() - start_time
                            progress = (completed / total_sets) * 100
                            estimated_remaining = (elapsed_time / completed) * (total_sets - completed) if completed > 0 else 0
                            print(f"\rProgress: {progress:.1f}% ({completed}/{total_sets}) "
                                  f"Elapsed: {elapsed_time:.1f}s "
                                  f"Remaining: {estimated_remaining:.1f}s "
                                  f"pure_step_time:{result['execution_time']:.4f}s", end="")
                
                # futuresリストを更新
                futures = remaining_futures
            else:
                # タイムアウト時はより小さいバッチで再試行
                if len(futures) > 0:
                    max_concurrent = max(1, max_concurrent // 2)
                else:
                    break
        
        print(f"\nAll tasks completed! Total processed: {completed}")

        # 純粋step時間の統計を表示
        if all_pure_step_times:
            print(f"\n=== 純粋step時間の統計 ===")
            print(f"平均step時間: {np.mean(all_pure_step_times):.4f}秒")
            print(f"最小step時間: {min(all_pure_step_times):.4f}秒")
            print(f"最大step時間: {max(all_pure_step_times):.4f}秒")
            print(f"標準偏差: {np.std(all_pure_step_times):.4f}秒")

        # パレートフロントの計算
        print(f"\nCalculating Pareto front from {len(all_results)} results...")
        non_dominated_inds = get_non_dominated_inds_minimize(np.array(all_results))
        pareto_front = np.array(all_results)[non_dominated_inds]
        print(f"Pareto front contains {len(pareto_front)} non-dominated solutions")
        
        # サンプリング情報も含めて結果を返す
        return {
            'results': all_results,
            'pareto_front': pareto_front,
            'reward_summary': all_reward_summary,
            'epi_summary': all_epi_summary,
            'pure_step_times': all_pure_step_times,  # 純粋step時間を追加
            'is_sampling': is_sampling,
            'total_evaluated': total_sets,
            'total_possible': 2 ** nb_jobs if nb_jobs < 30 else float('inf'),
            'sampling_rate': total_sets / (2**nb_jobs) if nb_jobs < 30 else 0,
            'non_dominated_indices': non_dominated_inds,
            'statistics': {
                'total_solutions': len(all_results),
                'pareto_solutions': len(pareto_front),
                'pareto_rate': len(pareto_front)/len(all_results)*100,
                'avg_pure_step_time': np.mean(all_pure_step_times) if all_pure_step_times else 0,
                'min_pure_step_time': min(all_pure_step_times) if all_pure_step_times else 0,
                'max_pure_step_time': max(all_pure_step_times) if all_pure_step_times else 0
            }
        }

    def run_exhaustive_search(self, env, nb_jobs: int):
        """
        全探索による最適なスケジューリングの探索（Rayによる並列化）
        ジョブ数が20以上の場合は2^15個のサンプルを均等にサンプリング
        
        Args:
            env: 初期化済みのSchedulingEnv環境
            nb_jobs: ジョブの数
        """
        # サンプリング方式の決定
        if nb_jobs >= self.max_jobs_full_search:
            # 20以上の場合はサンプリング実行
            print(f"Job count ({nb_jobs}) >= {self.max_jobs_full_search}. Using uniform sampling.")
            all_action_sets = self.generate_uniform_samples(nb_jobs, self.max_samples)
            total_sets = len(all_action_sets)
            is_sampling = True
        else:
            # 20未満の場合は全探索
            print(f"Job count ({nb_jobs}) < {self.max_jobs_full_search}. Performing full exhaustive search.")
            all_action_sets = list(itertools.product([0, 1], repeat=nb_jobs))
            total_sets = len(all_action_sets)
            is_sampling = False
        
        print(f"Total action sets to evaluate: {total_sets}")
        if is_sampling:
            print(f"Sampling rate: {total_sets / (2**nb_jobs) * 100:.4f}% of all possible combinations")

        # 環境の設定を準備
        env_config = {
            'params': {
                'max_step': env.max_step,
                'n_window': env.n_window,
                'n_on_premise_node': env.n_on_premise_node,
                'n_cloud_node': env.n_cloud_node,
                'n_job_queue_obs': env.n_job_queue_obs,
                'n_job_queue_bck': env.n_job_queue_bck,
                'weight_wt': env.weight_wt,
                'weight_cost': env.weight_cost,
                'penalty_not_allocate': env.penalty_not_allocate,
                'penalty_invalid_action': env.penalty_invalid_action,
                'jobs_set': env.jobs_set,
                'flag': 0
            }
        }

        # メトリクスの更新
        self.total_tasks = total_sets

        # ワーカー数の設定
        if self.num_workers is not None:
            num_workers = self.num_workers
            print(f"Using specified number of workers: {num_workers}")
        else:
            num_workers = int(ray.available_resources()['CPU'])
            print(f"Using available CPU cores: {num_workers}")
        
        # バッチサイズの設定（サンプリング時は効率化）
        if is_sampling:
            batch_size = max(100, min(10000, total_sets // (num_workers * 10)))
        else:
            batch_size = max(100, min(1000, total_sets // (num_workers * 10)))
        
        print(f"Using {num_workers} workers with batch size: {batch_size}")
        print(f"Estimated total batches: {(total_sets + batch_size - 1) // batch_size}")

        # 結果の収集と進捗表示
        all_results = []
        all_reward_summary = []
        all_epi_summary = []
        
        # バッチごとに処理
        futures = []
        completed = 0
        start_time = time.time()
        
        # アクションセットをバッチに分割して並列実行
        for i in range(0, total_sets, batch_size):
            batch_end = min(i + batch_size, total_sets)
            batch_action_sets = all_action_sets[i:batch_end]
            
            # バッチを並列実行
            future = evaluate_action_set_batch.remote(batch_action_sets, env_config)
            futures.append(future)
            
            # 一定数のバッチが溜まったら結果を取得（最適化版：複数の完了タスクを一度に取得）
            if len(futures) >= num_workers * 2:
                # 複数の完了タスクを一度に待つ
                num_returns = min(num_workers * 2, len(futures))
                done_id, remaining_futures = ray.wait(futures, num_returns=num_returns, timeout=30.0)
                
                if done_id:
                    # 完了した結果を並列に取得
                    batch_results_list = ray.get(done_id)
                    
                    # 各バッチの結果を処理
                    for batch_results in batch_results_list:
                        for result in batch_results:
                            # メトリクスの更新
                            self.completed_tasks += 1
                            self.execution_times.append(result['execution_time'])
                            
                            # 結果の保存
                            all_results.append(result['results'])
                            all_reward_summary.append(result['reward_summary'])
                            all_epi_summary.append(result['epi_summary'])
                            
                            # 進捗表示
                            completed += 1
                            if completed % 100 == 0 or completed == total_sets:
                                elapsed_time = time.time() - start_time
                                progress = (completed / total_sets) * 100
                                estimated_remaining = (elapsed_time / completed) * (total_sets - completed) if completed > 0 else 0
                                print(f"\rProgress: {progress:.1f}% ({completed}/{total_sets}) "
                                      f"Elapsed: {elapsed_time:.1f}s "
                                      f"Remaining: {estimated_remaining:.1f}s "
                                      f"value_cost:{result['results'][0]:.2f}, "
                                      f"value_wt:{result['results'][1]:.2f}", end="")
                    
                    # futuresリストを更新
                    futures = remaining_futures
        
        # 残りのバッチを処理（最適化版：複数の完了タスクを一度に取得）
        print(f"\nProcessing remaining {len(futures)} batches...")
        max_concurrent = min(num_workers * 2, len(futures)) if len(futures) > 0 else 1
        
        while futures:
            # 複数の完了タスクを一度に待つ
            num_returns = min(max_concurrent, len(futures))
            done_id, remaining_futures = ray.wait(futures, num_returns=num_returns, timeout=30.0)
            
            if done_id:
                # 完了した結果を並列に取得
                batch_results_list = ray.get(done_id)
                
                # 各バッチの結果を処理
                for batch_results in batch_results_list:
                    for result in batch_results:
                        # メトリクスの更新
                        self.completed_tasks += 1
                        self.execution_times.append(result['execution_time'])
                        
                        # 結果の保存
                        all_results.append(result['results'])
                        all_reward_summary.append(result['reward_summary'])
                        all_epi_summary.append(result['epi_summary'])
                        
                        # 進捗表示
                        completed += 1
                        if completed % 100 == 0 or completed == total_sets:
                            elapsed_time = time.time() - start_time
                            progress = (completed / total_sets) * 100
                            estimated_remaining = (elapsed_time / completed) * (total_sets - completed) if completed > 0 else 0
                            print(f"\rProgress: {progress:.1f}% ({completed}/{total_sets}) "
                                  f"Elapsed: {elapsed_time:.1f}s "
                                  f"Remaining: {estimated_remaining:.1f}s "
                                  f"value_cost:{result['results'][0]:.2f}, "
                                  f"value_wt:{result['results'][1]:.2f}", end="")
                
                # futuresリストを更新
                futures = remaining_futures
            else:
                # タイムアウト時はより小さいバッチで再試行
                if len(futures) > 0:
                    max_concurrent = max(1, max_concurrent // 2)
                else:
                    break
        
        print(f"\nAll tasks completed! Total processed: {completed}")

        # パレートフロントの計算
        print(f"\nCalculating Pareto front from {len(all_results)} results...")
        non_dominated_inds = get_non_dominated_inds_minimize(np.array(all_results))
        pareto_front = np.array(all_results)[non_dominated_inds]
        print(f"Pareto front contains {len(pareto_front)} non-dominated solutions")
        
        # 全結果を標準出力に表示（追加）
        print(f"\n=== 全結果の詳細 ===")
        print(f"{'No.':<6} {'コスト':<12} {'平均待ち時間':<15} {'パレート最適':<12}")
        print(f"{'-'*60}")
        
        for i, result in enumerate(all_results):
            cost, avg_waiting_time = result
            is_pareto = "Yes" if i in non_dominated_inds else "No"
            print(f"{i+1:<6} {cost:<12.2f} {avg_waiting_time:<15.2f} {is_pareto:<12}")
        
        # パレートフロントの詳細表示
        print(f"\n=== パレート最適解の詳細 ===")
        print(f"{'No.':<6} {'コスト':<12} {'平均待ち時間':<15}")
        print(f"{'-'*60}")
        
        for i, pareto_solution in enumerate(pareto_front):
            cost, avg_waiting_time = pareto_solution
            print(f"{i+1:<6} {cost:<12.2f} {avg_waiting_time:<15.2f}")
        
        # 統計情報の表示
        print(f"\n=== 統計情報 ===")
        costs = [r[0] for r in all_results]
        waiting_times = [r[1] for r in all_results]
        
        print(f"総解数: {len(all_results)}")
        print(f"パレート最適解数: {len(pareto_front)}")
        print(f"パレート率: {len(pareto_front)/len(all_results)*100:.2f}%")
        print(f"")
        print(f"コスト範囲:")
        print(f"  最小: {min(costs):.2f}")
        print(f"  最大: {max(costs):.2f}")
        print(f"  平均: {np.mean(costs):.2f}")
        print(f"")
        print(f"平均待ち時間範囲:")
        print(f"  最小: {min(waiting_times):.2f}")
        print(f"  最大: {max(waiting_times):.2f}")
        print(f"  平均: {np.mean(waiting_times):.2f}")
        
        # サンプリング情報も含めて結果を返す
        return {
            'results': all_results,
            'pareto_front': pareto_front,
            'reward_summary': all_reward_summary,
            'epi_summary': all_epi_summary,
            'is_sampling': is_sampling,
            'total_evaluated': total_sets,
            'total_possible': 2 ** nb_jobs if nb_jobs < 30 else float('inf'),
            'sampling_rate': total_sets / (2**nb_jobs) if nb_jobs < 30 else 0,
            'non_dominated_indices': non_dominated_inds,  # 追加: 非支配解のインデックス
            'statistics': {  # 追加: 統計情報
                'total_solutions': len(all_results),
                'pareto_solutions': len(pareto_front),
                'pareto_rate': len(pareto_front)/len(all_results)*100,
                'cost_range': (min(costs), max(costs), np.mean(costs)),
                'waiting_time_range': (min(waiting_times), max(waiting_times), np.mean(waiting_times))
            }
        } 