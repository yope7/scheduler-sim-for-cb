import numpy as np
import itertools
from src.agents.pcn_agent import get_non_dominated_inds_minimize
from numba import jit
from typing import List, Dict, Tuple
from src.envs.scheduling_env import SchedulingEnv
import time

def evaluate_action_set_batch(action_sets: List[List[int]], env_config: Dict) -> List[Dict]:
    """
    複数のアクションセットをバッチで評価する関数
    
    Args:
        action_sets: 評価するアクションセットのリスト
        env_config: 環境の設定パラメータ
    
    Returns:
        評価結果のリスト
    """
    # 環境のコピーを作成（バッチ全体で1回だけ）
    env = SchedulingEnv(**env_config['params'])
    results = []
    
    for action_set in action_sets:
        start_time = time.time()
        
        # エピソードの実行
        obs = env.reset()
        done = False
        total_reward = [0, 0]
        step = 0
        wt_sum = 0
        scheduled = False

        while not done:
            action = action_set[step]
            obs, reward, scheduled, wt_step, done = env.step(action)
            if scheduled:
                step += 1
            if done:
                env.finalize_window_history()
            total_reward[0] += reward[0]
            total_reward[1] += reward[1]
            wt_sum += wt_step

        # 結果の収集
        waiting_time, cost = env.get_episode_metrics()
        value_cost, value_wt = env.calc_objective_values()
        
        execution_time = time.time() - start_time
        
        results.append({
            'results': [value_cost, value_wt],
            'reward_summary': [total_reward[0], total_reward[1]],
            'epi_summary': [waiting_time, cost],
            'execution_time': execution_time
        })
    
    return results

class ExhaustiveSearchAgent:
    def __init__(self):
        # メトリクスの初期化
        self.total_tasks = 0
        self.completed_tasks = 0
        self.execution_times = []
    
    def run_exhaustive_search(self, env, nb_jobs: int):
        """
        全探索による最適なスケジューリングの探索（逐次実行）
        
        Args:
            env: 初期化済みのSchedulingEnv環境
            nb_jobs: ジョブの数
        """
        # 全ての可能なアクションセットを生成（0と1の組み合わせ）
        all_action_sets = list(itertools.product([0, 1], repeat=nb_jobs))
        total_sets = len(all_action_sets)
        print(f"Total action sets: {total_sets}")

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

        # バッチサイズの設定
        batch_size = 1  # 1つずつ処理
        print(f"Using batch size: {batch_size}")

        # アクションセットをバッチに分割
        action_batches = [all_action_sets[i:i + batch_size] for i in range(0, len(all_action_sets), batch_size)]
        
        # 逐次評価の実行
        all_results = []
        all_reward_summary = []
        all_epi_summary = []
        
        completed = 0
        start_time = time.time()
        
        for batch in action_batches:
            batch_results = evaluate_action_set_batch(batch, env_config)
            
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
                    estimated_remaining = (elapsed_time / completed) * (total_sets - completed)
                    print(f"\rProgress: {progress:.1f}% ({completed}/{total_sets}) "
                          f"Elapsed: {elapsed_time:.1f}s "
                          f"Remaining: {estimated_remaining:.1f}s "
                          f"value_cost:{result['results'][0]:.2f}, "
                          f"value_wt:{result['results'][1]:.2f}", end="")
        
        print("\nAll tasks completed!")

        # パレートフロントの計算
        non_dominated_inds = get_non_dominated_inds_minimize(np.array(all_results))
        pareto_front = np.array(all_results)[non_dominated_inds]
        
        return {
            'results': all_results,
            'pareto_front': pareto_front,
            'reward_summary': all_reward_summary,
            'epi_summary': all_epi_summary
        } 