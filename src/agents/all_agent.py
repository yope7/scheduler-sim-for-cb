import numpy as np
import itertools
from morl_baselines.common.pareto import get_non_dominated_inds_minimize
from numba import jit
class ExhaustiveSearchAgent:
    def __init__(self):
        pass
    def run_exhaustive_search(self, env, nb_jobs: int):
        """
        全探索による最適なスケジューリングの探索
        
        Args:
            env: 初期化済みのSchedulingEnv環境
            nb_jobs: ジョブの数
        """
        # 各ジョブに対するアクション（0 または 1）の全組み合わせを生成
        all_action_sets = list(itertools.product([0, 1], repeat=nb_jobs))
        print(f"Total action sets: {len(all_action_sets)}")

        results = []
        scheduling_details = []
        epi_summary = []
        reward_summary = []
        
        # 各組み合わせごとにエピソードをシミュレーション
        for i, action_set in enumerate(all_action_sets):
            print(f"Processing action set {i+1}/{len(all_action_sets)}: {action_set}")
            
            # 環境のリセット
            obs = env.reset()
            done = False
            total_reward = [0,0]
            step = 0
            wt_sum = 0
            scheduled = False

            # エピソードの実行
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
            on_premise_map, cloud_map = env.get_windows()
            epi_summary.append([waiting_time, cost])
            reward_summary.append([total_reward[0], total_reward[1]])
            value_cost, value_wt = env.calc_objective_values()
            results.append([value_cost, value_wt])

        # パレートフロントの計算
        non_dominated_inds = get_non_dominated_inds_minimize(np.array(results))
        pareto_front = np.array(results)[non_dominated_inds]
        
        return {
            'results': results,
            'pareto_front': pareto_front,
            'reward_summary': reward_summary,
            'epi_summary': epi_summary
        } 