#!/usr/bin/env python3
"""
純粋step時間計測機能のテストスクリプト
"""

import sys
import os
import time
import numpy as np

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.envs.scheduling_env import SchedulingEnv
from src.agents.nsga2_agent import NSGA2Agent
from src.agents.all_agent_distributed import ExhaustiveSearchAgentDistributed

def create_test_env():
    """テスト用の環境を作成"""
    # テスト用のジョブセットを作成（正しい形式）
    # jobs_setは[episode][job_index][attribute]の形式
    # ジョブ属性: [width, height, status, priority, id, arrival_time, processing_time, submitted_time]
    jobs_set = [
        [  # episode 0
            [2, 1, 0, 1, 0, 0, 10, 0],    # [width, height, status, priority, id, arrival_time, processing_time, submitted_time]
            [1, 2, 0, 2, 1, 2, 8, 2],     # [width, height, status, priority, id, arrival_time, processing_time, submitted_time]
            [3, 1, 0, 1, 2, 5, 12, 5],    # [width, height, status, priority, id, arrival_time, processing_time, submitted_time]
            [1, 1, 0, 3, 3, 8, 6, 8],     # [width, height, status, priority, id, arrival_time, processing_time, submitted_time]
            [2, 2, 0, 2, 4, 10, 15, 10],  # [width, height, status, priority, id, arrival_time, processing_time, submitted_time]
        ]
    ]
    
    # 環境の作成
    env = SchedulingEnv(
        max_step=100,
        n_window=5,
        n_on_premise_node=3,
        n_cloud_node=2,
        n_job_queue_obs=10,
        n_job_queue_bck=10,
        weight_wt=1.0,
        weight_cost=1.0,
        penalty_not_allocate=100,
        penalty_invalid_action=50,
        jobs_set=jobs_set,
        flag=0
    )
    
    return env

def test_nsga2_pure_step_time():
    """NSGA2Agentの純粋step時間計測をテスト"""
    print("=== NSGA2Agent 純粋step時間計測テスト ===")
    
    # 環境の作成
    env = create_test_env()
    
    # NSGA2Agentの初期化（小さいサイズでテスト）
    nsga2_agent = NSGA2Agent(pop_size=5, num_generations=1)
    
    # 初期集団の生成
    nsga2_agent.initialize_population(nb_jobs=5)
    
    print(f"初期集団サイズ: {len(nsga2_agent.population)}")
    
    # 個体の染色体を表示
    for i, individual in enumerate(nsga2_agent.population[:3]):
        print(f"個体 {i}: {individual.chromosome}")
    
    # 純粋step時間計測による評価
    start_time = time.time()
    nsga2_agent.evaluate_population_pure_step_time(env, n_jobs=1)
    total_time = time.time() - start_time
    
    print(f"NSGA2Agent 全体実行時間: {total_time:.4f}秒")
    print()

def test_exhaustive_pure_step_time():
    """ExhaustiveSearchAgentDistributedの純粋step時間計測をテスト"""
    print("=== ExhaustiveSearchAgentDistributed 純粋step時間計測テスト ===")
    
    # 環境の作成
    env = create_test_env()
    
    # ExhaustiveSearchAgentDistributedの初期化
    exhaustive_agent = ExhaustiveSearchAgentDistributed(num_workers=2)
    
    # 純粋step時間計測による全探索（小さいサイズでテスト）
    start_time = time.time()
    result = exhaustive_agent.run_exhaustive_search_pure_step_time(env, nb_jobs=5)
    total_time = time.time() - start_time
    
    print(f"ExhaustiveSearchAgentDistributed 全体実行時間: {total_time:.4f}秒")
    
    if result and 'statistics' in result:
        stats = result['statistics']
        print(f"評価された解の数: {stats['total_solutions']}")
        print(f"パレート最適解の数: {stats['pareto_solutions']}")
        print(f"平均純粋step時間: {stats['avg_pure_step_time']:.4f}秒")
        print(f"最小純粋step時間: {stats['min_pure_step_time']:.4f}秒")
        print(f"最大純粋step時間: {stats['max_pure_step_time']:.4f}秒")
    
    print()

def test_direct_comparison():
    """直接的な処理時間比較テスト"""
    print("=== 直接的な処理時間比較テスト ===")
    
    # 環境の作成
    env = create_test_env()
    
    # テスト用のアクションセット
    test_actions = [0, 1, 0, 1, 0]  # 5ジョブのテスト
    
    print("同じ環境で直接比較:")
    
    # 環境をリセット
    obs = env.reset()
    
    # 純粋step時間計測
    start_time = time.time()
    
    done = False
    step = 0
    while not done and step < len(test_actions):
        action = test_actions[step]
        obs, reward, scheduled, wt_step, done = env.step(action)
        if scheduled:
            step += 1
        if done:
            env.finalize_window_history()
    
    cost, _, avg_waiting_time = env.calc_objective_values()
    pure_step_time = time.time() - start_time
    
    print(f"直接計測 - 純粋step時間: {pure_step_time:.4f}秒")
    print(f"結果 - コスト: {cost:.2f}, 平均待ち時間: {avg_waiting_time:.2f}")
    print()

def main():
    """メイン関数"""
    print("純粋step時間計測機能のテストを開始します...")
    print()
    
    try:
        # NSGA2Agentのテスト
        test_nsga2_pure_step_time()
        
        # ExhaustiveSearchAgentDistributedのテスト
        test_exhaustive_pure_step_time()
        
        # 直接比較テスト
        test_direct_comparison()
        
        print("=== テスト完了 ===")
        print("両エージェントの純粋step時間を比較してください。")
        print("大きな差がある場合は、環境の内部実装やアルゴリズムの違いが原因の可能性があります。")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 