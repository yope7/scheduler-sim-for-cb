#!/usr/bin/env python3
"""
calc_objective_values関数の使用例を示すデモスクリプト
"""

import sys
import os
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.envs.scheduling_env import SchedulingEnv

def create_demo_jobs():
    """デモ用のジョブセットを作成"""
    jobs = [
        [0, 2, 1, 0, 1, 1, 0, 0],  # ジョブ1: 処理時間2, ノード1, 時刻0に到着
        [1, 3, 2, 0, 2, 2, 0, 1],  # ジョブ2: 処理時間3, ノード2, 時刻1に到着
        [2, 1, 1, 0, 3, 3, 0, 2],  # ジョブ3: 処理時間1, ノード1, 時刻2に到着
        [3, 2, 1, 0, 4, 4, 0, 3],  # ジョブ4: 処理時間2, ノード1, 時刻3に到着
    ]
    return [jobs]

def demo_calc_objective_values():
    """calc_objective_values関数のデモ"""
    print("=== calc_objective_values関数のデモ ===")
    
    # 環境を初期化
    env = SchedulingEnv(
        max_step=10,
        n_window=5,
        n_on_premise_node=3,
        n_cloud_node=2,
        n_job_queue_obs=5,
        n_job_queue_bck=5,
        weight_wt=1.0,
        weight_cost=1.0,
        penalty_not_allocate=1.0,
        penalty_invalid_action=1.0,
        jobs_set=create_demo_jobs(),
        job_type=0,
        flag=0
    )
    
    # 環境をリセット
    observation = env.reset()
    
    print("初期状態:")
    print(f"時刻: {env.time}")
    print(f"ジョブキュー: {env.job_queue[0] if np.any(env.job_queue[0] != 0) else '空'}")
    
    # シミュレーション実行
    print("\nシミュレーション開始:")
    
    for step in range(4):
        # 時刻を進める
        env.time = step
        env.append_new_job2job_queue()
        
        if np.any(env.job_queue[0] != 0):
            job = env.job_queue[0]
            print(f"\nステップ{step}: ジョブID={int(job[4])}をスケジュール")
            
            # ジョブをスケジュール
            action = [0, 0]  # オンプレミスに割り当て
            position = (0, 0)  # 位置(0,0)に配置
            waiting_time = env.do_schedule(action, job, position)
            
            print(f"  到着時刻: {job[-1]}")
            print(f"  スケジュール時刻: {env.time}")
            print(f"  待ち時間: {waiting_time}")
            print(f"  記録された待ち時間: {env.waiting_times}")
    
    # 最終的な結果を計算
    print("\n=== 最終結果 ===")
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    
    print(f"総コスト: {cost}")
    print(f"メイクスパン: {makespan}")
    print(f"平均待ち時間: {avg_waiting_time}")
    print(f"記録された待ち時間: {env.waiting_times}")
    
    print("\n=== 使用例 ===")
    print("calc_objective_values関数は以下の3つの値を返します:")
    print("1. cost: クラウド使用コスト")
    print("2. makespan: 全ジョブの完了時刻")
    print("3. avg_waiting_time: 平均待ち時間（新機能）")
    
    print("\n使用例:")
    print("cost, makespan, avg_waiting_time = env.calc_objective_values()")
    print("print(f'コスト: {cost}, メイクスパン: {makespan}, 平均待ち時間: {avg_waiting_time}')")

if __name__ == "__main__":
    demo_calc_objective_values() 