#!/usr/bin/env python3
"""
イベント駆動ジョブスケジューラ環境の簡単実行スクリプト
"""

import sys
import os
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.envs.scheduling_event_env import EventDrivenSchedulingEnv

def main():
    """メイン実行関数"""
    print("イベント駆動ジョブスケジューラ環境を実行中...")
    
    # サンプルジョブセットを作成
    jobs_set = [
        [
            [2, 1, 0, 0, 1, 8, 0, 0],     # ジョブ0: 2時間×1ノード、時刻0に到着
            [3, 2, 1, 1, 1, 10, 0, 0],    # ジョブ1: 3時間×2ノード、時刻1に到着
            [1, 1, 2, 3, 1, 6, 0, 0],     # ジョブ2: 1時間×1ノード、時刻3に到着
            [4, 2, 3, 4, 1, 12, 0, 0],    # ジョブ3: 4時間×2ノード、時刻4に到着
            [2, 3, 4, 6, 1, 14, 0, 0],    # ジョブ4: 2時間×3ノード、時刻6に到着
        ]
    ]
    
    # 環境を初期化
    env = EventDrivenSchedulingEnv(
        max_step=50,
        n_window=20,
        n_on_premise_node=6,
        n_cloud_node=6,
        n_job_queue_obs=5,
        n_job_queue_bck=10,
        weight_wt=1.0,
        weight_cost=1.0,
        penalty_not_allocate=1.0,
        penalty_invalid_action=1.0,
        jobs_set=jobs_set,
        job_type=0,
        flag=0
    )
    
    # 環境をリセット
    observation = env.reset()
    print(f"環境初期化完了")
    print(f"初期観測の形状: {observation.shape}")
    print(f"ジョブ数: {len(env.jobs)}")
    
    # シンプルなポリシーで実行
    step_count = 0
    total_reward = 0
    scheduled_jobs = 0
    
    print("\nスケジューリング開始...")
    
    while step_count < 30:
        # シンプルなポリシー: オンプレミスを優先
        action = 0  # オンプレミス
        
        # ステップ実行
        observation, rewards, scheduled, wt_step, done = env.step(action)
        
        step_count += 1
        total_reward += rewards[0]
        
        if scheduled:
            scheduled_jobs += 1
        
        if step_count % 5 == 0:
            print(f"ステップ {step_count}: 時刻={env.time}, スケジュール済み={scheduled_jobs}, "
                  f"待機中={len(env.job_queue)}, 実行中={len(env.running_jobs)}")
        
        if done:
            print(f"エピソード終了: ステップ {step_count}")
            break
    
    # 結果を表示
    print(f"\n=== 実行結果 ===")
    print(f"総ステップ数: {step_count}")
    print(f"スケジュール済みジョブ数: {scheduled_jobs}")
    print(f"完了済みジョブ数: {env.completed_jobs_count}")
    print(f"総報酬: {total_reward:.2f}")
    
    # 目的関数値を計算
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    print(f"コスト: {cost}")
    print(f"メイクスパン: {makespan}")
    print(f"平均待ち時間: {avg_waiting_time:.2f}")
    
    print("\n実行完了！")

if __name__ == "__main__":
    main() 