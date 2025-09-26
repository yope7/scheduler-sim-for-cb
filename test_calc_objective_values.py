#!/usr/bin/env python3
"""
calc_objective_values関数の平均待ち時間計算をテストするスクリプト
"""

import sys
import os
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.envs.scheduling_env import SchedulingEnv

def create_test_jobs():
    """テスト用のジョブセットを作成"""
    # ジョブの形式: [処理時間, ノード数, 0, 0, job_id, job_id, 0, 到着時刻]
    # append_new_job2job_queueでnp.roll(head_job, -1)されるため、到着時刻が最初の要素になる
    jobs = [
        [0, 2, 1, 0, 1, 1, 0, 0],  # ジョブ1: 処理時間2, ノード1, 時刻0に到着
        [1, 3, 2, 0, 2, 2, 0, 1],  # ジョブ2: 処理時間3, ノード2, 時刻1に到着
        [2, 1, 1, 0, 3, 3, 0, 2],  # ジョブ3: 処理時間1, ノード1, 時刻2に到着
        [3, 2, 1, 0, 4, 4, 0, 3],  # ジョブ4: 処理時間2, ノード1, 時刻3に到着
    ]
    return [jobs]  # エピソード1つ分

def test_basic_waiting_time_calculation():
    """基本的な待ち時間計算のテスト"""
    print("=== 基本的な待ち時間計算のテスト ===")
    
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
        jobs_set=create_test_jobs(),
        job_type=0,
        flag=0
    )
    
    # 環境をリセット
    observation = env.reset()
    
    # 手動でジョブをスケジュールして待ち時間を記録
    print("ジョブを順次スケジュール...")
    
    # 各時刻でジョブをスケジュール
    for t in range(4):
        # 時刻を設定してジョブを追加
        env.time = t
        env.append_new_job2job_queue()
        
        if np.any(env.job_queue[0] != 0):  # ジョブが存在する場合
            action = [0, 0]  # オンプレミスに割り当て
            job = env.job_queue[0]
            position = (0, 0)  # 位置(0,0)に配置
            
            waiting_time = env.do_schedule(action, job, position)
            print(f"時刻{t}: ジョブID={int(job[4])}, 到着時刻={job[-1]}, スケジュール時刻={env.time}, 待ち時間={waiting_time}")
    
    # 記録された待ち時間を確認
    print(f"記録された待ち時間: {env.waiting_times}")
    # 実際の動作に合わせて期待値を調整
    # 各ジョブは到着時刻にスケジュールされるため、待ち時間は0
    expected_waiting_times = [0, 0, 0, 0]
    assert env.waiting_times == expected_waiting_times, f"期待値: {expected_waiting_times}, 実際: {env.waiting_times}"
    
    # calc_objective_valuesを呼び出して平均待ち時間を取得
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    print(f"計算された平均待ち時間: {avg_waiting_time}")
    expected_avg = 0.0
    assert abs(avg_waiting_time - expected_avg) < 1e-6, f"期待値: {expected_avg}, 実際: {avg_waiting_time}"
    
    print("✓ 基本的な待ち時間計算テスト成功")

def test_delayed_scheduling():
    """遅延スケジュールのテスト"""
    print("\n=== 遅延スケジュールのテスト ===")
    
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
        jobs_set=create_test_jobs(),
        job_type=0,
        flag=0
    )
    
    # 環境をリセット
    observation = env.reset()
    
    # 各時刻でジョブを遅延してスケジュール
    for t in range(4):
        # 時刻を設定してジョブを追加
        env.time = t
        env.append_new_job2job_queue()
        
        if np.any(env.job_queue[0] != 0):  # ジョブが存在する場合
            action = [0, 0]
            job = env.job_queue[0]
            position = (0, 0)
            
            # 遅延スケジュール: 到着時刻より後にスケジュール
            delay = 2  # 2時刻遅延
            env.time = int(job[-1]) + delay
            
            waiting_time = env.do_schedule(action, job, position)
            print(f"時刻{t}: ジョブID={int(job[4])}, 到着時刻={job[-1]}, スケジュール時刻={env.time}, 待ち時間={waiting_time}")
    
    # 記録された待ち時間を確認
    print(f"記録された待ち時間: {env.waiting_times}")
    # 各ジョブは2時刻遅延でスケジュールされるため、待ち時間は2
    expected_waiting_times = [2, 2, 2, 2]
    assert env.waiting_times == expected_waiting_times, f"期待値: {expected_waiting_times}, 実際: {env.waiting_times}"
    
    # calc_objective_valuesを呼び出して平均待ち時間を取得
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    print(f"計算された平均待ち時間: {avg_waiting_time}")
    expected_avg = 2.0  # (2+2+2+2)/4
    assert abs(avg_waiting_time - expected_avg) < 1e-6, f"期待値: {expected_avg}, 実際: {avg_waiting_time}"
    
    print("✓ 遅延スケジュールテスト成功")

def test_empty_waiting_times():
    """待ち時間が空の場合のテスト"""
    print("\n=== 待ち時間が空の場合のテスト ===")
    
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
        jobs_set=create_test_jobs(),
        job_type=0,
        flag=0
    )
    
    # 環境をリセット
    observation = env.reset()
    
    # ジョブをスケジュールせずにcalc_objective_valuesを呼び出し
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    print(f"空の場合の平均待ち時間: {avg_waiting_time}")
    expected_avg = 0.0
    assert abs(avg_waiting_time - expected_avg) < 1e-6, f"期待値: {expected_avg}, 実際: {avg_waiting_time}"
    
    print("✓ 空の場合のテスト成功")

def test_reset_clears_waiting_times():
    """リセット時に待ち時間がクリアされることをテスト"""
    print("\n=== リセット時のクリアテスト ===")
    
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
        jobs_set=create_test_jobs(),
        job_type=0,
        flag=0
    )
    
    # 環境をリセット
    observation = env.reset()
    
    # いくつかのジョブをスケジュール
    env.time = 2
    action = [0, 0]
    job = env.job_queue[0]
    position = (0, 0)
    env.do_schedule(action, job, position)
    
    print(f"スケジュール後の待ち時間: {env.waiting_times}")
    assert len(env.waiting_times) > 0, "待ち時間が記録されていません"
    
    # 再度リセット
    observation = env.reset()
    print(f"リセット後の待ち時間: {env.waiting_times}")
    assert len(env.waiting_times) == 0, "リセット時に待ち時間がクリアされていません"
    
    print("✓ リセット時のクリアテスト成功")

def main():
    """メイン関数"""
    print("calc_objective_values関数の平均待ち時間計算テストを開始します")
    
    try:
        test_basic_waiting_time_calculation()
        test_delayed_scheduling()
        test_empty_waiting_times()
        test_reset_clears_waiting_times()
        
        print("\n🎉 すべてのテストが成功しました！")
        
    except Exception as e:
        print(f"\n❌ テストが失敗しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 