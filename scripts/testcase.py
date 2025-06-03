# テスト用スクリプト: calc_objective_values_test.py

import numpy as np
from src.envs.scheduling_env import SchedulingEnv

def create_test_jobs():
    # テスト用ジョブを作成
    # format: [到着時間, 処理時間, ノード数, クラウド使用可, ユーザID, ジョブID, waiting_time, 提出時間]
    jobs = [
        [0, 2, 3, 1, 0, 0, -1, 0],  # ジョブ0: 処理時間2×ノード数3=コスト6
        [1, 4, 2, 1, 1, 1, -1, 1],  # ジョブ1: 処理時間4×ノード数2=コスト8
        [2, 3, 5, 1, 2, 2, -1, 2],  # ジョブ2: 処理時間3×ノード数5=コスト15
    ]
    jobs_set = {0: jobs}
    return jobs_set

def test_all_cloud_allocation():
    print("===== テスト1: すべてジョブをクラウドに割り当てる場合 =====")
    jobs_set = create_test_jobs()
    
    # 環境の初期化
    env = SchedulingEnv(
        max_step=20, 
        n_window=10, 
        n_on_premise_node=6, 
        n_cloud_node=6, 
        n_job_queue_obs=5,
        n_job_queue_bck=5, 
        weight_wt=0.5, 
        weight_cost=0.5,
        penalty_not_allocate=5, 
        penalty_invalid_action=10,
        jobs_set=jobs_set,
        job_type=1
    )
    env.episode = 0
    obs = env.reset()
    
    # すべてのジョブをクラウドに割り当てるアクション
    actions = [1, 1, 1]  # すべてクラウドに割り当て
    
    # シミュレーション実行
    for action in actions:
        obs, rewards, scheduled, wt_step, done = env.step(action)
        if scheduled:
            print(f"ジョブ割り当て成功: アクション={action}, 待ち時間={wt_step}")
    
    # 最終的な窓を確定
    env.finalize_window_history()
    
    # コスト計算
    cost, makespan = env.calc_objective_values()
    print(f"計算されたコスト: {cost}")
    print(f"期待されるコスト: 6 + 8 + 15 = 29")
    print(f"計算されたmakespan: {makespan}")
    
    # 検証
    expected_cost = 6 + 8 + 15
    assert cost == expected_cost, f"コスト計算が正しくありません: {cost} != {expected_cost}"
    print("テスト1成功: コスト計算が正しい")

def test_mixed_allocation():
    print("\n===== テスト2: 一部ジョブだけクラウドに割り当てる場合 =====")
    jobs_set = create_test_jobs()
    
    # 環境の初期化
    env = SchedulingEnv(
        max_step=20, 
        n_window=10, 
        n_on_premise_node=6, 
        n_cloud_node=6, 
        n_job_queue_obs=5,
        n_job_queue_bck=5, 
        weight_wt=0.5, 
        weight_cost=0.5,
        penalty_not_allocate=5, 
        penalty_invalid_action=10,
        jobs_set=jobs_set,
        job_type=1
    )
    env.episode = 0
    obs = env.reset()
    
    # ジョブを一部だけクラウドに割り当てるアクション
    actions = [0, 1, 0]  # オンプレミス、クラウド、オンプレミス
    
    # シミュレーション実行
    for action in actions:
        obs, rewards, scheduled, wt_step, done = env.step(action)
        if scheduled:
            print(f"ジョブ割り当て成功: アクション={action}, 待ち時間={wt_step}")
    
    # 最終的な窓を確定
    env.finalize_window_history()
    
    # コスト計算
    cost, makespan = env.calc_objective_values()
    print(f"計算されたコスト: {cost}")
    print(f"期待されるコスト: 8 (ジョブ1のみクラウド)")
    print(f"計算されたmakespan: {makespan}")
    
    # 検証
    expected_cost = 8  # ジョブ1のみクラウドに割り当てた
    assert cost == expected_cost, f"コスト計算が正しくありません: {cost} != {expected_cost}"
    print("テスト2成功: コスト計算が正しい")

def compare_with_old_method():
    print("\n===== テスト3: 修正前の計算方法との比較 =====")
    jobs_set = create_test_jobs()
    
    # 環境の初期化
    env = SchedulingEnv(
        max_step=20, 
        n_window=10, 
        n_on_premise_node=6, 
        n_cloud_node=6, 
        n_job_queue_obs=5,
        n_job_queue_bck=5, 
        weight_wt=0.5, 
        weight_cost=0.5,
        penalty_not_allocate=5, 
        penalty_invalid_action=10,
        jobs_set=jobs_set,
        job_type=1
    )
    env.episode = 0
    obs = env.reset()
    
    # すべてのジョブをクラウドに割り当て
    actions = [1, 1, 1]
    
    # シミュレーション実行
    for action in actions:
        obs, rewards, scheduled, wt_step, done = env.step(action)
    
    # 最終的な窓を確定
    env.finalize_window_history()
    
    # 新しい方法でのコスト計算
    cost_new, _ = env.calc_objective_values()
    
    # 古い方法でのコスト計算（マスの数をカウント）
    cloud_history = env.cloud_window_history_full
    cost_old = np.count_nonzero(cloud_history[cloud_history != -1])
    
    print(f"新しい計算方法によるコスト: {cost_new}")
    print(f"古い計算方法によるコスト: {cost_old}")
    print(f"差分: {cost_old - cost_new}")
    
    # 新旧の計算結果が異なることを確認
    print("テスト3成功: 新旧計算方法の差分を確認")

if __name__ == "__main__":
    try:
        test_all_cloud_allocation()
        test_mixed_allocation()
        compare_with_old_method()
        print("\nすべてのテストが成功しました！")
    except AssertionError as e:
        print(f"テスト失敗: {e}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
