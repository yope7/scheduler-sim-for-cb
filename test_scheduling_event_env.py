#!/usr/bin/env python3
"""
イベント駆動ジョブスケジューラ環境のテストコード
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.envs.scheduling_event_env import EventDrivenSchedulingEnv, EventType, Event, Job

def create_sample_jobs_set():
    """サンプルのジョブセットを作成"""
    jobs_set = []
    
    # エピソード0のジョブセット
    episode_0_jobs = [
        # [処理時間, ノード数, job_id, 到着時刻, 優先度, デッドライン, ユーザーID, その他]
        [3, 2, 0, 0, 1, 10, 0, 0],    # ジョブ0: 3時間×2ノード、時刻0に到着
        [2, 1, 1, 2, 1, 8, 0, 0],     # ジョブ1: 2時間×1ノード、時刻2に到着
        [4, 3, 2, 5, 1, 15, 0, 0],    # ジョブ2: 4時間×3ノード、時刻5に到着
        [1, 1, 3, 7, 1, 12, 0, 0],    # ジョブ3: 1時間×1ノード、時刻7に到着
        [2, 2, 4, 10, 1, 18, 0, 0],   # ジョブ4: 2時間×2ノード、時刻10に到着
    ]
    
    # エピソード1のジョブセット（より多くのジョブ）
    episode_1_jobs = [
        [2, 1, 0, 0, 1, 8, 0, 0],     # ジョブ0: 2時間×1ノード、時刻0に到着
        [3, 2, 1, 1, 1, 10, 0, 0],    # ジョブ1: 3時間×2ノード、時刻1に到着
        [1, 1, 2, 3, 1, 6, 0, 0],     # ジョブ2: 1時間×1ノード、時刻3に到着
        [4, 2, 3, 4, 1, 12, 0, 0],    # ジョブ3: 4時間×2ノード、時刻4に到着
        [2, 3, 4, 6, 1, 14, 0, 0],    # ジョブ4: 2時間×3ノード、時刻6に到着
        [1, 1, 5, 8, 1, 12, 0, 0],    # ジョブ5: 1時間×1ノード、時刻8に到着
        [3, 1, 6, 9, 1, 16, 0, 0],    # ジョブ6: 3時間×1ノード、時刻9に到着
    ]
    
    jobs_set.append(episode_0_jobs)
    jobs_set.append(episode_1_jobs)
    
    return jobs_set

def test_basic_functionality():
    """基本的な機能のテスト"""
    print("=== 基本的な機能テスト ===")
    
    # サンプルジョブセットを作成
    jobs_set = create_sample_jobs_set()
    
    # 環境を初期化
    env = EventDrivenSchedulingEnv(
        max_step=100,
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
    
    print(f"環境初期化完了")
    print(f"オンプレミスノード数: {env.n_on_premise_node}")
    print(f"クラウドノード数: {env.n_cloud_node}")
    print(f"ウィンドウサイズ: {env.n_window}")
    print(f"行動空間: {env.action_space}")
    print(f"観測空間: {env.observation_space}")
    
    return env

def test_reset_and_observation():
    """リセットと観測のテスト"""
    print("\n=== リセットと観測テスト ===")
    
    env = test_basic_functionality()
    
    # 環境をリセット
    observation = env.reset()
    print(f"初期観測の形状: {observation.shape}")
    print(f"初期観測の値の範囲: [{observation.min():.3f}, {observation.max():.3f}]")
    print(f"初期時刻: {env.time}")
    print(f"初期ジョブキューサイズ: {len(env.job_queue)}")
    print(f"初期イベントキューサイズ: {len(env.event_queue)}")
    
    return env

def test_simple_scheduling():
    """簡単なスケジューリングのテスト"""
    print("\n=== 簡単なスケジューリングテスト ===")
    
    env = test_reset_and_observation()
    
    # 最初の数ステップを実行
    for step in range(10):
        print(f"\n--- ステップ {step} ---")
        print(f"現在時刻: {env.time}")
        print(f"ジョブキューサイズ: {len(env.job_queue)}")
        print(f"イベントキューサイズ: {len(env.event_queue)}")
        print(f"実行中ジョブ数: {len(env.running_jobs)}")
        
        if env.job_queue:
            print(f"待機中ジョブ: {[f'ID{j.job_id}({j.width}x{j.height})' for j in list(env.job_queue)[:3]]}")
        
        # ランダムな行動を選択
        action = np.random.randint(0, 2)  # 0: オンプレミス, 1: クラウド
        
        # ステップ実行
        observation, rewards, scheduled, wt_step, done = env.step(action)
        
        print(f"選択した行動: {'オンプレミス' if action == 0 else 'クラウド'}")
        print(f"スケジュール成功: {scheduled}")
        print(f"待ち時間: {wt_step}")
        print(f"報酬: {rewards}")
        print(f"完了フラグ: {done}")
        
        if done:
            print("エピソード終了")
            break
    
    return env

def test_event_processing():
    """イベント処理のテスト"""
    print("\n=== イベント処理テスト ===")
    
    env = test_basic_functionality()
    env.reset()
    
    # イベントを手動で追加してテスト
    print("イベントを手動で追加...")
    
    # ジョブ到着イベント
    job = Job(job_id=100, width=2, height=1, arrival_time=5)
    arrival_event = Event(
        event_type=EventType.JOB_ARRIVAL,
        timestamp=5,
        job_id=100,
        data={'job': job}
    )
    env.add_event(arrival_event)
    
    # 時間ステップイベント
    time_event = Event(EventType.TIME_STEP, timestamp=3)
    env.add_event(time_event)
    
    print(f"イベントキューサイズ: {len(env.event_queue)}")
    print(f"イベントキュー内容:")
    for i, event in enumerate(env.event_queue):
        print(f"  {i}: {event.event_type.value} at {event.timestamp}")
    
    # イベントを処理
    print("\nイベントを処理...")
    for i in range(3):
        next_event = env.get_next_event()
        if next_event:
            print(f"処理中: {next_event.event_type.value} at {next_event.timestamp}")
        else:
            print("イベントなし")
    
    return env

def test_complete_episode():
    """完全なエピソードのテスト"""
    print("\n=== 完全なエピソードテスト ===")
    
    env = test_basic_functionality()
    env.episode = 1  # エピソード1を使用（より多くのジョブ）
    observation = env.reset()
    
    step_count = 0
    total_reward = 0
    scheduled_jobs = 0
    
    print(f"エピソード開始: ジョブ数 = {len(env.jobs)}")
    
    while step_count < 50:  # 最大50ステップ
        # シンプルなポリシー: オンプレミスを優先、無効な場合はクラウド
        action = 0  # オンプレミス
        
        observation, rewards, scheduled, wt_step, done = env.step(action)
        
        step_count += 1
        total_reward += rewards[0]  # 待ち時間の報酬
        
        if scheduled:
            scheduled_jobs += 1
        
        if step_count % 10 == 0:
            print(f"ステップ {step_count}: 時刻={env.time}, スケジュール済み={scheduled_jobs}, "
                  f"待機中={len(env.job_queue)}, 実行中={len(env.running_jobs)}")
        
        if done:
            print(f"エピソード終了: ステップ {step_count}")
            break
    
    # 最終結果を表示
    print(f"\n=== エピソード結果 ===")
    print(f"総ステップ数: {step_count}")
    print(f"スケジュール済みジョブ数: {scheduled_jobs}")
    print(f"完了済みジョブ数: {env.completed_jobs_count}")
    print(f"総報酬: {total_reward:.2f}")
    print(f"平均待ち時間: {np.mean(env.waiting_times) if env.waiting_times else 0:.2f}")
    print(f"総コスト: {env.total_cost}")
    
    # 目的関数値を計算
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    print(f"計算された目的関数値:")
    print(f"  コスト: {cost}")
    print(f"  メイクスパン: {makespan}")
    print(f"  平均待ち時間: {avg_waiting_time:.2f}")
    
    return env

def test_window_visualization():
    """ウィンドウの可視化テスト"""
    print("\n=== ウィンドウ可視化テスト ===")
    
    env = test_complete_episode()
    
    # 現在のウィンドウ状態を表示
    print("\nオンプレミスウィンドウ:")
    print(env.on_premise_window['job_id'])
    
    print("\nクラウドウィンドウ:")
    print(env.cloud_window['job_id'])
    
    # 履歴を表示
    print(f"\nオンプレミス履歴の形状: {env.on_premise_window_history_full.shape}")
    print(f"クラウド履歴の形状: {env.cloud_window_history_full.shape}")
    
    # 履歴の最後の部分を表示
    print("\nオンプレミス履歴（最新10列）:")
    if env.on_premise_window_history_full.shape[1] > 10:
        print(env.on_premise_window_history_full[:, -10:])
    else:
        print(env.on_premise_window_history_full)
    
    print("\nクラウド履歴（最新10列）:")
    if env.cloud_window_history_full.shape[1] > 10:
        print(env.cloud_window_history_full[:, -10:])
    else:
        print(env.cloud_window_history_full)

def test_error_handling():
    """エラーハンドリングのテスト"""
    print("\n=== エラーハンドリングテスト ===")
    
    try:
        # 無効なパラメータで環境を作成
        env = EventDrivenSchedulingEnv(
            max_step=-1,  # 無効な値
            n_window=0,   # 無効な値
            n_on_premise_node=0,  # 無効な値
            n_cloud_node=0,       # 無効な値
            n_job_queue_obs=5,
            n_job_queue_bck=10,
            weight_wt=1.0,
            weight_cost=1.0,
            penalty_not_allocate=1.0,
            penalty_invalid_action=1.0,
            jobs_set=None
        )
        print("警告: 無効なパラメータでも環境が作成されました")
    except Exception as e:
        print(f"期待されるエラー: {e}")
    
    # 正常な環境でエラーケースをテスト
    env = test_basic_functionality()
    env.reset()
    
    # 無効な行動をテスト
    try:
        observation, rewards, scheduled, wt_step, done = env.step(999)  # 無効な行動
        print("警告: 無効な行動でもエラーが発生しませんでした")
    except Exception as e:
        print(f"期待されるエラー: {e}")

def run_all_tests():
    """すべてのテストを実行"""
    print("イベント駆動ジョブスケジューラ環境のテスト開始")
    print("=" * 50)
    
    try:
        # 基本機能テスト
        test_basic_functionality()
        
        # リセットと観測テスト
        test_reset_and_observation()
        
        # イベント処理テスト
        test_event_processing()
        
        # 簡単なスケジューリングテスト
        test_simple_scheduling()
        
        # 完全なエピソードテスト
        test_complete_episode()
        
        # ウィンドウ可視化テスト
        test_window_visualization()
        
        # エラーハンドリングテスト
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("すべてのテストが完了しました！")
        
    except Exception as e:
        print(f"\nテスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests() 