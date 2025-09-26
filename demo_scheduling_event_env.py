#!/usr/bin/env python3
"""
イベント駆動ジョブスケジューラ環境のデモスクリプト
"""

import sys
import os
import numpy as np
import time

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.envs.scheduling_event_env import EventDrivenSchedulingEnv, EventType, Event, Job

def create_demo_jobs():
    """デモ用のジョブセットを作成"""
    jobs = [
        # [処理時間, ノード数, job_id, 到着時刻, 優先度, デッドライン, ユーザーID, その他]
        [2, 1, 0, 0, 1, 8, 0, 0],     # ジョブ0: 2時間×1ノード、時刻0に到着
        [3, 2, 1, 1, 1, 10, 0, 0],    # ジョブ1: 3時間×2ノード、時刻1に到着
        [1, 1, 2, 3, 1, 6, 0, 0],     # ジョブ2: 1時間×1ノード、時刻3に到着
        [4, 2, 3, 4, 1, 12, 0, 0],    # ジョブ3: 4時間×2ノード、時刻4に到着
        [2, 3, 4, 6, 1, 14, 0, 0],    # ジョブ4: 2時間×3ノード、時刻6に到着
    ]
    return [jobs]  # 1エピソード分

def simple_policy(env, observation):
    """シンプルなポリシー: オンプレミスを優先、無効な場合はクラウド"""
    # オンプレミスを試す
    action = 0
    is_valid, _, _ = env.check_is_valid_action([0, 0])
    
    if not is_valid:
        # オンプレミスが無効な場合はクラウドを試す
        action = 1
        is_valid, _, _ = env.check_is_valid_action([0, 1])
        
        if not is_valid:
            # どちらも無効な場合はオンプレミスを選択（ペナルティを受ける）
            action = 0
    
    return action

def greedy_policy(env, observation):
    """貪欲ポリシー: 待ち時間が短い方を選択"""
    if not env.job_queue:
        return 0  # ジョブがない場合はオンプレミス
    
    # オンプレミスとクラウドの両方を試す
    on_premise_valid, on_premise_wt, _ = env.check_is_valid_action([0, 0])
    cloud_valid, cloud_wt, _ = env.check_is_valid_action([0, 1])
    
    if on_premise_valid and cloud_valid:
        # 両方有効な場合、待ち時間が短い方を選択
        return 0 if on_premise_wt <= cloud_wt else 1
    elif on_premise_valid:
        return 0
    elif cloud_valid:
        return 1
    else:
        return 0  # どちらも無効な場合はオンプレミス

def random_policy(env, observation):
    """ランダムポリシー"""
    return np.random.randint(0, 2)

def run_demo(policy_name="simple", max_steps=30, verbose=True):
    """デモを実行"""
    print(f"=== イベント駆動ジョブスケジューラ環境デモ ===")
    print(f"ポリシー: {policy_name}")
    print(f"最大ステップ数: {max_steps}")
    print("-" * 50)
    
    # 環境を初期化
    jobs_set = create_demo_jobs()
    env = EventDrivenSchedulingEnv(
        max_step=max_steps,
        n_window=15,
        n_on_premise_node=4,
        n_cloud_node=4,
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
    
    # ポリシーを選択
    if policy_name == "simple":
        policy = simple_policy
    elif policy_name == "greedy":
        policy = greedy_policy
    elif policy_name == "random":
        policy = random_policy
    else:
        policy = simple_policy
    
    # 統計情報
    step_count = 0
    total_reward = 0
    scheduled_jobs = 0
    total_waiting_time = 0
    total_cost = 0
    
    if verbose:
        print(f"初期状態:")
        print(f"  時刻: {env.time}")
        print(f"  ジョブキュー: {len(env.job_queue)}")
        print(f"  イベントキュー: {len(env.event_queue)}")
        print()
    
    # メインループ
    while step_count < max_steps:
        # ポリシーに基づいて行動を選択
        action = policy(env, observation)
        
        # ステップ実行
        observation, rewards, scheduled, wt_step, done = env.step(action)
        
        # 統計を更新
        step_count += 1
        total_reward += rewards[0]  # 待ち時間の報酬
        if scheduled:
            scheduled_jobs += 1
            total_waiting_time += wt_step
        total_cost += abs(rewards[1])  # コストの報酬（絶対値）
        
        if verbose:
            print(f"ステップ {step_count:2d}: 時刻={env.time:2d}, "
                  f"行動={'オンプレミス' if action == 0 else 'クラウド'}, "
                  f"スケジュール={'成功' if scheduled else '失敗'}, "
                  f"待ち時間={wt_step:2d}, "
                  f"報酬=[{rewards[0]:5.2f}, {rewards[1]:5.2f}], "
                  f"待機中={len(env.job_queue):2d}, "
                  f"実行中={len(env.running_jobs):2d}")
        
        if done:
            if verbose:
                print(f"エピソード終了: ステップ {step_count}")
            break
    
    # 結果を表示
    print("\n" + "=" * 50)
    print("デモ結果:")
    print(f"  総ステップ数: {step_count}")
    print(f"  スケジュール済みジョブ数: {scheduled_jobs}")
    print(f"  完了済みジョブ数: {env.completed_jobs_count}")
    print(f"  総報酬: {total_reward:.2f}")
    print(f"  平均待ち時間: {total_waiting_time / scheduled_jobs if scheduled_jobs > 0 else 0:.2f}")
    print(f"  総コスト: {total_cost:.2f}")
    
    # 目的関数値を計算
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    print(f"  計算された目的関数値:")
    print(f"    コスト: {cost}")
    print(f"    メイクスパン: {makespan}")
    print(f"    平均待ち時間: {avg_waiting_time:.2f}")
    
    # ウィンドウ状態を表示
    print(f"\n最終ウィンドウ状態:")
    print(f"  オンプレミス:")
    print(env.on_premise_window['job_id'])
    print(f"  クラウド:")
    print(env.cloud_window['job_id'])
    
    return {
        'step_count': step_count,
        'scheduled_jobs': scheduled_jobs,
        'completed_jobs': env.completed_jobs_count,
        'total_reward': total_reward,
        'avg_waiting_time': total_waiting_time / scheduled_jobs if scheduled_jobs > 0 else 0,
        'total_cost': total_cost,
        'makespan': makespan
    }

def compare_policies():
    """異なるポリシーの性能を比較"""
    print("=== ポリシー比較 ===")
    
    policies = ["simple", "greedy", "random"]
    results = {}
    
    for policy in policies:
        print(f"\n{policy.upper()} ポリシーを実行中...")
        result = run_demo(policy, max_steps=30, verbose=False)
        results[policy] = result
    
    # 結果を比較
    print("\n" + "=" * 60)
    print("ポリシー比較結果:")
    print(f"{'ポリシー':<10} {'ステップ':<8} {'スケジュール':<12} {'完了':<6} {'報酬':<8} {'待ち時間':<10} {'コスト':<8} {'メイクスパン':<10}")
    print("-" * 80)
    
    for policy, result in results.items():
        print(f"{policy:<10} {result['step_count']:<8} {result['scheduled_jobs']:<12} "
              f"{result['completed_jobs']:<6} {result['total_reward']:<8.2f} "
              f"{result['avg_waiting_time']:<10.2f} {result['total_cost']:<8.2f} "
              f"{result['makespan']:<10}")

def interactive_demo():
    """インタラクティブデモ"""
    print("=== インタラクティブデモ ===")
    print("手動で行動を選択できます。")
    print("0: オンプレミス, 1: クラウド, q: 終了")
    
    jobs_set = create_demo_jobs()
    env = EventDrivenSchedulingEnv(
        max_step=50,
        n_window=15,
        n_on_premise_node=4,
        n_cloud_node=4,
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
    
    observation = env.reset()
    step_count = 0
    
    while step_count < 50:
        print(f"\n--- ステップ {step_count} ---")
        print(f"時刻: {env.time}")
        print(f"待機中ジョブ: {len(env.job_queue)}")
        print(f"実行中ジョブ: {len(env.running_jobs)}")
        
        if env.job_queue:
            print("待機中ジョブ詳細:")
            for i, job in enumerate(list(env.job_queue)[:3]):
                print(f"  {i}: ID{job.job_id} ({job.width}x{job.height}) 到着時刻:{job.arrival_time}")
        
        # 有効な行動をチェック
        on_premise_valid, on_premise_wt, _ = env.check_is_valid_action([0, 0])
        cloud_valid, cloud_wt, _ = env.check_is_valid_action([0, 1])
        
        print(f"有効な行動:")
        print(f"  オンプレミス: {'有効' if on_premise_valid else '無効'} (待ち時間: {on_premise_wt if on_premise_valid else 'N/A'})")
        print(f"  クラウド: {'有効' if cloud_valid else '無効'} (待ち時間: {cloud_wt if cloud_valid else 'N/A'})")
        
        # ユーザー入力
        user_input = input("行動を選択 (0/1/q): ").strip().lower()
        
        if user_input == 'q':
            print("デモを終了します")
            break
        elif user_input in ['0', '1']:
            action = int(user_input)
        else:
            print("無効な入力です。0, 1, または q を入力してください。")
            continue
        
        # ステップ実行
        observation, rewards, scheduled, wt_step, done = env.step(action)
        
        print(f"結果: スケジュール={'成功' if scheduled else '失敗'}, "
              f"待ち時間={wt_step}, 報酬={rewards}")
        
        step_count += 1
        
        if done:
            print("エピソード終了")
            break
    
    # 最終結果
    cost, makespan, avg_waiting_time = env.calc_objective_values()
    print(f"\n最終結果:")
    print(f"  完了済みジョブ: {env.completed_jobs_count}")
    print(f"  総コスト: {cost}")
    print(f"  メイクスパン: {makespan}")
    print(f"  平均待ち時間: {avg_waiting_time:.2f}")

def main():
    """メイン関数"""
    print("イベント駆動ジョブスケジューラ環境デモ")
    print("=" * 50)
    
    while True:
        print("\n選択してください:")
        print("1. シンプルポリシーデモ")
        print("2. 貪欲ポリシーデモ")
        print("3. ランダムポリシーデモ")
        print("4. ポリシー比較")
        print("5. インタラクティブデモ")
        print("6. 終了")
        
        choice = input("選択 (1-6): ").strip()
        
        if choice == '1':
            run_demo("simple", max_steps=30)
        elif choice == '2':
            run_demo("greedy", max_steps=30)
        elif choice == '3':
            run_demo("random", max_steps=30)
        elif choice == '4':
            compare_policies()
        elif choice == '5':
            interactive_demo()
        elif choice == '6':
            print("デモを終了します")
            break
        else:
            print("無効な選択です。1-6を入力してください。")

if __name__ == "__main__":
    main() 