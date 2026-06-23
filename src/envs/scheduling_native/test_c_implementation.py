#!/usr/bin/env python3
"""
C言語実装のテストスクリプト
"""
import numpy as np
import time
import sys
import os

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position,
        time_transition,
        do_schedule,
        get_unique_job_ids,
        calculate_makespan
    )
    print("✓ C言語実装のインポートに成功しました")
except ImportError as e:
    print(f"✗ C言語実装のインポートに失敗しました: {e}")
    print("ビルドしてください: uv sync")
    sys.exit(1)


def test_cache_build():
    """キャッシュ構築のテスト"""
    print("\n=== キャッシュ構築のテスト ===")
    
    H, W = 10, 100
    window_status = np.zeros((H, W), dtype=np.int32)
    
    # いくつかのセルを占有
    window_status[0:3, 0:5] = 1
    window_status[5:7, 10:15] = 1
    
    cache = WindowCache(window_status, H, W)
    print(f"✓ キャッシュ構築成功: H={H}, W={W}")
    
    return cache


def test_find_allocation_position():
    """割り当て位置探索のテスト"""
    print("\n=== 割り当て位置探索のテスト ===")
    
    H, W = 10, 100
    window_status = np.zeros((H, W), dtype=np.int32)
    
    # いくつかのセルを占有
    window_status[0:3, 0:5] = 1
    window_status[5:7, 10:15] = 1
    
    cache = WindowCache(window_status, H, W)
    
    # ジョブ1: 3x5のジョブ（配置可能）
    position, waiting_time = find_allocation_position(
        cache, job_width=3, job_height=5,
        when_submitted=0, current_time=10
    )
    
    if position is not None:
        print(f"✓ 位置が見つかりました: {position}, 待ち時間: {waiting_time}")
    else:
        print("✗ 位置が見つかりませんでした")
    
    # ジョブ2: 100x10のジョブ（大きすぎる）
    position2, waiting_time2 = find_allocation_position(
        cache, job_width=100, job_height=10,
        when_submitted=0, current_time=10
    )
    
    if position2 is None:
        print("✓ 大きすぎるジョブは正しく拒否されました")
    else:
        print("✗ 大きすぎるジョブが受け入れられました")
    
    return cache


def test_time_transition():
    """時間遷移のテスト"""
    print("\n=== 時間遷移のテスト ===")
    
    H, W = 5, 20
    # 書き込み可能な配列を作成（既存のPython実装と同様）
    window_status = np.zeros((H, W), dtype=np.int32)
    window_job_id = np.full((H, W), -1, dtype=np.int32)
    
    # 最初の列にジョブを配置
    window_status[0:2, 0:3] = 1
    window_job_id[0:2, 0:3] = 1
    
    # 2列目にもジョブを配置（スライド後の確認用）
    window_status[0:2, 3:6] = 1
    window_job_id[0:2, 3:6] = 2
    
    print("スライド前:")
    print(f"  最初の列: status={window_status[:, 0]}, job_id={window_job_id[:, 0]}")
    print(f"  2列目: status={window_status[:, 1]}, job_id={window_job_id[:, 1]}")
    
    # C連続配列として保証（既存のPython実装と同様）
    window_status = np.ascontiguousarray(window_status, dtype=np.int32)
    window_job_id = np.ascontiguousarray(window_job_id, dtype=np.int32)
    
    # time_transitionを呼び出し（配列を返す）
    result = time_transition(window_status, window_job_id, H, W, slide=True)
    if result is not None:
        window_status, window_job_id = result
    
    print("スライド後:")
    print(f"  最初の列: status={window_status[:, 0]}, job_id={window_job_id[:, 0]}")
    print(f"  2列目: status={window_status[:, 1]}, job_id={window_job_id[:, 1]}")
    print(f"  最後の列: status={window_status[:, -1]}, job_id={window_job_id[:, -1]}")
    
    # 最初の列が2列目の値になっているか確認（スライドが正しく動作しているか）
    # スライド後、最初の列は元の2列目の値になる
    if (np.all(window_status[:, 0] == window_status[:, 1]) and 
        np.all(window_job_id[:, 0] == window_job_id[:, 1])):
        print("✓ 時間遷移が正しく動作しています")
    else:
        print("✗ 時間遷移に問題があります")
        print(f"  期待: 最初の列が2列目の値と一致")
        print(f"  実際: 最初の列={window_status[:, 0]}, 2列目={window_status[:, 1]}")


def test_do_schedule():
    """スケジュール実行のテスト"""
    print("\n=== スケジュール実行のテスト ===")
    
    H, W = 10, 50
    window_status = np.zeros((H, W), dtype=np.int32)
    window_job_id = np.full((H, W), -1, dtype=np.int32)
    
    # 位置 (2, 5) に3x4のジョブを配置
    position = (2, 5)
    do_schedule(
        window_status, window_job_id, H, W,
        job_width=3, job_height=4, job_id=1,
        position=position
    )
    
    # 配置された領域を確認
    allocated_region = window_status[2:6, 5:8]
    allocated_job_id = window_job_id[2:6, 5:8]
    
    if np.all(allocated_region == 1) and np.all(allocated_job_id == 1):
        print("✓ ジョブが正しく配置されました")
    else:
        print("✗ ジョブの配置に問題があります")
        print(f"  status: {allocated_region}")
        print(f"  job_id: {allocated_job_id}")


def test_get_unique_job_ids():
    """ユニークなジョブID取得のテスト"""
    print("\n=== ユニークなジョブID取得のテスト ===")
    
    H, W = 5, 20
    history_matrix = np.full((H, W), -1, dtype=np.int32)
    
    # いくつかのジョブIDを配置
    history_matrix[0:2, 0:5] = 1
    history_matrix[2:4, 5:10] = 2
    history_matrix[0:1, 10:15] = 1  # 重複
    
    unique_ids = get_unique_job_ids(history_matrix, H, W, max_job_id=100)
    
    print(f"ユニークなジョブID: {unique_ids}")
    
    if len(unique_ids) == 2 and 1 in unique_ids and 2 in unique_ids:
        print("✓ ユニークなジョブIDが正しく取得されました")
    else:
        print("✗ ユニークなジョブIDの取得に問題があります")


def test_calculate_makespan():
    """makespan計算のテスト"""
    print("\n=== makespan計算のテスト ===")
    
    H, W = 5, 20
    window_matrix = np.full((H, W), -1, dtype=np.int32)
    
    # いくつかのジョブを配置
    window_matrix[0:2, 0:5] = 1
    window_matrix[2:4, 5:15] = 2  # 最大列インデックスは14
    
    makespan = calculate_makespan(window_matrix, H, W)
    
    print(f"makespan: {makespan}")
    
    if makespan == 14:
        print("✓ makespanが正しく計算されました")
    else:
        print(f"✗ makespanの計算に問題があります（期待値: 14, 実際: {makespan}）")


def benchmark_find_allocation_position():
    """割り当て位置探索のベンチマーク"""
    print("\n=== 割り当て位置探索のベンチマーク ===")
    
    H, W = 50, 500
    window_status = np.zeros((H, W), dtype=np.int32)
    
    # ランダムにセルを占有
    np.random.seed(42)
    occupied_mask = np.random.random((H, W)) < 0.3
    window_status[occupied_mask] = 1
    
    cache = WindowCache(window_status, H, W)
    
    # ベンチマーク実行
    n_iterations = 1000
    job_width, job_height = 5, 3
    
    start_time = time.time()
    for _ in range(n_iterations):
        position, waiting_time = find_allocation_position(
            cache, job_width, job_height,
            when_submitted=0, current_time=10
        )
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / n_iterations * 1000  # ms
    
    print(f"✓ {n_iterations}回の探索を {elapsed:.3f}秒で完了")
    print(f"  平均時間: {avg_time:.3f}ms/回")


def main():
    """メイン関数"""
    print("=" * 60)
    print("C言語実装のテスト")
    print("=" * 60)
    
    try:
        test_cache_build()
        test_find_allocation_position()
        test_time_transition()
        test_do_schedule()
        test_get_unique_job_ids()
        test_calculate_makespan()
        benchmark_find_allocation_position()
        
        print("\n" + "=" * 60)
        print("✓ すべてのテストが完了しました")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

