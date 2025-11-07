#!/usr/bin/env python3
"""
既存のPython実装とC言語実装の完全な互換性テスト
同じ入力に対して同じ出力が得られることを確認
"""
import numpy as np
import sys
import os

# パスを追加（プロジェクトのルートディレクトリを追加）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position as c_find_allocation_position,
        time_transition as c_time_transition,
        do_schedule as c_do_schedule,
        get_unique_job_ids as c_get_unique_job_ids,
        calculate_makespan as c_calculate_makespan
    )
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    print("警告: C言語実装が利用できません。")
    sys.exit(1)

from src.envs.scheduling_env import (
    time_transition_njit,
    get_unique_job_ids_njit,
    calculate_makespan_batch_njit
)


def test_time_transition_compatibility():
    """time_transitionの互換性テスト"""
    print("\n=== time_transition互換性テスト ===")
    
    H, W = 10, 50
    np.random.seed(42)
    
    # 同じ初期状態を作成
    window_status_py = np.random.randint(0, 2, (H, W), dtype=np.int32)
    window_job_id_py = np.random.randint(-1, 10, (H, W), dtype=np.int32)
    
    window_status_c = window_status_py.copy()
    window_job_id_c = window_job_id_py.copy()
    
    # Python実装を実行
    window_status_py_cont = np.ascontiguousarray(window_status_py, dtype=np.int32)
    window_job_id_py_cont = np.ascontiguousarray(window_job_id_py, dtype=np.int32)
    result_py = time_transition_njit(
        window_status_py_cont, window_job_id_py_cont,
        np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.int32),
        True, False
    )
    window_status_py_result = result_py[0]
    window_job_id_py_result = result_py[1]
    
    # C言語実装を実行
    window_status_c_cont = np.ascontiguousarray(window_status_c, dtype=np.int32)
    window_job_id_c_cont = np.ascontiguousarray(window_job_id_c, dtype=np.int32)
    result_c = c_time_transition(
        window_status_c_cont, window_job_id_c_cont,
        H, W, True
    )
    if result_c is not None:
        window_status_c_result, window_job_id_c_result = result_c
    else:
        window_status_c_result = window_status_c_cont
        window_job_id_c_result = window_job_id_c_cont
    
    # 結果を比較
    status_match = np.array_equal(window_status_py_result, window_status_c_result)
    job_id_match = np.array_equal(window_job_id_py_result, window_job_id_c_result)
    
    if status_match and job_id_match:
        print("✓ time_transition: 完全に同じ結果")
    else:
        print("✗ time_transition: 結果が異なります")
        if not status_match:
            print(f"  status不一致:")
            print(f"    Python: {window_status_py_result[:, 0]}")
            print(f"    C:      {window_status_c_result[:, 0]}")
        if not job_id_match:
            print(f"  job_id不一致:")
            print(f"    Python: {window_job_id_py_result[:, 0]}")
            print(f"    C:      {window_job_id_c_result[:, 0]}")


def test_find_allocation_position_compatibility():
    """find_allocation_positionの互換性テスト"""
    print("\n=== find_allocation_position互換性テスト ===")
    
    H, W = 20, 100
    np.random.seed(42)
    
    # 同じ初期状態を作成
    window_status = np.random.randint(0, 2, (H, W), dtype=np.int32)
    
    # Python実装のキャッシュを構築（簡易版）
    status = window_status
    occ = (status != 0).astype(np.int32)
    free_per_col = status.shape[0] - occ.sum(axis=0)
    ps = np.zeros((H+1, W+1), dtype=np.int32)
    ps[1:,1:] = np.cumsum(np.cumsum(occ.astype(np.int32), axis=0, dtype=np.int32), axis=1, dtype=np.int32)
    free_nodes_list = [np.flatnonzero(status[:, c] == 0) for c in range(status.shape[1])]
    
    # C言語実装のキャッシュを構築
    cache = WindowCache(window_status, H, W)
    
    # テストケース
    test_cases = [
        (5, 3, 0, 10),   # 通常のジョブ
        (10, 5, 0, 10),  # 大きめのジョブ
        (3, 2, 0, 10),   # 小さめのジョブ
    ]
    
    all_match = True
    for job_width, job_height, when_submitted, current_time in test_cases:
        # Python実装（簡易版）
        # スライディングウィンドウの最小値計算
        from numpy.lib.stride_tricks import sliding_window_view
        k = job_width
        need = job_height
        if k <= free_per_col.shape[0]:
            mins = sliding_window_view(free_per_col, k).min(axis=1)
        else:
            mins = np.array([])
        
        # First-Fit探索（簡易版）
        limit_a = W - k + 1
        py_found = False
        py_position = None
        py_waiting_time = np.inf
        
        for a in range(limit_a):
            if len(mins) > 0 and mins[a] < need:
                continue
            a2 = a + k
            max_i = H - job_height + 1
            for i in range(max_i):
                i2 = i + job_height
                occ_sum = ps[i2, a2] - ps[i, a2] - ps[i2, a] + ps[i, a]
                if occ_sum == 0:
                    py_found = True
                    py_position = (i, a)
                    py_waiting_time = current_time + a - when_submitted
                    break
            if py_found:
                break
        
        # C言語実装
        c_position, c_waiting_time = c_find_allocation_position(
            cache, job_width, job_height, when_submitted, current_time
        )
        
        # 結果を比較
        if py_found:
            if c_position is None:
                print(f"✗ ケース ({job_width}, {job_height}): Pythonは見つかったがCは見つからなかった")
                all_match = False
            elif py_position != c_position:
                print(f"✗ ケース ({job_width}, {job_height}): 位置が異なる")
                print(f"    Python: {py_position}, C: {c_position}")
                all_match = False
            elif abs(py_waiting_time - c_waiting_time) > 1e-6:
                print(f"✗ ケース ({job_width}, {job_height}): 待ち時間が異なる")
                print(f"    Python: {py_waiting_time}, C: {c_waiting_time}")
                all_match = False
            else:
                print(f"✓ ケース ({job_width}, {job_height}): 完全に同じ結果")
        else:
            if c_position is not None:
                print(f"✗ ケース ({job_width}, {job_height}): Pythonは見つからなかったがCは見つかった")
                all_match = False
            else:
                print(f"✓ ケース ({job_width}, {job_height}): 両方とも見つからなかった")
    
    if all_match:
        print("✓ find_allocation_position: すべてのテストケースで完全に同じ結果")
    else:
        print("✗ find_allocation_position: 一部のテストケースで結果が異なります")


def test_get_unique_job_ids_compatibility():
    """get_unique_job_idsの互換性テスト"""
    print("\n=== get_unique_job_ids互換性テスト ===")
    
    H, W = 10, 50
    np.random.seed(42)
    
    # 同じ初期状態を作成
    history_matrix = np.random.randint(-1, 10, (H, W), dtype=np.int32)
    max_job_id = 50000
    
    # Python実装
    py_result = get_unique_job_ids_njit(history_matrix, max_job_id)
    py_sorted = np.sort(py_result)
    
    # C言語実装
    c_result = c_get_unique_job_ids(history_matrix, H, W, max_job_id)
    c_sorted = np.sort(c_result)
    
    # 結果を比較
    if np.array_equal(py_sorted, c_sorted):
        print("✓ get_unique_job_ids: 完全に同じ結果")
        print(f"  ユニークなジョブID数: {len(py_result)}")
    else:
        print("✗ get_unique_job_ids: 結果が異なります")
        print(f"  Python: {py_sorted}")
        print(f"  C:      {c_sorted}")


def test_calculate_makespan_compatibility():
    """calculate_makespanの互換性テスト"""
    print("\n=== calculate_makespan互換性テスト ===")
    
    H, W = 10, 50
    np.random.seed(42)
    
    # 同じ初期状態を作成
    window_matrix = np.random.randint(-1, 10, (H, W), dtype=np.int32)
    
    # Python実装
    py_makespan_onpre, py_makespan_cloud = calculate_makespan_batch_njit(
        window_matrix, window_matrix
    )
    py_makespan = max(py_makespan_onpre, py_makespan_cloud)
    
    # C言語実装
    c_makespan = c_calculate_makespan(window_matrix, H, W)
    
    # 結果を比較
    if py_makespan == c_makespan:
        print("✓ calculate_makespan: 完全に同じ結果")
        print(f"  makespan: {py_makespan}")
    else:
        print("✗ calculate_makespan: 結果が異なります")
        print(f"  Python: {py_makespan}")
        print(f"  C:      {c_makespan}")


def main():
    """メイン関数"""
    print("=" * 60)
    print("既存のPython実装とC言語実装の完全な互換性テスト")
    print("=" * 60)
    
    try:
        test_time_transition_compatibility()
        test_find_allocation_position_compatibility()
        test_get_unique_job_ids_compatibility()
        test_calculate_makespan_compatibility()
        
        print("\n" + "=" * 60)
        print("✓ すべての互換性テストが完了しました")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

