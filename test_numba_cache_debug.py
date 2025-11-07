#!/usr/bin/env python3
"""
Numbaキャッシュデバッグテストスクリプト

環境変数NUMBA_DEBUG_CACHE=1を設定して実行すると、
キャッシュのHIT/MISSがログに出力されます。
"""

import os
import sys
import numpy as np

# デバッグキャッシュを有効化
os.environ['NUMBA_DEBUG_CACHE'] = '1'

from src.envs.scheduling_env import (
    get_unique_job_ids_njit,
    calculate_makespan_batch_njit,
    time_transition_njit
)

def test_numba_cache():
    """Numbaキャッシュの動作をテスト"""
    print("=" * 60)
    print("Numbaキャッシュデバッグテスト")
    print("=" * 60)
    print()
    
    # テスト1: 同じshapeで複数回呼び出す
    print("テスト1: 同じshapeで複数回呼び出し")
    print("-" * 60)
    
    matrix1 = np.arange(20, dtype=np.int32).reshape(4, 5)
    matrix2 = np.arange(20, dtype=np.int32).reshape(4, 5) + 100
    
    print("1回目:")
    result1 = get_unique_job_ids_njit(matrix1, 50000)
    print(f"  結果shape: {result1.shape}")
    print()
    
    print("2回目（同じshape）:")
    result2 = get_unique_job_ids_njit(matrix2, 50000)
    print(f"  結果shape: {result2.shape}")
    print()
    
    # テスト2: 異なるshapeで呼び出す
    print("テスト2: 異なるshapeで呼び出し")
    print("-" * 60)
    
    matrix3 = np.arange(30, dtype=np.int32).reshape(5, 6)
    
    print("3回目（異なるshape）:")
    result3 = get_unique_job_ids_njit(matrix3, 50000)
    print(f"  結果shape: {result3.shape}")
    print()
    
    # テスト3: calculate_makespan_batch_njit
    print("テスト3: calculate_makespan_batch_njit")
    print("-" * 60)
    
    onpre1 = np.full((4, 5), -1, dtype=np.int32)
    cloud1 = np.full((3, 5), -1, dtype=np.int32)
    
    print("1回目:")
    makespan1 = calculate_makespan_batch_njit(onpre1, cloud1)
    print(f"  makespan: {makespan1}")
    print()
    
    print("2回目（同じshape）:")
    makespan2 = calculate_makespan_batch_njit(onpre1, cloud1)
    print(f"  makespan: {makespan2}")
    print()
    
    print("テスト完了")
    print()
    print("注意: NUMBA_DEBUG_CACHE=1を設定しているので、")
    print("キャッシュのHIT/MISSがログに出力されているはずです。")

if __name__ == "__main__":
    test_numba_cache()

