#!/usr/bin/env python3
"""
Numbaキャッシュ診断スクリプト

使い方:
    export NUMBA_DEBUG_CACHE=1
    python diagnose_numba_cache.py
"""

import os
import sys
import numpy as np

# 環境変数を設定（まだ設定されていない場合）
if 'NUMBA_DEBUG_CACHE' not in os.environ:
    os.environ['NUMBA_DEBUG_CACHE'] = '1'
    print("NUMBA_DEBUG_CACHE=1 を設定しました")

from numba import njit

# テスト用の関数を定義（scheduling_env.pyから）
@njit(cache=True, fastmath=True)
def test_function_1(matrix):
    """シンプルな関数 - キャッシュが効くはず"""
    result = np.zeros(matrix.shape[0], dtype=np.int32)
    for i in range(matrix.shape[0]):
        result[i] = matrix[i, 0] * 2
    return result

@njit(cache=True, fastmath=True)
def test_function_2(arr):
    """配列を受け取る関数"""
    return arr.sum()

def main():
    print("=" * 60)
    print("Numbaキャッシュ診断スクリプト")
    print("=" * 60)
    print()
    
    print("1. 初回実行（コンパイルされるはず）")
    print("-" * 60)
    arr1 = np.arange(10, dtype=np.int32).reshape(5, 2)
    result1 = test_function_1(arr1)
    print(f"結果: {result1}")
    print()
    
    print("2. 2回目実行（キャッシュから読み込まれるはず）")
    print("-" * 60)
    arr2 = np.arange(10, dtype=np.int32).reshape(5, 2)
    result2 = test_function_1(arr2)
    print(f"結果: {result2}")
    print()
    
    print("3. 異なるdtypeで実行（再コンパイルされるはず）")
    print("-" * 60)
    arr3 = np.arange(10, dtype=np.float32).reshape(5, 2)
    result3 = test_function_2(arr3.astype(np.int32))  # 型を変換
    print(f"結果: {result3}")
    print()
    
    print("診断完了")
    print()
    print("注意: NUMBA_DEBUG_CACHE=1 を設定しているので、")
    print("キャッシュのHIT/MISSがログに出力されているはずです。")
    print()
    print("キャッシュが効いていない場合:")
    print("  - ログに 'MISS' と理由が表示されます")
    print("  - ログに 'object mode' と表示されている場合、")
    print("    関数内でPythonオブジェクトを使用しています")

if __name__ == "__main__":
    main()

