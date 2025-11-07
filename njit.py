import numpy as np
import time
from numba import njit

# 二重forループの関数（Numbaなし）
def double_loop_python(n):
    """通常のPythonでの二重forループ"""
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j
    return result

# 二重forループの関数（Numbaあり、キャッシュなし）
@njit
def double_loop_numba(n):
    """Numbaでコンパイルされた二重forループ（キャッシュなし）"""
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j
    return result

# 二重forループの関数（Numbaあり、キャッシュあり）
@njit(cache=True)
def double_loop_numba_cached(n):
    """Numbaでコンパイルされた二重forループ（キャッシュあり）"""
    result = 0
    for i in range(n):
        for j in range(n):
            result += i * j
    return result

def benchmark_functions():
    """実行速度を比較する関数"""
    # テスト用のサイズ
    sizes = [100, 500, 1000, 2000]
    
    print("二重forループの実行速度比較")
    print("=" * 50)
    print(f"{'サイズ':<8} {'Python (秒)':<15} {'Numba (秒)':<15} {'高速化倍率':<10}")
    print("-" * 50)
    
    for n in sizes:
        # Python版の実行時間測定
        start_time = time.time()
        result_python = double_loop_python(n)
        python_time = time.time() - start_time
        
        # Numba版の実行時間測定（初回実行でコンパイル時間も含む）
        start_time = time.time()
        result_numba = double_loop_numba(n)
        numba_time = time.time() - start_time
        
        # 結果が同じかチェック
        if result_python != result_numba:
            print(f"警告: サイズ {n} で結果が異なります!")
            print(f"  Python: {result_python}")
            print(f"  Numba:  {result_numba}")
        
        # 高速化倍率を計算
        speedup = python_time / numba_time if numba_time > 0 else float('inf')
        
        print(f"{n:<8} {python_time:<15.6f} {numba_time:<15.6f} {speedup:<10.2f}x")
    
    print("\n注意: 初回実行時はNumbaでコンパイル時間が含まれます")
    print("より正確な比較のため、2回目以降の実行時間も測定します...")
    
    # 2回目の実行（コンパイル済み）
    print("\n2回目の実行（コンパイル済み）:")
    print("-" * 30)
    
    n = 2000
    # Python版
    start_time = time.time()
    result_python = double_loop_python(n)
    python_time_2 = time.time() - start_time
    
    # Numba版（2回目、キャッシュなし）
    start_time = time.time()
    result_numba = double_loop_numba(n)
    numba_time_2 = time.time() - start_time
    
    # Numba版（キャッシュあり、初回実行でコンパイル）
    start_time = time.time()
    result_numba_cached = double_loop_numba_cached(n)
    numba_cached_time_1 = time.time() - start_time
    
    # Numba版（キャッシュあり、2回目 - キャッシュから読み込み）
    start_time = time.time()
    result_numba_cached_2 = double_loop_numba_cached(n)
    numba_cached_time_2 = time.time() - start_time
    
    speedup_2 = python_time_2 / numba_time_2 if numba_time_2 > 0 else float('inf')
    
    print(f"サイズ {n}:")
    print(f"  Python: {python_time_2:.6f} 秒")
    print(f"  Numba (キャッシュなし): {numba_time_2:.6f} 秒")
    print(f"  Numba (キャッシュあり、初回): {numba_cached_time_1:.6f} 秒")
    print(f"  Numba (キャッシュあり、2回目): {numba_cached_time_2:.6f} 秒")
    print(f"  高速化倍率: {speedup_2:.2f}x")
    print(f"\nキャッシュの効果:")
    print(f"  初回実行時: コンパイル時間 = {numba_cached_time_1:.6f} 秒")
    print(f"  2回目実行時: キャッシュ読み込み = {numba_cached_time_2:.6f} 秒")
    if numba_cached_time_1 > 0:
        cache_speedup = numba_cached_time_1 / numba_cached_time_2 if numba_cached_time_2 > 0 else float('inf')
        print(f"  キャッシュによる高速化: {cache_speedup:.2f}x (次回実行時)")
    
    print("\n説明:")
    print("  - cache=True を使うと、コンパイル済みコードが ~/.cache/numba/ に保存されます")
    print("  - 次回実行時、同じシグネチャ・同じコードであればキャッシュから読み込まれます")
    print("  - プログラムを終了しても、キャッシュは保持されるため、次回の起動時に再利用できます")

if __name__ == "__main__":
    benchmark_functions()
