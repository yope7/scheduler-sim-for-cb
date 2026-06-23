#!/usr/bin/env python3
"""
C言語実装が実際に使用されているか確認するスクリプト（プロファイリング結果の確認）
"""
import sys
import os
import cProfile
import pstats
import io

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from test_large_scale_timing_c import run_environment_timing_test

def verify_c_usage():
    """C言語実装が実際に使用されているか確認"""
    print("="*60)
    print("C言語実装の使用確認（プロファイリング）")
    print("="*60)
    
    # プロファイリングを実行
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        results = run_environment_timing_test(
            50,  # 小さいジョブ数でテスト
            use_heuristic=True,
            seed=0,
            nb_steps=1,
            nb_episodes=0
        )
    finally:
        profiler.disable()
    
    # プロファイリング結果を分析
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(50)
    
    output = s.getvalue()
    
    # C言語実装の関数が呼ばれているか確認
    print("\n--- C言語実装の関数呼び出し確認 ---")
    
    c_functions = [
        'scheduling_env_core',
        'find_allocation_position',
        'time_transition',
        'do_schedule',
        'WindowCache',
    ]
    
    found_c_functions = []
    for func in c_functions:
        if func in output:
            found_c_functions.append(func)
            print(f"  ✓ {func} が呼ばれています")
        else:
            print(f"  ✗ {func} が見つかりません")
    
    # Python実装の関数が呼ばれているか確認
    print("\n--- Python実装の関数呼び出し確認 ---")
    
    python_functions = [
        'scheduling_env.py:759(_rebuild_cache_if_needed)',
        'scheduling_env.py:1042(find_allocation_position)',
        'scheduling_env.py:544(time_transition)',
    ]
    
    found_python_functions = []
    for func in python_functions:
        if func in output:
            found_python_functions.append(func)
            print(f"  ⚠ {func} が呼ばれています（これは問題の可能性があります）")
        else:
            print(f"  ✓ {func} は呼ばれていません（期待通り）")
    
    # 結果の要約
    print("\n--- 結果の要約 ---")
    if found_c_functions:
        print(f"  ✓ C言語実装の関数が {len(found_c_functions)} 個見つかりました")
    else:
        print("  ✗ C言語実装の関数が見つかりませんでした")
    
    if found_python_functions:
        print(f"  ⚠ Python実装の関数が {len(found_python_functions)} 個見つかりました")
        print("     これは、親クラスのメソッドが呼ばれている可能性があります")
    else:
        print("  ✓ Python実装の関数は呼ばれていません（期待通り）")
    
    # 詳細なプロファイリング結果を表示
    print("\n--- プロファイリング結果（Top 20） ---")
    print(output[:5000])  # 最初の5000文字を表示
    
    return found_c_functions, found_python_functions

if __name__ == "__main__":
    verify_c_usage()

