#!/usr/bin/env python3
"""
最小限のプロファイルを作成（SnakeViz用に最適化）
プロジェクト内の関数のみを残し、小さい関数を除外
"""
import pstats
import sys

if len(sys.argv) < 2:
    print("使用法: python create_minimal_profile.py <入力プロファイル> [出力プロファイル]")
    print("例: python create_minimal_profile.py profile_20jobs.prof profile_minimal.prof")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.prof', '_minimal.prof')

print(f"プロファイルを読み込み中: {input_file}")
stats = pstats.Stats(input_file)
stats.strip_dirs()
stats.sort_stats('cumulative')

# プロジェクト関連のモジュール名
project_keywords = [
    'scheduling_env',
    'heuristic_agent', 
    'job_generator',
    'test_large_scale_timing'
]

# 除外するモジュール（外部ライブラリ）
exclude_keywords = [
    'numba/core',
    'numpy/',
    'sklearn',
    'gym/',
    'collections/',
    'yaml',
    'argparse',
    'typing',
    'dispatcher.py',
    'compiler.py',
    'compiler_lock.py',
    'compiler_machinery.py',
    'importlib',  # import関連を除外
    'frozen importlib',
    'base.py',  # numbaのbase.py
    'typed_passes.py',
    'cpu.py',
    'ffi.py',
    '<built-in',  # 組み込み関数
    '<frozen',  # frozenモジュール
]

print("フィルタリング中...")
minimal_stats = pstats.Stats()

# 関数をフィルタリング
included_count = 0
excluded_count = 0
total_cumulative = 0
included_cumulative = 0

for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
    total_cumulative += ct
    filename = func_name[0]
    
    # プロジェクト関連かどうか確認
    is_project_file = any(keyword in filename for keyword in project_keywords)
    
    # 除外対象かどうか確認
    is_excluded = any(keyword in filename for keyword in exclude_keywords)
    
    # 組み込み関数やメソッドを除外
    is_builtin = filename.startswith('{') or filename.startswith('<')
    
    # フィルタリング条件：
    # 1. プロジェクトファイルである、かつ
    # 2. 組み込み関数ではない、かつ
    # 3. 除外対象ではない
    if not is_excluded and not is_builtin and is_project_file:
        minimal_stats.stats[func_name] = (cc, nc, tt, ct, callers)
        included_count += 1
        included_cumulative += ct
    else:
        excluded_count += 1

print(f"元の関数数: {len(stats.stats)}")
print(f"フィルタリング後: {included_count} 関数")
print(f"除外: {excluded_count} 関数")
print(f"累積時間の保持率: {included_cumulative/total_cumulative*100:.1f}%")

minimal_stats.dump_stats(output_file)
print(f"\n最小限のプロファイルを保存しました: {output_file}")
print(f"ファイルサイズを確認してください。")
print(f"\n可視化するには: snakeviz {output_file}")

