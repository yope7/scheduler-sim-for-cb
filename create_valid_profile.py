#!/usr/bin/env python3
"""
プロファイルファイルを修正してSnakeVizで使用可能にする
呼び出し関係（callers/callees）も適切に処理
"""
import pstats
import sys

if len(sys.argv) < 2:
    print("使用法: python create_valid_profile.py <入力プロファイル> [出力プロファイル]")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.prof', '_valid.prof')

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

print("関数をフィルタリング中...")

# まず、保持する関数を決定
valid_functions = {}
for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
    filename = func_name[0]
    
    # プロジェクトファイルかどうか
    is_project_file = any(keyword in filename for keyword in project_keywords)
    
    # 組み込み関数や除外対象かどうか
    is_builtin = filename.startswith('{') or filename.startswith('<')
    is_excluded = any(keyword in filename for keyword in [
        'numba/core', 'numpy/', 'sklearn', 'gym/', 'collections/', 
        'yaml', 'argparse', 'typing', 'importlib', 'frozen importlib',
        'dispatcher.py', 'compiler.py', 'compiler_lock.py', 'compiler_machinery.py',
        'base.py', 'typed_passes.py', 'cpu.py', 'ffi.py'
    ])
    
    # プロジェクトファイルのみを保持
    if is_project_file and not is_builtin and not is_excluded:
        valid_functions[func_name] = (cc, nc, tt, ct, callers)

print(f"保持する関数数: {len(valid_functions)}")

# 新しいプロファイルを作成（呼び出し関係も修正）
new_stats = pstats.Stats()
for func_name, (cc, nc, tt, ct, callers) in valid_functions.items():
    # 呼び出し元（callers）もフィルタリング
    filtered_callers = {}
    for caller_name, call_count in callers.items():
        if caller_name in valid_functions:
            filtered_callers[caller_name] = call_count
    
    # 新しい統計データを作成
    new_stats.stats[func_name] = (cc, nc, tt, ct, filtered_callers)

# 保存
new_stats.dump_stats(output_file)
print(f"修正済みプロファイルを保存しました: {output_file}")
print(f"可視化するには: snakeviz {output_file}")

# 統計を表示
new_stats.sort_stats('cumulative')
print("\n=== プロファイル内容 ===")
print(f"関数数: {len(new_stats.stats)}")
new_stats.print_stats(15)










