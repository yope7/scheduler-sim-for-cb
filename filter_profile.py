#!/usr/bin/env python3
"""
プロファイルファイルをフィルタリングしてSnakeVizで扱いやすくする
"""
import pstats
import sys

if len(sys.argv) < 2:
    print("使用法: python filter_profile.py <入力プロファイル> [出力プロファイル]")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.prof', '_filtered.prof')

print(f"プロファイルを読み込み中: {input_file}")
stats = pstats.Stats(input_file)
stats.strip_dirs()

# プロジェクト内のモジュール（src/で始まる）のみを保持
project_modules = ['scheduling_env', 'heuristic_agent', 'job_generator', 'test_large_scale_timing']

# 統計をフィルタリング
filtered = pstats.Stats()
for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
    # プロジェクト内のモジュールまたは累積時間が大きいものを保持
    if any(mod in func_name[0] for mod in project_modules) or ct >= 0.001:
        filtered.stats[func_name] = (cc, nc, tt, ct, callers)

filtered.dump_stats(output_file)
print(f"フィルタリング済みプロファイルを保存しました: {output_file}")
print(f"可視化するには: snakeviz {output_file}")










