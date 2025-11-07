#!/usr/bin/env python3
"""
プロファイルをテキストベースで表示（SnakeVizの代替）
"""
import pstats
import sys

if len(sys.argv) < 2:
    print("使用法: python view_profile_simple.py <プロファイルファイル>")
    sys.exit(1)

profile_file = sys.argv[1]
stats = pstats.Stats(profile_file)
stats.strip_dirs()

# プロジェクト内のモジュールのみを表示
stats.sort_stats('cumulative')
print("=" * 80)
print("プロファイル統計 - プロジェクト内の関数のみ")
print("=" * 80)
stats.print_stats('scheduling_env', 'heuristic_agent', 'test_large_scale_timing')

print("\n" + "=" * 80)
print("呼び出し元統計 (どの関数がどの関数を呼んでいるか)")
print("=" * 80)
for func_name in sorted(stats.stats.keys()):
    filename = func_name[0]
    if any(kw in filename for kw in ['scheduling_env', 'heuristic_agent', 'test_large_scale_timing']):
        cc, nc, tt, ct, callers = stats.stats[func_name]
        if callers:
            print(f"\n{func_name[2]}:{func_name[1]} (累積時間: {ct:.4f}秒)")
            for caller, count in list(callers.items())[:5]:
                if any(kw in caller[0] for kw in ['scheduling_env', 'heuristic_agent', 'test_large_scale_timing']):
                    print(f"  ← {caller[2]}:{caller[1]} ({count}回)")










