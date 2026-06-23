#!/usr/bin/env python3
"""
MMケースの代表点を詳細に分析するスクリプト
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from extract_representative_points import process_folder, find_representative_points, extract_pareto_front, parse_solutions_file
import math

def analyze_mm_case():
    """MMケース（execution_nsga_128-256）を詳細に分析"""
    base_dir = PROJECT_ROOT / 'CANDAR_resdata'
    mm_folder = base_dir / 'execution_nsga_128-256'
    
    if not mm_folder.exists():
        print(f"エラー: {mm_folder} が見つかりません")
        return
    
    # 最終世代のファイルを取得
    solution_files = sorted(mm_folder.glob('solutions_generation_*.txt'))
    if not solution_files:
        print(f"エラー: {mm_folder} にsolutions_generation_*.txtファイルが見つかりません")
        return
    
    final_file = solution_files[-1]
    print(f"分析対象ファイル: {final_file}")
    print("="*80)
    
    # データを読み込み
    costs, waiting_times = parse_solutions_file(final_file)
    print(f"読み込んだ解の総数: {len(costs)}")
    
    # パレートフロントを抽出
    pareto_costs, pareto_waiting_times = extract_pareto_front(costs, waiting_times)
    print(f"パレート最適解の数: {len(pareto_costs)}")
    
    # 代表点を抽出
    cost_min_point, balance_point, wt_min_point = find_representative_points(
        pareto_costs, pareto_waiting_times
    )
    
    print("\n" + "="*80)
    print("MMケース（execution_nsga_128-256）の代表点")
    print("="*80)
    
    print(f"\n1. 総クラウドコスト最小点:")
    print(f"   待ち時間: {cost_min_point[1]:.2f}")
    print(f"   総クラウドコスト: {cost_min_point[0]:.2f}")
    
    print(f"\n2. バランスの取れた点:")
    print(f"   待ち時間: {balance_point[1]:.2f}")
    print(f"   総クラウドコスト: {balance_point[0]:.2f}")
    
    print(f"\n3. 待ち時間最小点:")
    print(f"   待ち時間: {wt_min_point[1]:.2f}")
    print(f"   総クラウドコスト: {wt_min_point[0]:.2f}")
    
    # LaTeX表用の出力
    print("\n" + "="*80)
    print("LaTeX表用の値")
    print("="*80)
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{PFから抽出した代表ポリシ．}")
    print("\\label{tab:representative}")
    print("\\begin{tabular}{l l l r r}")
    print("\\toprule")
    print("解の種類 & 待ち時間 & 総クラウドコスト \\\\")
    print("\\midrule")
    print(f"総クラウドコスト最小点 & {cost_min_point[1]:.2f} & {cost_min_point[0]:.2f} \\\\")
    print(f"バランスの取れた点     & {balance_point[1]:.2f} & {balance_point[0]:.2f} \\\\")
    print(f"待ち時間最小点 & {wt_min_point[1]:.2f} & {wt_min_point[0]:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 検証: パレートフロントの範囲を表示
    print("\n" + "="*80)
    print("パレートフロントの範囲（検証用）")
    print("="*80)
    print(f"コスト範囲: {min(pareto_costs):.2f} ～ {max(pareto_costs):.2f}")
    print(f"待ち時間範囲: {min(pareto_waiting_times):.2f} ～ {max(pareto_waiting_times):.2f}")
    
    # バランス点の計算過程を表示
    print("\n" + "="*80)
    print("バランス点の計算過程（検証用）")
    print("="*80)
    cost_min = min(pareto_costs)
    cost_max = max(pareto_costs)
    wt_min = min(pareto_waiting_times)
    wt_max = max(pareto_waiting_times)
    
    cost_range = cost_max - cost_min if cost_max != cost_min else 1.0
    wt_range = wt_max - wt_min if wt_max != wt_min else 1.0
    
    print(f"コスト範囲: {cost_min:.2f} ～ {cost_max:.2f} (範囲: {cost_range:.2f})")
    print(f"待ち時間範囲: {wt_min:.2f} ～ {wt_max:.2f} (範囲: {wt_range:.2f})")
    
    # バランス点の正規化値を計算
    balance_normalized_cost = (balance_point[0] - cost_min) / cost_range
    balance_normalized_wt = (balance_point[1] - wt_min) / wt_range
    balance_distance = math.sqrt(balance_normalized_cost**2 + balance_normalized_wt**2)
    
    print(f"\nバランス点の正規化値:")
    print(f"  正規化コスト: {balance_normalized_cost:.4f}")
    print(f"  正規化待ち時間: {balance_normalized_wt:.4f}")
    print(f"  理想点(0,0)からの距離: {balance_distance:.4f}")
    
    # 他の代表点との比較
    print("\n" + "="*80)
    print("代表点間の比較")
    print("="*80)
    print(f"コスト最小点 → バランス点:")
    print(f"  コスト増加: {balance_point[0] - cost_min_point[0]:.2f} ({((balance_point[0] - cost_min_point[0]) / cost_min_point[0] * 100):.1f}%)")
    print(f"  待ち時間削減: {cost_min_point[1] - balance_point[1]:.2f} ({((cost_min_point[1] - balance_point[1]) / cost_min_point[1] * 100):.1f}%)")
    
    print(f"\nバランス点 → 待ち時間最小点:")
    print(f"  コスト増加: {wt_min_point[0] - balance_point[0]:.2f} ({((wt_min_point[0] - balance_point[0]) / balance_point[0] * 100):.1f}%)")
    print(f"  待ち時間削減: {balance_point[1] - wt_min_point[1]:.2f} ({((balance_point[1] - wt_min_point[1]) / balance_point[1] * 100):.1f}%)")

if __name__ == '__main__':
    analyze_mm_case()
