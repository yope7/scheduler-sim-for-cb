#!/usr/bin/env python3
"""
PFから3つの代表点を抽出するスクリプト
- 総クラウドコスト最小点
- 待ち時間最小点
- バランスの取れた点（正規化後のユークリッド距離最小）
"""

import os
import glob
import math
from pathlib import Path

def parse_solutions_file(filepath):
    """solutions_generation_*.txtファイルを解析して、コストと待ち時間を抽出"""
    costs = []
    waiting_times = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # ヘッダー行をスキップ（最初の4行）
        for line in lines[4:]:
            line = line.strip()
            if not line or line.startswith('---') or line.startswith('総個体数') or line.startswith('保存時刻'):
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    cost = float(parts[1])
                    waiting_time = float(parts[2])
                    costs.append(cost)
                    waiting_times.append(waiting_time)
                except (ValueError, IndexError):
                    continue
    
    return costs, waiting_times

def extract_pareto_front(costs, waiting_times):
    """パレートフロントを抽出（両方最小化）"""
    if len(costs) == 0:
        return [], []
    
    # パレート最適解のインデックスを取得
    pareto_indices = []
    
    for i in range(len(costs)):
        is_dominated = False
        for j in range(len(costs)):
            if i != j:
                # jがiを支配するかチェック（両方小さい）
                if (costs[j] <= costs[i] and waiting_times[j] <= waiting_times[i] and 
                    (costs[j] < costs[i] or waiting_times[j] < waiting_times[i])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_indices.append(i)
    
    pareto_costs = [costs[i] for i in pareto_indices]
    pareto_waiting_times = [waiting_times[i] for i in pareto_indices]
    
    # コストでソート
    sorted_pairs = sorted(zip(pareto_costs, pareto_waiting_times))
    pareto_costs = [p[0] for p in sorted_pairs]
    pareto_waiting_times = [p[1] for p in sorted_pairs]
    
    return pareto_costs, pareto_waiting_times

def find_representative_points(costs, waiting_times):
    """3つの代表点を抽出"""
    if len(costs) == 0:
        return None, None, None
    
    # 1. 総クラウドコスト最小点
    cost_min_idx = min(range(len(costs)), key=lambda i: costs[i])
    cost_min_point = (costs[cost_min_idx], waiting_times[cost_min_idx])
    
    # 2. 待ち時間最小点
    wt_min_idx = min(range(len(waiting_times)), key=lambda i: waiting_times[i])
    wt_min_point = (costs[wt_min_idx], waiting_times[wt_min_idx])
    
    # 3. バランスの取れた点（正規化後のユークリッド距離最小）
    # 正規化: 各目的の最小値と最大値を使用
    cost_min = min(costs)
    cost_max = max(costs)
    wt_min = min(waiting_times)
    wt_max = max(waiting_times)
    
    # 正規化範囲を計算（0除算を避ける）
    cost_range = cost_max - cost_min
    wt_range = wt_max - wt_min
    
    if cost_range == 0:
        cost_range = 1.0
    if wt_range == 0:
        wt_range = 1.0
    
    # 正規化された点
    normalized_costs = [(c - cost_min) / cost_range for c in costs]
    normalized_waiting_times = [(w - wt_min) / wt_range for w in waiting_times]
    
    # 理想点（正規化後の原点 (0, 0)）
    ideal_point = (0.0, 0.0)
    
    # 各点から理想点へのユークリッド距離を計算
    distances = []
    for i in range(len(normalized_costs)):
        dx = normalized_costs[i] - ideal_point[0]
        dy = normalized_waiting_times[i] - ideal_point[1]
        dist = math.sqrt(dx * dx + dy * dy)
        distances.append(dist)
    
    # 距離が最小の点
    balance_idx = min(range(len(distances)), key=lambda i: distances[i])
    balance_point = (costs[balance_idx], waiting_times[balance_idx])
    
    return cost_min_point, balance_point, wt_min_point

def process_folder(folder_path):
    """フォルダ内の最終世代ファイルを処理"""
    folder_path = Path(folder_path)
    
    # solutions_generation_*.txtファイルを検索
    solution_files = sorted(folder_path.glob('solutions_generation_*.txt'))
    
    if not solution_files:
        print(f"警告: {folder_path} にsolutions_generation_*.txtファイルが見つかりません")
        return None
    
    # 最終世代のファイルを使用
    final_file = solution_files[-1]
    print(f"\n処理中: {final_file}")
    
    # データを読み込み
    costs, waiting_times = parse_solutions_file(final_file)
    
    if len(costs) == 0:
        print(f"警告: {final_file} からデータを読み込めませんでした")
        return None
    
    print(f"  読み込んだ解の数: {len(costs)}")
    
    # パレートフロントを抽出
    pareto_costs, pareto_waiting_times = extract_pareto_front(costs, waiting_times)
    print(f"  パレート最適解の数: {len(pareto_costs)}")
    
    if len(pareto_costs) == 0:
        print(f"警告: パレートフロントが見つかりませんでした")
        return None
    
    # 代表点を抽出
    cost_min_point, balance_point, wt_min_point = find_representative_points(
        pareto_costs, pareto_waiting_times
    )
    
    return {
        'folder': str(folder_path),
        'total_solutions': len(costs),
        'pareto_solutions': len(pareto_costs),
        'cost_min': cost_min_point,
        'balance': balance_point,
        'wt_min': wt_min_point,
        'all_pareto_costs': pareto_costs,
        'all_pareto_waiting_times': pareto_waiting_times
    }

def main():
    """メイン処理"""
    base_dir = Path('/home/noguchi/scheduler-sim-for-cb/CANDAR_resdata')
    
    # NSGA-II実行フォルダを検索
    nsga_folders = sorted(base_dir.glob('execution_nsga_*'))
    
    print(f"見つかったNSGA-IIフォルダ数: {len(nsga_folders)}")
    
    results = {}
    
    for folder in nsga_folders:
        result = process_folder(folder)
        if result:
            results[folder.name] = result
    
    # 結果を表示
    print("\n" + "="*80)
    print("代表点抽出結果")
    print("="*80)
    
    for folder_name, result in results.items():
        print(f"\n【{folder_name}】")
        print(f"  総解数: {result['total_solutions']}")
        print(f"  パレート最適解数: {result['pareto_solutions']}")
        print(f"\n  代表点:")
        print(f"    総クラウドコスト最小点: コスト={result['cost_min'][0]:.2f}, 待ち時間={result['cost_min'][1]:.2f}")
        print(f"    バランスの取れた点:     コスト={result['balance'][0]:.2f}, 待ち時間={result['balance'][1]:.2f}")
        print(f"    待ち時間最小点:         コスト={result['wt_min'][0]:.2f}, 待ち時間={result['wt_min'][1]:.2f}")
    
    # MMケースを特定（おそらく128-256や32-512など）
    # ユーザーに確認するため、すべての結果を表示
    print("\n" + "="*80)
    print("LaTeX表用の値（MMケースを特定してください）")
    print("="*80)
    
    # 最も可能性の高いケース（128-256）を最初に表示
    if 'execution_nsga_128-256' in results:
        result = results['execution_nsga_128-256']
        print(f"\n【execution_nsga_128-256（MMケース候補）】")
        print(f"総クラウドコスト最小点 & {result['cost_min'][1]:.2f} & {result['cost_min'][0]:.2f} \\\\")
        print(f"バランスの取れた点     & {result['balance'][1]:.2f} & {result['balance'][0]:.2f} \\\\")
        print(f"待ち時間最小点 & {result['wt_min'][1]:.2f} & {result['wt_min'][0]:.2f} \\\\")
    
    return results

if __name__ == '__main__':
    results = main()

