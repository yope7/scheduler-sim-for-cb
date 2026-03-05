#!/usr/bin/env python3
"""
PCNの結果から3つの代表点を抽出するスクリプト
- 総クラウドコスト最小点
- 待ち時間最小点
- バランスの取れた点（正規化後のユークリッド距離最小）
"""

import re
import math
from pathlib import Path

def parse_pcn_pareto_file(filepath):
    """PCNのpareto_front_details_*.txtファイルを解析して、コストと待ち時間を抽出"""
    costs = []
    waiting_times = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 実数値空間のパレートフロントセクションを探す
        match = re.search(r'=== 最終実数値空間のパレートフロント ===\n非支配解数: (\d+)\n(.*?)(?=\n===|\Z)', content, re.DOTALL)
        
        if not match:
            print(f"警告: {filepath} から実数値空間のパレートフロントが見つかりません")
            return [], []
        
        solutions_text = match.group(2)
        
        # 各行から解を抽出 (形式: 解1: [コスト, 待ち時間])
        for line in solutions_text.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('解'):
                continue
            
            # 解X: [コスト, 待ち時間] の形式をパース
            match_solution = re.search(r'解\d+:\s*\[([\d.]+),\s*([\d.]+)\]', line)
            if match_solution:
                cost = float(match_solution.group(1))
                waiting_time = float(match_solution.group(2))
                costs.append(cost)
                waiting_times.append(waiting_time)
    
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

def process_pcn_folder(folder_path):
    """PCNフォルダ内の最終結果ファイルを処理"""
    folder_path = Path(folder_path)
    
    # finalフォルダ内のpareto_front_details_*.txtファイルを検索
    final_dir = folder_path / 'final'
    if not final_dir.exists():
        print(f"警告: {folder_path} にfinalフォルダが見つかりません")
        return None
    
    detail_files = sorted(final_dir.glob('pareto_front_details_*.txt'))
    
    if not detail_files:
        print(f"警告: {final_dir} にpareto_front_details_*.txtファイルが見つかりません")
        return None
    
    # 最新のファイルを使用
    final_file = detail_files[-1]
    print(f"\n処理中: {final_file}")
    
    # データを読み込み
    costs, waiting_times = parse_pcn_pareto_file(final_file)
    
    if len(costs) == 0:
        print(f"警告: {final_file} からデータを読み込めませんでした")
        return None
    
    print(f"  読み込んだ解の数: {len(costs)}")
    
    # パレートフロントを抽出（既にパレート最適解として保存されているが、念のため再計算）
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

def analyze_pcn_mm_case():
    """PCNのMMケース（execution_pcn_128-256）を詳細に分析"""
    base_dir = Path('/home/noguchi/scheduler-sim-for-cb/CANDAR_resdata')
    mm_folder = base_dir / 'execution_pcn_128-256'
    
    result = process_pcn_folder(mm_folder)
    
    if not result:
        print("エラー: PCNのMMケースのデータを読み込めませんでした")
        return
    
    print("\n" + "="*80)
    print("PCN MMケース（execution_pcn_128-256）の代表点")
    print("="*80)
    
    print(f"\n1. 総クラウドコスト最小点:")
    print(f"   待ち時間: {result['cost_min'][1]:.2f}")
    print(f"   総クラウドコスト: {result['cost_min'][0]:.2f}")
    
    print(f"\n2. バランスの取れた点:")
    print(f"   待ち時間: {result['balance'][1]:.2f}")
    print(f"   総クラウドコスト: {result['balance'][0]:.2f}")
    
    print(f"\n3. 待ち時間最小点:")
    print(f"   待ち時間: {result['wt_min'][1]:.2f}")
    print(f"   総クラウドコスト: {result['wt_min'][0]:.2f}")
    
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
    print(f"総クラウドコスト最小点 & {result['cost_min'][1]:.2f} & {result['cost_min'][0]:.2f} \\\\")
    print(f"バランスの取れた点     & {result['balance'][1]:.2f} & {result['balance'][0]:.2f} \\\\")
    print(f"待ち時間最小点 & {result['wt_min'][1]:.2f} & {result['wt_min'][0]:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 検証: パレートフロントの範囲を表示
    print("\n" + "="*80)
    print("パレートフロントの範囲（検証用）")
    print("="*80)
    print(f"コスト範囲: {min(result['all_pareto_costs']):.2f} ～ {max(result['all_pareto_costs']):.2f}")
    print(f"待ち時間範囲: {min(result['all_pareto_waiting_times']):.2f} ～ {max(result['all_pareto_waiting_times']):.2f}")
    
    # バランス点の計算過程を表示
    print("\n" + "="*80)
    print("バランス点の計算過程（検証用）")
    print("="*80)
    cost_min = min(result['all_pareto_costs'])
    cost_max = max(result['all_pareto_costs'])
    wt_min = min(result['all_pareto_waiting_times'])
    wt_max = max(result['all_pareto_waiting_times'])
    
    cost_range = cost_max - cost_min if cost_max != cost_min else 1.0
    wt_range = wt_max - wt_min if wt_max != wt_min else 1.0
    
    print(f"コスト範囲: {cost_min:.2f} ～ {cost_max:.2f} (範囲: {cost_range:.2f})")
    print(f"待ち時間範囲: {wt_min:.2f} ～ {wt_max:.2f} (範囲: {wt_range:.2f})")
    
    # バランス点の正規化値を計算
    balance_normalized_cost = (result['balance'][0] - cost_min) / cost_range
    balance_normalized_wt = (result['balance'][1] - wt_min) / wt_range
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
    print(f"  コスト増加: {result['balance'][0] - result['cost_min'][0]:.2f} ({((result['balance'][0] - result['cost_min'][0]) / result['cost_min'][0] * 100):.1f}%)")
    print(f"  待ち時間削減: {result['cost_min'][1] - result['balance'][1]:.2f} ({((result['cost_min'][1] - result['balance'][1]) / result['cost_min'][1] * 100):.1f}%)")
    
    print(f"\nバランス点 → 待ち時間最小点:")
    print(f"  コスト増加: {result['wt_min'][0] - result['balance'][0]:.2f} ({((result['wt_min'][0] - result['balance'][0]) / result['balance'][0] * 100):.1f}%)")
    print(f"  待ち時間削減: {result['balance'][1] - result['wt_min'][1]:.2f} ({((result['balance'][1] - result['wt_min'][1]) / result['balance'][1] * 100):.1f}%)")

def main():
    """メイン処理"""
    base_dir = Path('/home/noguchi/scheduler-sim-for-cb/CANDAR_resdata')
    
    # PCN実行フォルダを検索
    pcn_folders = sorted(base_dir.glob('execution_pcn_*'))
    
    print(f"見つかったPCNフォルダ数: {len(pcn_folders)}")
    
    results = {}
    
    for folder in pcn_folders:
        result = process_pcn_folder(folder)
        if result:
            results[folder.name] = result
    
    # 結果を表示
    print("\n" + "="*80)
    print("PCN代表点抽出結果")
    print("="*80)
    
    for folder_name, result in results.items():
        print(f"\n【{folder_name}】")
        print(f"  総解数: {result['total_solutions']}")
        print(f"  パレート最適解数: {result['pareto_solutions']}")
        print(f"\n  代表点:")
        print(f"    総クラウドコスト最小点: コスト={result['cost_min'][0]:.2f}, 待ち時間={result['cost_min'][1]:.2f}")
        print(f"    バランスの取れた点:     コスト={result['balance'][0]:.2f}, 待ち時間={result['balance'][1]:.2f}")
        print(f"    待ち時間最小点:         コスト={result['wt_min'][0]:.2f}, 待ち時間={result['wt_min'][1]:.2f}")
    
    # MMケースを詳細に分析
    if 'execution_pcn_128-256' in results:
        print("\n" + "="*80)
        analyze_pcn_mm_case()
    
    return results

if __name__ == '__main__':
    results = main()

