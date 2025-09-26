#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽出したデータから各世代のHV（Hypervolume）を計算するスクリプト
ref point: (20000, 200)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォントの設定
rcParams['font.family'] = 'DejaVu Sans'

def calculate_hypervolume(points, reference_point):
    """
    2次元の点群からHypervolumeを計算
    
    Args:
        points (list): [(cost, waiting_time), ...] の形式の点のリスト
        reference_point (tuple): (cost, waiting_time) の参照点
    
    Returns:
        float: Hypervolume値
    """
    if not points:
        return 0.0
    
    # パレートフロントを計算（支配されない解のみを抽出）
    pareto_front = get_pareto_front(points)
    
    if not pareto_front:
        return 0.0
    
    # パレートフロントの点をコストでソート
    pareto_front.sort(key=lambda x: x[0])
    
    # Hypervolumeを計算
    hv = 0.0
    prev_cost = reference_point[0]
    
    for cost, waiting_time in pareto_front:
        # この点が支配する長方形の面積を計算
        area = (prev_cost - cost) * (reference_point[1] - waiting_time)
        hv += area
        prev_cost = cost
    
    return hv

def get_pareto_front(points):
    """
    パレートフロント（支配されない解）を計算
    
    Args:
        points (list): [(cost, waiting_time), ...] の形式の点のリスト
    
    Returns:
        list: パレートフロントの点のリスト
    """
    pareto_front = []
    
    for i, point in enumerate(points):
        dominated = False
        
        for j, other_point in enumerate(points):
            if i != j:
                # other_pointがpointを支配しているかチェック
                if (other_point[0] <= point[0] and other_point[1] <= point[1] and 
                    (other_point[0] < point[0] or other_point[1] < point[1])):
                    dominated = True
                    break
        
        if not dominated:
            pareto_front.append(point)
    
    return pareto_front

def load_extracted_data(filename):
    """
    抽出されたデータを読み込み
    
    Args:
        filename (str): JSONファイル名
    
    Returns:
        dict: 世代ごとの解のデータ
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_all_hv(data, reference_point):
    """
    全世代のHVを計算
    
    Args:
        data (dict): 世代ごとの解のデータ
        reference_point (tuple): 参照点
    
    Returns:
        dict: 世代ごとのHV値
    """
    hv_results = {}
    
    for generation in sorted(data.keys()):
        solutions = data[generation]
        
        # コストと待ち時間のペアを作成
        points = [(sol['cost'], sol['waiting_time']) for sol in solutions]
        
        # HVを計算
        hv = calculate_hypervolume(points, reference_point)
        hv_results[generation] = hv
        
        print(f"世代 {generation}: HV = {hv:.2f}")
    
    return hv_results

def plot_generation_vs_hv(hv_results, output_file="generation_vs_hv.png"):
    """
    世代とHVの関係をグラフ化
    
    Args:
        hv_results (dict): 世代ごとのHV値
        output_file (str): 出力ファイル名
    """
    generations = sorted(hv_results.keys())
    hv_values = [hv_results[gen] for gen in generations]
    
    plt.figure(figsize=(12, 8))
    plt.plot(generations, hv_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('世代', fontsize=14)
    plt.ylabel('Hypervolume', fontsize=14)
    plt.title('世代とHypervolumeの関係\n(Ref Point: Cost=20000, Waiting Time=200)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # データポイントにラベルを追加
    for i, (gen, hv) in enumerate(zip(generations, hv_values)):
        plt.annotate(f'{hv:.0f}', (gen, hv), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"グラフを {output_file} に保存しました")

def save_hv_results(hv_results, output_file):
    """
    HV計算結果を保存
    
    Args:
        hv_results (dict): 世代ごとのHV値
        output_file (str): 出力ファイル名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("世代\tHypervolume\n")
        f.write("----------------\n")
        for generation in sorted(hv_results.keys()):
            f.write(f"{generation}\t{hv_results[generation]:.2f}\n")
    
    print(f"HV計算結果を {output_file} に保存しました")

if __name__ == "__main__":
    # 参照点を設定
    REFERENCE_POINT = (20000, 200)
    print(f"参照点: Cost={REFERENCE_POINT[0]}, Waiting Time={REFERENCE_POINT[1]}")
    print("=" * 50)
    
    # 抽出されたデータを読み込み
    data = load_extracted_data("extracted_solutions_data.json")
    
    # 全世代のHVを計算
    print("各世代のHypervolumeを計算中...")
    hv_results = calculate_all_hv(data, REFERENCE_POINT)
    
    # 結果を保存
    save_hv_results(hv_results, "hv_results.txt")
    
    # グラフを作成
    print("\nグラフを作成中...")
    plot_generation_vs_hv(hv_results)
    
    # 統計情報を表示
    print("\n統計情報:")
    hv_values = list(hv_results.values())
    print(f"最小HV: {min(hv_values):.2f}")
    print(f"最大HV: {max(hv_values):.2f}")
    print(f"平均HV: {np.mean(hv_values):.2f}")
    print(f"標準偏差: {np.std(hv_values):.2f}") 