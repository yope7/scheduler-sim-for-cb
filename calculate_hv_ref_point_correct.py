#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正しいハイパーボリューム（HV）計算
morl_baselinesライブラリを使用してリファレンスポイント（20万、200）でのHV計算
指定されたファイルからデータを抽出→np.arrayに変換→HV計算→結果出力
"""

import numpy as np
import re
from typing import Tuple, List, Optional
from morl_baselines.common.performance_indicators import hypervolume


def extract_pareto_data_from_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定されたファイルからパレート最適解のデータを抽出してnumpy配列に変換
    
    Args:
        file_path (str): データファイルのパス
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (コスト配列, 待ち時間配列)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # パレート最適解詳細セクションを抽出
        pareto_section_match = re.search(
            r'パレート最適解詳細:(.*?)(?=\n\n|\Z)', 
            content, 
            re.DOTALL
        )
        
        if not pareto_section_match:
            raise ValueError("パレート最適解詳細セクションが見つかりません")
        
        pareto_section = pareto_section_match.group(1)
        
        # データ行を抽出（ヘッダー行を除く）
        lines = pareto_section.strip().split('\n')
        data_lines = []
        
        for line in lines:
            # データ行のパターン: 数字 + 空白 + 数値 + 空白 + 数値 + 空白 + 数値 + 空白 + 数値
            if re.match(r'^\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+', line):
                data_lines.append(line)
        
        if not data_lines:
            raise ValueError("データ行が見つかりません")
        
        # データを解析
        costs = []
        wait_times = []
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 5:
                # データ形式: "1    0.000        1.000      86957.00     32.94"
                cost = float(parts[3])  # 4番目が実際のコスト
                wait_time = float(parts[4])  # 5番目が実際の待ち時間
                costs.append(cost)
                wait_times.append(wait_time)
        
        # 重複を削除してユニークな点のみを残す
        points = list(set(zip(costs, wait_times)))
        if points:
            costs, wait_times = zip(*points)
        
        # numpy配列に変換
        costs_array = np.array(costs)
        wait_times_array = np.array(wait_times)
        
        print(f"抽出されたデータ数: {len(costs_array)}")
        if len(costs_array) > 0:
            print(f"コスト範囲: {costs_array.min():.2f} - {costs_array.max():.2f}")
            print(f"待ち時間範囲: {wait_times_array.min():.2f} - {wait_times_array.max():.2f}")
        
        return costs_array, wait_times_array
        
    except Exception as e:
        print(f"ファイル読み込みエラー: {e}")
        return np.array([]), np.array([])


def main():
    """
    メイン関数：データ抽出→HV計算→結果出力
    """
    # ファイルパス
    file_path = "distributed_pareto_results/pareto_details_distributed_20250818_160526.txt"
    
    # リファレンスポイント（コスト: 20万, 待ち時間: 200）
    reference_point = np.array([200000.0, 200.0])
    
    print("=" * 60)
    print("ハイパーボリューム計算（リファレンスポイント: [200000, 200]）")
    print("morl_baselinesライブラリを使用")
    print("=" * 60)
    
    # ステップ1: データ抽出
    print("\n1. データ抽出中...")
    costs, wait_times = extract_pareto_data_from_file(file_path)
    
    if len(costs) == 0:
        print("データの抽出に失敗しました。")
        return
    
    # ステップ2: numpy配列に変換（既に完了）
    print("\n2. numpy配列への変換完了")
    
    # ステップ3: HV計算
    print("\n3. ハイパーボリューム計算中...")
    
    # パレート点を2次元配列に変換
    pareto_points = np.column_stack([costs, wait_times])
    
    print(f"パレート点の形状: {pareto_points.shape}")
    print(f"最初の5点:")
    for i, point in enumerate(pareto_points[:5]):
        print(f"  点{i+1}: [コスト={point[0]:.2f}, 待ち時間={point[1]:.2f}]")
    
    # morl_baselinesのhypervolume関数を使用
    try:
        hv_value = hypervolume(pareto_points, reference_point)
        
        # ステップ4: 結果出力
        print("\n4. 結果出力")
        print("-" * 60)
        print(f"リファレンスポイント: [{reference_point[0]:,.0f}, {reference_point[1]:.0f}]")
        print(f"ユニークなパレート最適解数: {len(pareto_points)}")
        print(f"ハイパーボリューム: {hv_value:,.2f}")
        
        # 正規化された値でのHVも表示
        ref_area = reference_point[0] * reference_point[1]
        normalized_hv = hv_value / ref_area
        print(f"正規化されたHV: {normalized_hv:.6f}")
        
        # データの範囲と比較
        cost_range = costs.max() - costs.min()
        wait_time_range = wait_times.max() - wait_times.min()
        data_area = cost_range * wait_time_range
        print(f"\nデータ範囲の面積: {data_area:,.2f}")
        print(f"リファレンス面積: {ref_area:,.0f}")
        print(f"HV/データ面積比: {hv_value/data_area:.2f}")
        
    except Exception as e:
        print(f"ハイパーボリューム計算エラー: {e}")
        print("代替計算を試行します...")
        
        # 代替計算として、各点の貢献を計算
        total_area = 0.0
        for point in pareto_points:
            # 各点からリファレンスポイントまでの長方形面積
            area = (reference_point[0] - point[0]) * (reference_point[1] - point[1])
            if area > 0:
                total_area += area
        
        print(f"代替計算結果: {total_area:,.2f}")
    
    print("\n計算完了！")


if __name__ == "__main__":
    main() 