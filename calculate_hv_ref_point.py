#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リファレンスポイント（20万、200）でのハイパーボリューム（HV）計算
指定されたファイルからデータを抽出→np.arrayに変換→HV計算→結果出力
真値からのGD、IGD+も計算してグラフに表示
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


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
                # 3番目が実際のコスト、4番目が実際の待ち時間
                # データ形式: "1    0.000        1.000      86957.00     32.94"
                cost = float(parts[3])  # 4番目が実際のコスト
                wait_time = float(parts[4])  # 5番目が実際の待ち時間
                costs.append(cost)
                wait_times.append(wait_time)
        
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


def get_true_pareto_front() -> np.ndarray:
    """
    真のパレートフロントを取得（all_morningデータを使用）
    
    Returns:
        np.ndarray: 真のパレートフロントの点群
    """
    # all_morningデータ（真のパレートフロント）
    all_morning = np.array([
        [6.539200e+04, 2.262500e+01],
        [9.928600e+04, 1.937500e+00],
        [8.725200e+04, 6.156250e+00],
        [6.290300e+04, 2.468750e+01],
        [2.808600e+04, 6.403125e+01],
        [6.695000e+04, 2.162500e+01],
        [4.089400e+04, 4.553125e+01],
        [8.581800e+04, 7.406250e+00],
        [6.774400e+04, 1.978125e+01],
        [8.732100e+04, 5.906250e+00],
        [8.218200e+04, 8.656250e+00],
        [7.681300e+04, 1.387500e+01],
        [0.000000e+00, 1.051250e+02],
        [2.325000e+04, 6.806250e+01],
        [7.873800e+04, 1.156250e+01],
        [1.747500e+04, 8.115625e+01],
        [6.714700e+04, 2.078125e+01],
        [1.010750e+05, 1.718750e+00],
        [8.217000e+04, 9.343750e+00],
        [7.217800e+04, 1.709375e+01],
        [5.702300e+04, 2.831250e+01],
        [5.682800e+04, 2.993750e+01],
        [1.803000e+04, 8.018750e+01],
        [5.900000e+04, 2.700000e+01],
        [7.711400e+04, 1.362500e+01],
        [5.032600e+04, 3.643750e+01],
        [2.620800e+04, 6.443750e+01],
        [8.998000e+04, 4.250000e+00],
        [4.762200e+04, 3.925000e+01],
        [4.472300e+04, 4.128125e+01],
        [9.504100e+04, 3.156250e+00],
        [8.613700e+04, 6.500000e+00],
        [9.471700e+04, 3.312500e+00],
        [8.490100e+04, 7.468750e+00],
        [7.754600e+04, 1.265625e+01],
        [4.791500e+04, 3.859375e+01],
        [6.614100e+04, 2.259375e+01],
        [6.903400e+04, 1.840625e+01],
        [5.225600e+04, 3.450000e+01],
        [8.158800e+04, 1.006250e+01],
        [7.552900e+04, 1.437500e+01],
        [2.516600e+04, 6.800000e+01],
        [8.129000e+04, 1.012500e+01],
        [1.105040e+05, 1.250000e-01],
        [6.632600e+04, 2.190625e+01],
        [7.989400e+04, 1.090625e+01],
        [4.739900e+04, 4.000000e+01],
        [8.121800e+04, 1.015625e+01],
        [7.109400e+04, 1.787500e+01],
        [8.477300e+04, 7.593750e+00],
        [7.472100e+04, 1.443750e+01],
        [3.131100e+04, 6.062500e+01],
        [8.033600e+04, 1.053125e+01],
        [6.436800e+04, 2.356250e+01],
        [9.513500e+04, 2.968750e+00],
        [8.243900e+04, 8.468750e+00],
        [9.574200e+04, 2.750000e+00],
        [7.112900e+04, 1.746875e+01],
        [6.973300e+04, 1.796875e+01],
        [8.902200e+04, 4.437500e+00],
        [1.004520e+05, 1.843750e+00],
        [9.871100e+04, 2.218750e+00],
        [5.678300e+04, 3.009375e+01],
        [9.139500e+04, 4.062500e+00],
        [9.238600e+04, 3.812500e+00],
        [6.369800e+04, 2.387500e+01],
        [7.247900e+04, 1.596875e+01],
        [1.090010e+05, 5.937500e-01],
        [6.020900e+04, 2.543750e+01],
        [1.017080e+05, 1.656250e+00],
        [3.152400e+04, 5.871875e+01],
        [5.296800e+04, 3.375000e+01],
        [8.818200e+04, 5.843750e+00],
        [6.298700e+04, 2.450000e+01],
        [9.217200e+04, 3.875000e+00],
        [1.666600e+04, 8.550000e+01],
        [8.594900e+04, 7.250000e+00],
        [8.841400e+04, 5.812500e+00],
        [1.007250e+05, 1.812500e+00],
        [8.196500e+04, 9.375000e+00],
        [8.051800e+04, 1.050000e+01],
        [9.264300e+04, 3.531250e+00],
        [1.030360e+05, 1.375000e+00],
        [7.459700e+04, 1.525000e+01],
        [8.464800e+04, 7.906250e+00],
        [8.847900e+04, 5.656250e+00],
        [2.820600e+04, 6.337500e+01],
        [3.068400e+04, 6.168750e+01],
        [1.066790e+05, 9.375000e-01],
        [8.192700e+04, 9.625000e+00],
        [5.155100e+04, 3.490625e+01],
        [3.596500e+04, 4.990625e+01],
        [6.676900e+04, 2.181250e+01],
        [4.067700e+04, 4.953125e+01],
        [7.364200e+04, 1.593750e+01],
        [8.190100e+04, 9.781250e+00],
        [5.772300e+04, 2.781250e+01],
        [9.421700e+04, 3.437500e+00],
        [3.842200e+04, 4.984375e+01],
        [8.870200e+04, 5.437500e+00],
        [7.237700e+04, 1.671875e+01],
        [9.704800e+04, 2.562500e+00],
        [4.125400e+04, 4.403125e+01],
        [7.884000e+03, 9.206250e+01],
        [5.602400e+04, 3.178125e+01],
        [1.059230e+05, 1.031250e+00],
        [7.393200e+04, 1.587500e+01],
        [4.861400e+04, 3.725000e+01],
        [3.341800e+04, 5.212500e+01],
        [2.148100e+04, 7.525000e+01],
        [1.071000e+05, 8.125000e-01],
        [5.997800e+04, 2.671875e+01],
        [6.272700e+04, 2.528125e+01],
        [8.447700e+04, 8.281250e+00],
        [1.671700e+04, 8.206250e+01],
        [5.249200e+04, 3.440625e+01],
        [3.018900e+04, 6.212500e+01],
        [6.310100e+04, 2.406250e+01],
        [6.005400e+04, 2.637500e+01],
        [6.462000e+04, 2.303125e+01],
        [7.784900e+04, 1.256250e+01],
        [5.400900e+04, 3.250000e+01],
        [2.973200e+04, 6.328125e+01],
        [9.702900e+04, 2.625000e+00],
        [9.882900e+04, 2.062500e+00],
        [1.130220e+05, 0.000000e+00],
        [9.738600e+04, 2.437500e+00],
        [2.173100e+04, 7.378125e+01]
    ])
    
    return all_morning


def calculate_hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    2次元のハイパーボリュームを計算
    
    Args:
        points (np.ndarray): 形状 (n, 2) の点の配列
        reference_point (np.ndarray): リファレンスポイント [x, y]
        
    Returns:
        float: ハイパーボリューム値
    """
    if len(points) == 0:
        return 0.0
    
    # 点をリファレンスポイントで正規化
    normalized_points = points - reference_point
    
    # 負の値を0に設定（リファレンスポイントより悪い解は無視）
    normalized_points = np.maximum(normalized_points, 0)
    
    # 点をソート（x座標で昇順、y座標で降順）
    sorted_indices = np.lexsort([-normalized_points[:, 1], normalized_points[:, 0]])
    sorted_points = normalized_points[sorted_indices]
    
    # ハイパーボリューム計算
    hv = 0.0
    max_y = 0.0
    
    for point in sorted_points:
        if point[1] > max_y:
            # 新しいy座標の最大値が見つかった場合
            width = point[0]
            height = point[1] - max_y
            hv += width * height
            max_y = point[1]
        elif point[0] > 0 and point[1] > 0:
            # 同じy座標でも、x座標が異なる場合の処理
            width = point[0]
            height = point[1]
            hv += width * height
    
    return hv


def calculate_gd(pareto_front: np.ndarray, true_pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    GD（Generational Distance）を計算
    
    Args:
        pareto_front (np.ndarray): 現在のパレートフロント
        true_pareto_front (np.ndarray): 真のパレートフロント（参照用）
        ref_point (np.ndarray): 参照点（正規化用）
        
    Returns:
        float: GD値
    """
    if len(pareto_front) == 0 or len(true_pareto_front) == 0:
        return float('inf')
    
    pareto_front = np.array(pareto_front)
    true_pareto_front = np.array(true_pareto_front)
    
    # 各現在のパレートフロントの点について、真のフロントでの最小距離を計算
    distances = []
    
    for current_point in pareto_front:
        min_distance = float('inf')
        
        for true_point in true_pareto_front:
            # 正規化された距離を計算
            normalized_diff = (current_point - true_point) / ref_point
            distance = np.linalg.norm(normalized_diff)
            
            if distance < min_distance:
                min_distance = distance
        
        distances.append(min_distance)
    
    gd = np.mean(distances)
    return gd


def calculate_igd_plus(pareto_front: np.ndarray, true_pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    IGD+（Inverted Generational Distance Plus）を計算
    
    Args:
        pareto_front (np.ndarray): 現在のパレートフロント
        true_pareto_front (np.ndarray): 真のパレートフロント（参照用）
        ref_point (np.ndarray): 参照点（正規化用）
        
    Returns:
        float: IGD+値
    """
    if len(pareto_front) == 0 or len(true_pareto_front) == 0:
        return float('inf')
    
    pareto_front = np.array(pareto_front)
    true_pareto_front = np.array(true_pareto_front)
    
    # 各真のパレートフロントの点について、現在のフロントでの最小距離を計算
    distances = []
    
    for true_point in true_pareto_front:
        min_distance = float('inf')
        
        for current_point in pareto_front:
            # 正規化された距離を計算
            normalized_diff = (current_point - true_point) / ref_point
            # IGD+では負の値を0に置き換える
            normalized_diff = np.maximum(normalized_diff, 0)
            distance = np.linalg.norm(normalized_diff)
            
            if distance < min_distance:
                min_distance = distance
        
        distances.append(min_distance)
    
    igd_plus = np.mean(distances)
    return igd_plus


def main():
    """
    メイン関数：データ抽出→HV計算→GD/IGD+計算→結果出力
    """
    # ファイルパス
    file_path = "distributed_pareto_results/pareto_details_distributed_20250818_160526.txt"
    
    # 複数のリファレンスポイントを定義
    reference_points = [
        np.array([200000, 200]),    # 元のリクエスト
        np.array([100000, 100]),    # より適切な値
        np.array([90000, 50]),      # データ範囲に近い値
        np.array([87000, 50]),      # データ範囲に適した値
        np.array([90000, 46]),      # データ範囲に適した値
        np.array([85000, 40]),      # データ範囲より小さい値（HV計算用）
        np.array([80000, 30]),      # データ範囲より小さい値（HV計算用）
    ]
    
    print("=" * 60)
    print("ハイパーボリューム計算（複数リファレンスポイント）")
    print("=" * 60)
    
    # ステップ1: データ抽出
    print("\n1. データ抽出中...")
    costs, wait_times = extract_pareto_data_from_file(file_path)
    
    if len(costs) == 0:
        print("データの抽出に失敗しました。")
        return
    
    # ステップ2: numpy配列に変換（既に完了）
    print("\n2. numpy配列への変換完了")
    
    # ステップ3: 真のパレートフロントを取得
    print("\n3. 真のパレートフロントを取得中...")
    true_pareto_front = get_true_pareto_front()
    print(f"真のパレートフロント点数: {len(true_pareto_front)}")
    
    # ステップ4: 指標計算
    print("\n4. 指標計算中...")
    points = np.column_stack([costs, wait_times])
    
    # リファレンスポイントを選択（HV計算用）
    hv_ref_point = np.array([90000, 46])  # データ範囲に適した値
    
    # HV計算
    hv_value = calculate_hypervolume(points, hv_ref_point)
    
    # GD計算
    gd_value = calculate_gd(points, true_pareto_front, hv_ref_point)
    
    # IGD+計算
    igd_plus_value = calculate_igd_plus(points, true_pareto_front, hv_ref_point)
    
    # ステップ5: 結果出力
    print("\n5. 結果出力")
    print("-" * 60)
    
    # HV計算結果（複数リファレンスポイント）
    print("ハイパーボリューム計算結果:")
    for i, ref_point in enumerate(reference_points):
        hv_value_multi = calculate_hypervolume(points, ref_point)
        
        print(f"リファレンスポイント {i+1}: [{ref_point[0]:,}, {ref_point[1]}]")
        print(f"  パレート最適解数: {len(points)}")
        print(f"  ハイパーボリューム: {hv_value_multi:.2f}")
        
        # 正規化された値でのHVも表示
        normalized_hv = hv_value_multi / (ref_point[0] * ref_point[1])
        print(f"  正規化されたHV: {normalized_hv:.6f}")
        print()
    
    # 真値との比較指標
    print("真値との比較指標:")
    print(f"リファレンスポイント: [{hv_ref_point[0]:,}, {hv_ref_point[1]}]")
    print(f"  パレート最適解数: {len(points)}")
    print(f"  ハイパーボリューム: {hv_value:.2f}")
    print(f"  正規化されたHV: {hv_value / (hv_ref_point[0] * hv_ref_point[1]):.6f}")
    print(f"  GD (Generational Distance): {gd_value:.6f}")
    print(f"  IGD+ (Inverted Generational Distance Plus): {igd_plus_value:.6f}")
    
    # ステップ6: グラフ作成
    print("\n6. グラフ作成中...")
    
    # 3つの指標を並べて表示するサブプロット
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # サブプロット1: パレートフロントの可視化
    ax1.scatter(true_pareto_front[:, 0], true_pareto_front[:, 1], 
                c='red', s=50, label='真のパレートフロント', alpha=0.7, zorder=5)
    ax1.scatter(points[:, 0], points[:, 1], 
                c='blue', s=30, label='現在のパレートフロント', alpha=0.6)
    ax1.axhline(y=hv_ref_point[1], color='gray', linestyle='--', alpha=0.7, label=f'リファレンス点 y={hv_ref_point[1]}')
    ax1.axvline(x=hv_ref_point[0], color='gray', linestyle='--', alpha=0.7, label=f'リファレンス点 x={hv_ref_point[0]:,}')
    ax1.set_xlabel('コスト')
    ax1.set_ylabel('待ち時間')
    ax1.set_title('パレートフロント比較')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # サブプロット2: ハイパーボリューム（複数リファレンスポイント）
    ref_point_labels = [f'[{ref[0]/1000:.0f}k, {ref[1]}]' for ref in reference_points]
    hv_values = [calculate_hypervolume(points, ref) for ref in reference_points]
    normalized_hv_values = [hv / (ref[0] * ref[1]) for hv, ref in zip(hv_values, reference_points)]
    
    ax2.bar(range(len(reference_points)), normalized_hv_values, alpha=0.7, color='green')
    ax2.set_xlabel('リファレンスポイント')
    ax2.set_ylabel('正規化されたハイパーボリューム')
    ax2.set_title('リファレンスポイント別ハイパーボリューム')
    ax2.set_xticks(range(len(reference_points)))
    ax2.set_xticklabels(ref_point_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 各バーに値を表示
    for i, (hv, norm_hv) in enumerate(zip(hv_values, normalized_hv_values)):
        ax2.annotate(f'{norm_hv:.6f}', (i, norm_hv), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # サブプロット3: 真値との比較指標
    metrics = ['HV', 'GD', 'IGD+']
    metric_values = [
        hv_value / (hv_ref_point[0] * hv_ref_point[1]),  # 正規化されたHV
        gd_value,  # GD
        igd_plus_value  # IGD+
    ]
    
    colors = ['green', 'orange', 'red']
    bars = ax3.bar(metrics, metric_values, color=colors, alpha=0.7)
    ax3.set_xlabel('評価指標')
    ax3.set_ylabel('値')
    ax3.set_title('真値との比較指標')
    ax3.grid(True, alpha=0.3)
    
    # 各バーに値を表示
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax3.annotate(f'{value:.6f}', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("pareto_analysis_with_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n計算完了！")
    print(f"グラフを 'pareto_analysis_with_metrics.png' として保存しました。")


if __name__ == "__main__":
    main() 