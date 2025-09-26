#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リファレンスポイント（20万、200）でのハイパーボリューム計算
前の世代の点も含めた累積的なHV計算を含む実装
YSAUDA
"""

import numpy as np
import re
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt


def extract_pareto_data(file_path: str) -> np.ndarray:
    """ファイルからパレート最適解のデータを抽出"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # パレート最適解詳細セクションを抽出
    pareto_section = re.search(r'全解の結果:(.*?)(?=\n\n|\Z)', content, re.DOTALL).group(1)
    # print(pareto_section)
    
    # データ行を抽出して解析
    costs, wait_times = [], []
    for line in pareto_section.strip().split('\n'):
        if re.match(r'^\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+', line):
            parts = line.split()
            costs.append(float(parts[3]))      # 実際のコスト
            wait_times.append(float(parts[4])) # 実際の待ち時間
    
    # 重複を削除
    points = list(set(zip(costs, wait_times)))
    if points:
        costs, wait_times = zip(*points)
    
    return np.column_stack([costs, wait_times])


def is_dominated(point1: np.ndarray, point2: np.ndarray) -> bool:
    """点1が点2に支配されているかチェック（最小化問題）"""
    # 最小化問題: 両方の目的関数で点1 >= 点2 かつ 少なくとも一方で点1 > 点2
    return np.all(point1 >= point2) and np.any(point1 > point2)


def calculate_pareto_front(points: np.ndarray) -> np.ndarray:
    """パレートフロントを計算"""
    if len(points) == 0:
        return np.array([])
    
    pareto_front = []
    
    for i, point in enumerate(points):
        is_pareto_optimal = True
        
        # 他のすべての点と比較
        for j, other_point in enumerate(points):
            if i != j and is_dominated(point, other_point):
                is_pareto_optimal = False
                break
        
        if is_pareto_optimal:
            pareto_front.append(point)
    
    return np.array(pareto_front)


def calculate_hypervolume(points, ref_point):
    """
    2目的最小化問題におけるハイパーボリュームを計算
    points: パレートフロントの点群 (N x 2)
    ref_point: 参照点 [x_ref, y_ref] (すべての点より劣る座標)
    """
    if len(points) == 0:
        return 0.0
    
    points = np.array(points)
    ref_point = np.array(ref_point)
    
    # 第1目的でソート（昇順）
    sorted_points = points[np.argsort(points[:, 0])]
    
    hypervolume = 0.0
    prev_x = ref_point[0]
    
    print(f"  HV計算詳細:")
    print(f"    参照点: [{ref_point[0]:.2f}, {ref_point[1]:.2f}]")
    print(f"    ソートされた点: {len(sorted_points)}個")
    
    for i, point in enumerate(sorted_points):
        # 現在の点と参照点で長方形の面積を計算
        width = point[0] - prev_x
        height = ref_point[1] - point[1]
        
        if width > 0 and height > 0:
            area = width * height
            hypervolume += area
            print(f"    点{i+1}: [{point[0]:.2f}, {point[1]:.2f}] -> 幅:{width:.2f}, 高さ:{height:.2f}, 面積:{area:.2f}")
        else:
            print(f"    点{i+1}: [{point[0]:.2f}, {point[1]:.2f}] -> 無効 (幅:{width:.2f}, 高さ:{height:.2f})")
        
        prev_x = point[0]
    
    print(f"    総HV: {hypervolume:.2f}")
    return hypervolume


def calculate_hypervolume_correct(points, ref_point):
    """
    正しい2次元ハイパーボリューム計算
    pymooのHV計算を使用して正確な値を取得
    """
    if len(points) == 0:
        return 0.0
    
    points = np.array(points)
    ref_point = np.array(ref_point)
    
    # pymooのHV計算を使用
    try:
        from pymoo.indicators.hv import Hypervolume
        hv_calculator = Hypervolume(ref_point=ref_point)
        hv_value = hv_calculator.do(points)
        print(f"  pymoo HV計算結果: {hv_value:.2f}")
        return hv_value
    except ImportError:
        print("  pymooが利用できません。簡易計算を使用します。")
        return calculate_hypervolume_simple(points, ref_point)


def calculate_hypervolume_simple(points, ref_point):
    """
    簡易的な2次元HV計算（pymooが利用できない場合のフォールバック）
    """
    if len(points) == 0:
        return 0.0
    
    points = np.array(points)
    ref_point = np.array(ref_point)
    
    # 第1目的でソート（昇順）
    sorted_points = points[np.argsort(points[:, 0])]
    
    # 簡易的なHV計算
    total_area = 0.0
    prev_x = ref_point[0]
    
    print(f"  簡易HV計算詳細:")
    print(f"    参照点: [{ref_point[0]:.2f}, {ref_point[1]:.2f}]")
    print(f"    ソートされた点: {len(sorted_points)}個")
    
    for i, point in enumerate(sorted_points):
        # 各点から参照点までの長方形の面積
        width = ref_point[0] - point[0]
        height = ref_point[1] - point[1]
        
        if width > 0 and height > 0:
            area = width * height
            total_area += area
            print(f"    点{i+1}: [{point[0]:.2f}, {point[1]:.2f}] -> 幅:{width:.2f}, 高さ:{height:.2f}, 面積:{area:.2f}")
        else:
            print(f"    点{i+1}: [{point[0]:.2f}, {point[1]:.2f}] -> 無効 (幅:{width:.2f}, 高さ:{height:.2f})")
    
    print(f"    総HV（簡易計算）: {total_area:.2f}")
    return total_area


def calculate_igd_plus(pareto_front, true_pareto_front, ref_point):
    """
    IGD+（Inverted Generational Distance Plus）を計算
    pareto_front: 現在のパレートフロント
    true_pareto_front: 真のパレートフロント（参照用）
    ref_point: 参照点
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


def calculate_gd(pareto_front, true_pareto_front, ref_point):
    """
    GD（Generational Distance）を計算
    pareto_front: 現在のパレートフロント
    true_pareto_front: 真のパレートフロント（参照用）
    ref_point: 参照点
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


def get_true_pareto_front():
    """
    真のパレートフロントを取得（all_morningデータを使用）
    """
    # 提供された真のパレートフロントデータ
    true_points = np.array([
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
    
    # パレートフロントを計算（重複除去と支配関係の確認）
    true_pareto = calculate_pareto_front(true_points)
    print(f"真のパレートフロント: {len(true_points)}点 → {len(true_pareto)}点（パレート最適）")
    
    return true_pareto


def process_hypervolume_calculation(file_name: str, cumulative_points: np.ndarray = None, prev_hv: float = None, true_pareto_front: np.ndarray = None) -> tuple:
    """ファイル名を引数としてハイパーボリューム計算を実行"""
    # 設定
    file_path = f"distributed_pareto_results/{file_name}"
    reference_point = np.array([200000.0, 200.0])
    
    # データ抽出
    current_points = extract_pareto_data(file_path)
    
    # データの範囲を確認
    if len(current_points) > 0:
        min_cost = np.min(current_points[:, 0])
        max_cost = np.max(current_points[:, 0])
        min_wait = np.min(current_points[:, 1])
        max_wait = np.max(current_points[:, 1])
        print(f"  データ範囲: コスト[{min_cost:.2f}, {max_cost:.2f}], 待ち時間[{min_wait:.2f}, {max_wait:.2f}]")
    
    # 累積的な点の集合を作成
    if cumulative_points is not None:
        all_points = np.vstack([cumulative_points, current_points])
        print(f"前の世代の点数: {len(cumulative_points)}")
    else:
        all_points = current_points
        print("初回実行（前の世代なし）")
    
    # 重複を削除
    unique_points = np.unique(all_points, axis=0)
    print(f"重複除去後の点数: {len(unique_points)}")
    
    # 累積データの範囲を確認
    if len(unique_points) > 0:
        cum_min_cost = np.min(unique_points[:, 0])
        cum_max_cost = np.max(unique_points[:, 0])
        cum_min_wait = np.min(unique_points[:, 1])
        cum_max_wait = np.max(unique_points[:, 1])
        print(f"  累積データ範囲: コスト[{cum_min_cost:.2f}, {cum_max_cost:.2f}], 待ち時間[{cum_min_wait:.2f}, {cum_max_wait:.2f}]")
    
    # パレートフロント計算
    pareto_front = calculate_pareto_front(unique_points)
    
    # ハイパーボリューム計算
    hv_value = calculate_hypervolume_correct(pareto_front, reference_point)
    
    # IGD+とGDの計算
    igd_plus_value = None
    gd_value = None
    
    if true_pareto_front is not None and len(true_pareto_front) > 0:
        igd_plus_value = calculate_igd_plus(pareto_front, true_pareto_front, reference_point)
        gd_value = calculate_gd(pareto_front, true_pareto_front, reference_point)
        print(f"  IGD+: {igd_plus_value:.6f}")
        print(f"  GD: {gd_value:.6f}")
    
    # 結果出力
    print(f"\nファイル: {file_name}")
    print(f"現在の世代の点数: {len(current_points)}")
    print(f"累積点数: {len(unique_points)}")
    print(f"パレートフロント点数: {len(pareto_front)}")
    print(f"ハイパーボリューム: {hv_value:,.2f}")
    
    # 前の世代との比較
    if prev_hv is not None:
        hv_diff = hv_value - prev_hv
        print(f"前の世代からの変化: {hv_diff:+,.2f}")
        if hv_diff < 0:
            print("⚠️  HVが減少しています！")
            print("デバッグ情報:")
            print(f"  前の累積点数: {len(cumulative_points) if cumulative_points is not None else 0}")
            print(f"  現在の累積点数: {len(unique_points)}")
            print(f"  新しく追加された点: {len(current_points)}")
            
            # パレートフロントの詳細を表示
            print("  現在のパレートフロント:")
            for i, point in enumerate(pareto_front):
                print(f"    PF{i+1}: [{point[0]:.2f}, {point[1]:.2f}]")
            
            # 前の世代のパレートフロントとの比較
            if cumulative_points is not None:
                prev_pareto = calculate_pareto_front(cumulative_points)
                print(f"  前の世代のパレートフロント点数: {len(prev_pareto)}")
                print("  前の世代のパレートフロント:")
                for i, point in enumerate(prev_pareto):
                    print(f"    PF{i+1}: [{point[0]:.2f}, {point[1]:.2f}]")
                
                # 前の世代のHVを再計算して比較
                prev_hv_recalc = calculate_hypervolume_correct(prev_pareto, reference_point)
                print(f"  前の世代のHV（再計算）: {prev_hv_recalc:,.2f}")
                print(f"  実際の差: {hv_value - prev_hv_recalc:+,.2f}")
                
                # どの点が新しく追加されたかを確認
                print("  新しく追加された点（現在の世代）:")
                for i, point in enumerate(current_points):
                    print(f"    点{i+1}: [{point[0]:.2f}, {point[1]:.2f}]")
                
                # 支配関係の詳細を確認
                print("  支配関係の詳細:")
                for i, new_point in enumerate(current_points):
                    for j, prev_point in enumerate(prev_pareto):
                        if is_dominated(prev_point, new_point):
                            print(f"    新しい点{i+1} [{new_point[0]:.2f}, {new_point[1]:.2f}] が前の点{j+1} [{prev_point[0]:.2f}, {prev_point[1]:.2f}] を支配")
                        elif is_dominated(new_point, prev_point):
                            print(f"    前の点{j+1} [{prev_point[0]:.2f}, {prev_point[1]:.2f}] が新しい点{i+1} [{new_point[0]:.2f}, {new_point[1]:.2f}] を支配")
    
    # 正規化された値
    ref_area = reference_point[0] * reference_point[1]
    normalized_hv = hv_value / ref_area
    print(f"正規化されたHV: {normalized_hv:.6f}")
    
    return normalized_hv, unique_points, pareto_front, igd_plus_value, gd_value


def main():
    """メイン関数"""
    
    # 世代順のファイル名リスト（古い順）
    filenames = [
        "pareto_details_distributed_20250819_133014.txt",  # 10世代
        "pareto_details_distributed_20250819_131904.txt",  # 20世代
        "pareto_details_distributed_20250819_133502.txt",  # 40世代
        "pareto_details_distributed_20250819_134224.txt",  # 60世代
        "pareto_details_distributed_20250819_134454.txt",  # 80世代
        "pareto_details_distributed_20250819_135546.txt",  # 100世代
    ]
    
    # 対応する世代数
    generations = [10, 20, 40, 60, 80, 100]
    
    hvs = []
    igd_plus_values = []
    gd_values = []
    cumulative_points = None
    prev_hv = None
    
    # 真のパレートフロントを取得
    true_pareto_front = get_true_pareto_front()
    print(f"真のパレートフロント点数: {len(true_pareto_front)}")
    
    print("累積的なHV計算を開始します...")
    print("=" * 60)

    reference_point = np.array([200000.0, 200.0])
    
    for i, (file_name, gen) in enumerate(zip(filenames, generations)):
        print(f"\n--- 世代 {gen} の処理 ---")
        
        # 累積的なHV計算
        normalized_hv, cumulative_points, pareto_front, igd_plus, gd = process_hypervolume_calculation(
            file_name, cumulative_points, prev_hv, true_pareto_front
        )
        
        hvs.append(normalized_hv)
        igd_plus_values.append(igd_plus if igd_plus is not None else float('inf'))
        gd_values.append(gd if gd is not None else float('inf'))
        prev_hv = normalized_hv * (reference_point[0] * reference_point[1])  # 非正規化された値
        
        # パレートフロントの可視化
        # plt.figure(figsize=(10, 6))
        # plt.scatter(cumulative_points[:, 0], cumulative_points[:, 1], 
        #            alpha=0.3, s=20, label='all points', color='lightblue')
        # plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
        #            s=50, label='pareto front', color='red', zorder=5)
        # plt.axhline(y=reference_point[1], color='gray', linestyle='--', alpha=0.7)
        # plt.axvline(x=reference_point[0], color='gray', linestyle='--', alpha=0.7)
        # plt.xlabel('cost')
        # plt.ylabel('waiting time')
        # plt.title(f'generation {gen} cumulative pareto front')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.savefig(f"res_0819/cumulative_pareto_front_gen_{gen}.png", dpi=300, bbox_inches='tight')
        # plt.close()
    
    # 最終的なHV推移の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(generations, hvs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('generation')
    plt.ylabel('normalized hypervolume')
    plt.title('cumulative hypervolume progression')
    plt.grid(True, alpha=0.3)
    plt.xticks(generations)
    
    # 各点に値を表示
    for i, (gen, hv) in enumerate(zip(generations, hvs)):
        plt.annotate(f'{hv:.6f}', (gen, hv), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig("res_0819/cumulative_hv_progression.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 最終的な指標推移の可視化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # HV推移
    ax1.plot(generations, hvs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('generation')
    ax1.set_ylabel('normalized hypervolume')
    ax1.set_title('cumulative hypervolume progression')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(generations)
    
    # 各点に値を表示
    for i, (gen, hv) in enumerate(zip(generations, hvs)):
        ax1.annotate(f'{hv:.6f}', (gen, hv), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # IGD+推移
    ax2.plot(generations, igd_plus_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('generation')
    ax2.set_ylabel('IGD+')
    ax2.set_title('IGD+ progression')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(generations)
    
    # 各点に値を表示
    for i, (gen, igd) in enumerate(zip(generations, igd_plus_values)):
        if igd != float('inf'):
            ax2.annotate(f'{igd:.6f}', (gen, igd), 
                        textcoords="offset points", 
                        xytext=(0,10), ha='center')
    
    # GD推移
    ax3.plot(generations, gd_values, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('generation')
    ax3.set_ylabel('GD')
    ax3.set_title('GD progression')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(generations)
    
    # 各点に値を表示
    for i, (gen, gd) in enumerate(zip(generations, gd_values)):
        if gd != float('inf'):
            ax3.annotate(f'{gd:.6f}', (gen, gd), 
                        textcoords="offset points", 
                        xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig("res_0819/cumulative_metrics_progression.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("累積的なHV計算が完了しました！")
    print(f"最終的な正規化HV: {hvs[-1]:.6f}")
    print(f"最終的なIGD+: {igd_plus_values[-1]:.6f}")
    print(f"最終的なGD: {gd_values[-1]:.6f}")
    
    # 結果の詳細表示
    print("\n詳細結果:")
    print("世代\tHV\t\tIGD+\t\tGD")
    print("-" * 50)
    for i, (gen, hv, igd, gd) in enumerate(zip(generations, hvs, igd_plus_values, gd_values)):
        igd_str = f"{igd:.6f}" if igd != float('inf') else "inf"
        gd_str = f"{gd:.6f}" if gd != float('inf') else "inf"
        print(f"{gen}\t{hv:.6f}\t{igd_str}\t{gd_str}")


if __name__ == "__main__":
    
    main() 