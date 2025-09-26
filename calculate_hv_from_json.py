import numpy as np
import json
from typing import List, Tuple
from pymoo.indicators.hv import HV
import os
import re
import matplotlib.pyplot as plt

"""
NSGA
"""
def extract_pareto_data_from_txt_file(file_path: str) -> np.ndarray:
    """
    solutions_generation_xxx.txtファイルからパレートデータを抽出
    
    Args:
        file_path: テキストファイルのパス
    
    Returns:
        numpy配列: [[cost, waiting_time], ...]
    """
    pareto_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # ヘッダー行をスキップしてデータ行を処理
        for line in lines:
            # 個体ID、コスト、待ち時間、染色体の形式で解析
            if re.match(r'^\d+\t', line):  # タブ区切りのデータ行
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        cost = float(parts[1])
                        waiting_time = float(parts[2])
                        pareto_data.append([cost, waiting_time])
                    except ValueError:
                        continue  # 数値変換できない行はスキップ
        
        print(f"ファイル {file_path} から {len(pareto_data)} 個の解を抽出")
        return np.array(pareto_data)
        
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
        return np.array([])
    except Exception as e:
        print(f"ファイル読み込みエラー {file_path}: {e}")
        return np.array([])


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


def is_dominated(point1: np.ndarray, point2: np.ndarray) -> bool:
    """点1が点2に支配されているかチェック"""
    return np.all(point2 <= point1) and np.any(point2 < point1)


def calculate_pareto_front(points: np.ndarray) -> np.ndarray:
    """パレートフロントを計算"""
    if len(points) == 0:
        return np.array([])
    
    pareto_front = []
    
    for i, point in enumerate(points):
        is_pareto = True
        
        for j, other_point in enumerate(points):
            if i != j and is_dominated(point, other_point):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_front.append(point)
    
    return np.array(pareto_front)


def calculate_hypervolume_with_pymoo(points, ref_point):
    """
    PymooのHVクラスを使用してハイパーボリュームを計算
    points: パレートフロントの点群 (N x 2)
    ref_point: 参照点 [x_ref, y_ref] (すべての点より劣る座標)
    """
    if len(points) == 0:
        return 0.0
    
    # PymooのHVクラスを使用
    hv_calculator = HV(ref_point=ref_point)
    hypervolume = hv_calculator.do(points)
    
    return hypervolume


def process_hypervolume_calculation_from_txt(execution_dir: str, iteration: int, previous_points: np.ndarray = None, true_pareto_front: np.ndarray = None) -> Tuple[float, np.ndarray, float, float]:
    """指定されたiterationのハイパーボリューム計算を実行（前のiterationの解も考慮）"""
    print(f"\n=== Iteration {iteration} のハイパーボリューム計算 ===")
    
    # ファイルパスを構築
    file_path = os.path.join(execution_dir, f"solutions_generation_{iteration:03d}.txt")
    
    # データを抽出
    current_pareto_data = extract_pareto_data_from_txt_file(file_path)
    
    if len(current_pareto_data) == 0:
        print(f"Iteration {iteration}: データが見つかりません")
        return 0.0, np.array([]), float('inf'), float('inf')
    
    print(f"現在のiterationのデータ数: {len(current_pareto_data)}")
    
    # 前のiterationの解と現在の解を結合
    if previous_points is not None and len(previous_points) > 0:
        print(f"前のiterationのデータ数: {len(previous_points)}")
        combined_points = np.vstack([previous_points, current_pareto_data])
        print(f"結合後の総データ数: {len(combined_points)}")
    else:
        combined_points = current_pareto_data
        print("前のiterationのデータなし")
    
    # リファレンスポイントを設定
    # コストと待機時間の最大値より少し大きい値を使用
    max_cost = np.max(combined_points[:, 0])
    max_waiting_time = np.max(combined_points[:, 1])
    
    # リファレンスポイントを設定（最大値の1.1倍）
    reference_point = np.array([200000, 200])
    
    print(f"リファレンスポイント: {reference_point}")
    
    # パレートフロントを計算
    pareto_front = calculate_pareto_front(combined_points)
    print(f"パレートフロントの点数: {len(pareto_front)}")
    
    # PymooのHVクラスを使用してハイパーボリュームを計算
    hv = calculate_hypervolume_with_pymoo(pareto_front, reference_point)
    print(f"ハイパーボリューム: {hv:.2f}")

    # 正規化された値
    ref_area = reference_point[0] * reference_point[1]
    normalized_hv = hv / ref_area
    print(f"正規化されたHV: {normalized_hv:.6f}")
    
    # GD計算
    gd_value = float('inf')
    if true_pareto_front is not None and len(true_pareto_front) > 0:
        gd_value = calculate_gd(pareto_front, true_pareto_front, reference_point)
        print(f"GD (Generational Distance): {gd_value:.6f}")
    
    # IGD+計算
    igd_plus_value = float('inf')
    if true_pareto_front is not None and len(true_pareto_front) > 0:
        igd_plus_value = calculate_igd_plus(pareto_front, true_pareto_front, reference_point)
        print(f"IGD+ (Inverted Generational Distance Plus): {igd_plus_value:.6f}")

    return normalized_hv, pareto_front, gd_value, igd_plus_value


def main():
    """メイン関数"""
    # execution_20250819_161331ディレクトリのパス
    execution_dir = "execution_20250819_161331"
    
    # ディレクトリが存在するかチェック
    if not os.path.exists(execution_dir):
        print(f"エラー: ディレクトリ {execution_dir} が見つかりません")
        return
    
    # 0から100まで5刻みでiterationを処理
    iterations = list(range(0, 101, 5))
    
    print(f"処理するiteration数: {len(iterations)}")
    print(f"Iterations: {iterations}")
    
    # 真のパレートフロントを取得
    true_pareto_front = get_true_pareto_front()
    print(f"真のパレートフロント点数: {len(true_pareto_front)}")
    
    normalized_hvs = []
    gd_values = []
    igd_plus_values = []
    previous_pareto_front = None
    
    # 各iterationのHVを計算（前のiterationの解も考慮）
    for iteration in iterations:
        normalized_hv, pareto_front, gd_value, igd_plus_value = process_hypervolume_calculation_from_txt(
            execution_dir, iteration, previous_pareto_front, true_pareto_front
        )
        print(f"Iteration {iteration}: 正規化されたHV: {normalized_hv:.6f}, GD: {gd_value:.6f}, IGD+: {igd_plus_value:.6f}")
        
        normalized_hvs.append(normalized_hv)
        gd_values.append(gd_value)
        igd_plus_values.append(igd_plus_value)
        
        # 次のiterationのために現在のパレートフロントを保存
        previous_pareto_front = pareto_front

    print("\n=== 全iterationの処理が完了しました ===")
    print("各iterationの指標:")
    print("Iteration\tHV\t\tGD\t\tIGD+")
    print("-" * 60)
    for i, (iteration, hv, gd, igd_plus) in enumerate(zip(iterations, normalized_hvs, gd_values, igd_plus_values)):
        gd_str = f"{gd:.6f}" if gd != float('inf') else "inf"
        igd_plus_str = f"{igd_plus:.6f}" if igd_plus != float('inf') else "inf"
        print(f"{iteration}\t\t{hv:.6f}\t{gd_str}\t{igd_plus_str}")

    # 統計情報の表示
    print(f"\n=== 統計情報 ===")
    print(f"HVの平均値: {np.mean(normalized_hvs):.6f}")
    print(f"HVの標準偏差: {np.std(normalized_hvs):.6f}")
    print(f"HVの最小値: {np.min(normalized_hvs):.6f}")
    print(f"HVの最大値: {np.max(normalized_hvs):.6f}")
    
    # GDの統計（inf値を除外）
    gd_finite = [gd for gd in gd_values if gd != float('inf')]
    if gd_finite:
        print(f"\nGDの平均値: {np.mean(gd_finite):.6f}")
        print(f"GDの標準偏差: {np.std(gd_finite):.6f}")
        print(f"GDの最小値: {np.min(gd_finite):.6f}")
        print(f"GDの最大値: {np.max(gd_finite):.6f}")
    
    # IGD+の統計（inf値を除外）
    igd_plus_finite = [igd for igd in igd_plus_values if igd != float('inf')]
    if igd_plus_finite:
        print(f"\nIGD+の平均値: {np.mean(igd_plus_finite):.6f}")
        print(f"IGD+の標準偏差: {np.std(igd_plus_finite):.6f}")
        print(f"IGD+の最小値: {np.min(igd_plus_finite):.6f}")
        print(f"IGD+の最大値: {np.max(igd_plus_finite):.6f}")

    # 3つの指標を並べて表示するサブプロット
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # サブプロット1: ハイパーボリューム
    ax1.plot(iterations, normalized_hvs, marker='o', linewidth=2, markersize=6, color='blue', label='Hypervolume')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Normalized Hypervolume')
    ax1.set_title('Hypervolume vs Iteration')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # サブプロット2: GD
    ax2.plot(iterations, gd_values, marker='s', linewidth=2, markersize=6, color='orange', label='Generational Distance')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('GD')
    ax2.set_title('Generational Distance vs Iteration')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # サブプロット3: IGD+
    ax3.plot(iterations, igd_plus_values, marker='^', linewidth=2, markersize=6, color='red', label='IGD+')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('IGD+')
    ax3.set_title('IGD+ vs Iteration')
    ax3.set_ylim(0, 0.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # グラフを保存
    plt.savefig('nsga_metrics_vs_iteration.png', dpi=300, bbox_inches='tight')
    plt.savefig('nsga_metrics_vs_iteration.pdf', bbox_inches='tight')
    
    print("\nグラフを保存しました:")
    print("- nsga_metrics_vs_iteration.png")
    print("- nsga_metrics_vs_iteration.pdf")
    
    # グラフを表示
    plt.show()
    
    # 最終的なパレートフロントの可視化
    if 1== 2 and previous_pareto_front is not None and len(previous_pareto_front) > 0:
        print("\n最終的なパレートフロントの可視化...")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(true_pareto_front[:, 0], true_pareto_front[:, 1], 
                    c='red', s=50, label='真のパレートフロント', alpha=0.7, zorder=5)
        plt.scatter(previous_pareto_front[:, 0], previous_pareto_front[:, 1], 
                    c='blue', s=30, label='最終的なパレートフロント', alpha=0.6)
        
        # リファレンスポイントを表示
        ref_point = np.array([90000, 46])
        plt.axhline(y=ref_point[1], color='gray', linestyle='--', alpha=0.7, label=f'リファレンス点 y={ref_point[1]}')
        plt.axvline(x=ref_point[0], color='gray', linestyle='--', alpha=0.7, label=f'リファレンス点 x={ref_point[0]:,}')
        
        plt.xlabel('コスト')
        plt.ylabel('待ち時間')
        plt.title('最終的なパレートフロント比較 (NSGA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('nsga_final_pareto_front_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('nsga_final_pareto_front_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        print("最終的なパレートフロント比較グラフを保存しました:")
        print("- nsga_final_pareto_front_comparison.png")
        print("- nsga_final_pareto_front_comparison.pdf")


if __name__ == "__main__":
    main() 