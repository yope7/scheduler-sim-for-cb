import os
import re
import glob
import numpy as np
from pathlib import Path
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
"""
PCN

"""
def calc_hypervolume_with_pymoo(points, ref_point):
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


def extract_real_values_from_file(file_path):
    """ファイルから実数値空間の非支配解の値を抽出"""
    real_values = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # "=== 実数値空間の非支配解 (Iter X) ===" 以降の行を検索
        pattern = r'=== 実数値空間の非支配解 \(Iter \d+\) ===\n(.*?)(?=\n\n|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            lines = match.group(1).strip().split('\n')
            for line in lines:
                if line.startswith('解'):
                    # "解X: [value1, value2]" の形式から値を抽出
                    value_match = re.search(r'\[([^\]]+)\]', line)
                    if value_match:
                        values_str = value_match.group(1)
                        # カンマで区切られた値を数値に変換
                        values = [float(x.strip()) for x in values_str.split(',')]
                        real_values.append(values)
    
    except Exception as e:
        print(f"ファイル {file_path} の読み込みエラー: {e}")
    
    # numpy配列に変換
    if real_values:
        return np.array(real_values)
    else:
        return np.array([])

def calculate_combined_pareto_front(points1, points2):
    """2つの点群を結合してパレートフロントを計算"""
    if len(points1) == 0 and len(points2) == 0:
        return np.array([])
    elif len(points1) == 0:
        return points2
    elif len(points2) == 0:
        return points1
    
    # 2つの点群を結合
    combined_points = np.vstack([points1, points2])
    
    # パレートフロントを計算（支配関係をチェック）
    pareto_front = []
    
    for i, point in enumerate(combined_points):
        is_pareto = True
        
        for j, other_point in enumerate(combined_points):
            if i != j:
                # 点iが点jに支配されているかチェック
                if np.all(other_point <= point) and np.any(other_point < point):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_front.append(point)
    
    return np.array(pareto_front)

def main():
    # 指定されたディレクトリ
    base_dir = "execution_20250818_011822_final"
    
    # iteration_**ディレクトリを取得
    iteration_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('iteration_')])
    
    print(f"見つかったiterationディレクトリ: {len(iteration_dirs)}個")
    print("=" * 50)
    
    # 各iterationディレクトリから値を抽出
    all_iteration_data = {}
    hv_list = []  # HV値を格納するリスト
    gd_list = []  # GD値を格納するリスト
    igd_plus_list = []  # IGD+値を格納するリスト
    previous_pareto_front = None
    
    # 真のパレートフロントを取得
    true_pareto_front = get_true_pareto_front()
    print(f"真のパレートフロント点数: {len(true_pareto_front)}")
    
    # リファレンスポイントを設定
    ref_point = np.array([200000, 200])  # データ範囲に適した値
    
    for iteration_dir in iteration_dirs:
        iteration_path = os.path.join(base_dir, iteration_dir)
        
        # ディレクトリ内のファイルを検索
        files = glob.glob(os.path.join(iteration_path, "*.txt"))
        
        for file_path in files:
            if "pareto_front_details" in file_path:
                current_values = extract_real_values_from_file(file_path)
                if current_values.size > 0:  # 空の配列でない場合
                    all_iteration_data[iteration_dir] = current_values
                    
                    # 前のiterationの解と現在の解を結合してパレートフロントを計算
                    if previous_pareto_front is not None and len(previous_pareto_front) > 0:
                        print(f"{iteration_dir}: 前のiterationの解数: {len(previous_pareto_front)}")
                        combined_pareto_front = calculate_combined_pareto_front(previous_pareto_front, current_values)
                        print(f"{iteration_dir}: 結合後のパレートフロント解数: {len(combined_pareto_front)}")
                    else:
                        combined_pareto_front = current_values
                        print(f"{iteration_dir}: 前のiterationの解なし")
                    
                    # 参照点を設定（すべての点より劣る座標）
                    # データの範囲を確認して適切な参照点を設定
                    max_x = np.max(combined_pareto_front[:, 0])
                    max_y = np.max(combined_pareto_front[:, 1])
                    
                    # PymooのHVクラスを使用してHVを計算
                    hv = calc_hypervolume_with_pymoo(combined_pareto_front, ref_point)
                    #normalized hv
                    normalized_hv = hv / (ref_point[0] * ref_point[1])
                    hv_list.append(normalized_hv)
                    
                    # GD計算
                    gd_value = calculate_gd(combined_pareto_front, true_pareto_front, ref_point)
                    gd_list.append(gd_value)
                    
                    # IGD+計算
                    igd_plus_value = calculate_igd_plus(combined_pareto_front, true_pareto_front, ref_point)
                    igd_plus_list.append(igd_plus_value)
                    
                    print(f"{iteration_dir}: {current_values.shape[0]}個の解を抽出 (形状: {current_values.shape}), 結合後HV: {hv:.6f}, GD: {gd_value:.6f}, IGD+: {igd_plus_value:.6f}")
                    
                    # 次のiterationのために現在の結合されたパレートフロントを保存
                    previous_pareto_front = combined_pareto_front
                    break
    
    print("\n" + "=" * 50)
    print("指標計算結果:")
    print("=" * 50)
    
    # 各指標のリストを表示
    for i, (iteration, hv, gd, igd_plus) in enumerate(zip(all_iteration_data.keys(), hv_list, gd_list, igd_plus_list)):
        print(f"{iteration}: HV = {hv:.6f}, GD = {gd:.6f}, IGD+ = {igd_plus:.6f}")
    
    print(f"\nHVリストの長さ: {len(hv_list)}")
    print(f"HVの平均値: {np.mean(hv_list):.6f}")
    print(f"HVの標準偏差: {np.std(hv_list):.6f}")
    print(f"HVの最小値: {np.min(hv_list):.6f}")
    print(f"HVの最大値: {np.max(hv_list):.6f}")
    
    print(f"\nGDリストの長さ: {len(gd_list)}")
    print(f"GDの平均値: {np.mean(gd_list):.6f}")
    print(f"GDの標準偏差: {np.std(gd_list):.6f}")
    print(f"GDの最小値: {np.min(gd_list):.6f}")
    print(f"GDの最大値: {np.max(gd_list):.6f}")
    
    print(f"\nIGD+リストの長さ: {len(igd_plus_list)}")
    print(f"IGD+の平均値: {np.mean(igd_plus_list):.6f}")
    print(f"IGD+の標準偏差: {np.std(igd_plus_list):.6f}")
    print(f"IGD+の最小値: {np.min(igd_plus_list):.6f}")
    print(f"IGD+の最大値: {np.max(igd_plus_list):.6f}")
    
    print("\n" + "=" * 50)
    print("抽出された実数値空間の非支配解 (numpy配列形式):")
    print("=" * 50)
    
    # 統計情報
    total_solutions = sum(values.shape[0] for values in all_iteration_data.values())
    print(f"\n総解数: {total_solutions}")
    print(f"処理したiteration数: {len(all_iteration_data)}")
    
    # 全データを1つの大きな配列に結合する例
    if all_iteration_data:
        all_solutions = np.vstack(list(all_iteration_data.values()))
        print(f"\n全解を結合した配列の形状: {all_solutions.shape}")
        print(f"全解を結合した配列のデータ型: {all_solutions.dtype}")
    
    # 各指標をnumpy配列に変換
    hv_array = np.array(hv_list)
    gd_array = np.array(gd_list)
    igd_plus_array = np.array(igd_plus_list)
    print(f"\nHV配列の形状: {hv_array.shape}")
    print(f"GD配列の形状: {gd_array.shape}")
    print(f"IGD+配列の形状: {igd_plus_array.shape}")
    
    # グラフ作成と保存
    print("\n" + "=" * 50)
    print("グラフ作成中...")
    
    # iteration番号を抽出（例: "iteration_005" -> 5）
    iteration_numbers = []
    for iteration_dir in iteration_dirs:
        if "iteration_" in iteration_dir:
            try:
                num = int(iteration_dir.split("_")[1])
                iteration_numbers.append(num)
            except (ValueError, IndexError):
                continue
    
    # データの長さを合わせる
    min_length = min(len(iteration_numbers), len(hv_list))
    iteration_numbers = iteration_numbers[:min_length]
    hv_values = hv_list[:min_length]
    gd_values = gd_list[:min_length]
    igd_plus_values = igd_plus_list[:min_length]
    
    # 3つの指標を並べて表示するサブプロット
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # サブプロット1: ハイパーボリューム
    ax1.plot(iteration_numbers, hv_values, marker='o', linewidth=2, markersize=8, 
             color='blue', label='Hypervolume')
    ax1.plot(iteration_numbers, hv_values, '--', alpha=0.3, color='red', 
             label='Expected Trend')
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Normalized Hypervolume', fontsize=14)
    ax1.set_title('Hypervolume vs Iteration (Cumulative Pareto Front)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # サブプロット2: GD
    ax2.plot(iteration_numbers, gd_values, marker='s', linewidth=2, markersize=8, 
             color='orange', label='Generational Distance')
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('GD', fontsize=14)
    ax2.set_title('Generational Distance vs Iteration', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # サブプロット3: IGD+
    ax3.plot(iteration_numbers, igd_plus_values, marker='^', linewidth=2, markersize=8, 
             color='red', label='IGD+')
    ax3.set_xlabel('Iteration', fontsize=14)
    ax3.set_ylabel('IGD+', fontsize=14)
    ax3.set_title('IGD+ vs Iteration', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.1)
    ax3.legend(fontsize=12)
    
    # x軸の目盛りを調整
    for ax in [ax1, ax2, ax3]:
        if len(iteration_numbers) > 0:
            ax.set_xticks(iteration_numbers[::max(1, len(iteration_numbers)//10)])
    
    plt.tight_layout()
    
    # グラフを保存
    plt.savefig('metrics_vs_iteration.png', dpi=300, bbox_inches='tight')
    plt.savefig('metrics_vs_iteration.pdf', bbox_inches='tight')
    
    print("グラフを保存しました:")
    print("- metrics_vs_iteration.png")
    print("- metrics_vs_iteration.pdf")
    
    # グラフを表示
    plt.show()
    
    # 最終的なパレートフロントの可視化
    if previous_pareto_front is not None and len(previous_pareto_front) > 0:
        print("\n最終的なパレートフロントの可視化...")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(true_pareto_front[:, 0], true_pareto_front[:, 1], 
                    c='red', s=50, label='真のパレートフロント', alpha=0.7, zorder=5)
        plt.scatter(previous_pareto_front[:, 0], previous_pareto_front[:, 1], 
                    c='blue', s=30, label='最終的なパレートフロント', alpha=0.6)
        plt.axhline(y=ref_point[1], color='gray', linestyle='--', alpha=0.7, label=f'リファレンス点 y={ref_point[1]}')
        plt.axvline(x=ref_point[0], color='gray', linestyle='--', alpha=0.7, label=f'リファレンス点 x={ref_point[0]:,}')
        plt.xlabel('コスト')
        plt.ylabel('待ち時間')
        plt.title('最終的なパレートフロント比較')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('final_pareto_front_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('final_pareto_front_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        print("最終的なパレートフロント比較グラフを保存しました:")
        print("- final_pareto_front_comparison.png")
        print("- final_pareto_front_comparison.pdf")
    
    return all_iteration_data, hv_list, hv_array, gd_list, gd_array, igd_plus_list, igd_plus_array

if __name__ == "__main__":
    all_iteration_data, hv_list, hv_array, gd_list, gd_array, igd_plus_list, igd_plus_array = main() 