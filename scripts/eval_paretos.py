import numpy as np
import matplotlib.pyplot as plt

def pareto_front(direction, data):
    if direction == "min":
        # 最小化問題におけるパレートフロントをreturnする
        pareto = []
        for i, point1 in enumerate(data):
            is_dominated = False
            for j, point2 in enumerate(data):
                if i != j:
                    # point2がpoint1を支配するかチェック（最小化なので、すべての目的で小さいか等しく、少なくとも一つで厳密に小さい）
                    if np.all([point2[k] <= point1[k] for k in range(len(point1))]) and np.any([point2[k] < point1[k] for k in range(len(point1))]):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto.append(point1)
        return np.array(pareto)
    else:
        # 最大化問題におけるパレートフロントをreturnする
        pareto = []
        for i, point1 in enumerate(data):
            is_dominated = False
            for j, point2 in enumerate(data):
                if i != j:
                    # point2がpoint1を支配するかチェック（最大化なので、すべての目的で大きいか等しく、少なくとも一つで厳密に大きい）
                    if np.all([point2[k] >= point1[k] for k in range(len(point1))]) and np.any([point2[k] > point1[k] for k in range(len(point1))]):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto.append(point1)
        return np.array(pareto)

pcn_2009=np.array([
[0, 105.125],
[124501, 0.25],
[103401, 2.34375],
[54171, 31.125],
[30049, 58.59375],
[54171, 31.125],
[61015, 27.8125],
[103401, 2.34375],
[7665, 90.875],
[47487, 38.71875],
[112150, 0.625],
[34854, 56.375],
[88499, 7.25],
[124501, 0.25],
[72859, 15.40625],
[61015, 27.8125],
[54171, 31.125],
[30049, 58.59375],
[72859, 15.40625],
[720, 104.34375],
[29981, 58.84375],
[103401, 2.34375]
])

yasuda=np.array([
    [64.81, 44933.00],
    [35.88, 62180.00],
    [20.66, 93317.00],
    [14.78, 92160.00],
    [15.59, 98527.00],
    [3.50, 108807.00],
    [46.56, 90120.00],
    [66.72, 35807.00],
    [0.44, 122803.00],
    [2.12, 103387.00],
    [36.50, 80954.00],
    [28.41, 80930.00],
    [43.84, 62394.00],
    [29.03, 63137.00],
    [16.19, 85175.00],
    [69.38, 46496.00],
    [9.03, 93396.00],
    [52.09, 63571.00],
    [65.78, 59421.00],
    [27.72, 91181.00],
    [33.75, 66557.00],
    [44.34, 55182.00],
    [46.38, 54558.00],
    [44.81, 61798.00],
    [42.81, 73576.00],
    [58.78, 34824.00],
    [47.31, 64021.00],
    [31.56, 69642.00],
    [24.03, 83448.00],
    [11.44, 95605.00],
    [32.62, 92585.00],
    [14.03, 100152.00],
])
#yasudaの各要素を入れ替える
yasuda = yasuda[:, ::-1]

yasuda_pareto=pareto_front("min", yasuda)

all_night2 = np.array([
    [7.814400e+04, 1.475000e+01],
    [5.017600e+04, 4.343750e+01],
    [1.179950e+05, 1.250000e-01],
    [7.369500e+04, 1.775000e+01],
    [4.491300e+04, 4.615625e+01],
    [4.430800e+04, 5.065625e+01],
    [9.420300e+04, 3.687500e+00],
    [1.199160e+05, 0.000000e+00],
    [4.179200e+04, 5.106250e+01],
    [1.064870e+05, 1.343750e+00],
    [6.652600e+04, 2.987500e+01],
    [1.154980e+05, 9.062500e-01],
    [9.947000e+04, 3.656250e+00],
    [8.727600e+04, 8.062500e+00],
    [5.328200e+04, 3.590625e+01],
    [8.283400e+04, 1.125000e+01],
    [5.735200e+04, 3.321875e+01],
    [1.098000e+05, 1.000000e+00],
    [1.093410e+05, 1.250000e+00],
    [1.032200e+05, 2.062500e+00],
    [1.038740e+05, 1.593750e+00],
    [6.128800e+04, 3.003125e+01],
    [1.014360e+05, 2.156250e+00],
    [5.114900e+04, 4.225000e+01],
    [6.714700e+04, 2.078125e+01],
    [5.492400e+04, 3.406250e+01],
    [8.534100e+04, 9.687500e+00],
    [7.623100e+04, 1.618750e+01],
    [1.001650e+05, 2.593750e+00],
    [2.620800e+04, 6.443750e+01],
    [1.011160e+05, 2.375000e+00],
    [1.162400e+05, 7.500000e-01],
    [3.077600e+04, 6.428125e+01],
    [8.998000e+04, 4.250000e+00],
    [7.142900e+04, 1.875000e+01],
    [6.032200e+04, 3.259375e+01],
    [3.964900e+04, 5.325000e+01],
    [0.000000e+00, 1.051250e+02],
    [2.325000e+04, 6.806250e+01],
    [1.237600e+04, 1.001875e+02],
    [6.890200e+04, 2.068750e+01],
    [7.873800e+04, 1.156250e+01],
    [1.747500e+04, 8.115625e+01],
    [3.144800e+04, 6.409375e+01],
    [6.673600e+04, 2.690625e+01],
    [7.456600e+04, 1.715625e+01]
])

nsga_1509=np.array([
[130875, 0.00],
[9252, 102.41],
[25518, 85.41],
[38066, 56.72],
[39262, 52.25],
[96845, 3.53],
[51259, 47.00],
[93934, 9.06],
[111414, 1.41],
[78820, 19.62],
[86970, 16.56],
[55701, 36.47],
[113017, 0.31],
[54106, 42.75],
[87479, 16.16],
[61451, 35.22],
[64866, 29.31],
[72834, 21.44],
[61862, 34.53],
[54203, 39.94],
[64276, 29.91],
[63933, 31.81],
[66282, 22.94],
[87917, 10.44],
[94876, 7.50]])

# 分析用の変数名設定
nsga_pareto = nsga_1509.copy()
pcn_data = pcn_2009.copy()
yasuda = yasuda_pareto.copy()
all_solutions = all_night2.copy()

def normalize_data(datasets):
    """
    理想点-ナディア点正規化により全データを0-1スケールに変換
    datasets: 辞書形式のデータセット {name: data}
    返り値: 正規化されたデータセット, 理想点, ナディア点
    """
    # 全データを結合してグローバルな理想点とナディア点を計算
    all_data = []
    for data in datasets.values():
        all_data.extend(data)
    all_data = np.array(all_data)
    
    # 理想点（各目的の最小値）とナディア点（各目的の最大値）
    ideal_point = np.min(all_data, axis=0)
    nadir_point = np.max(all_data, axis=0)
    
    print(f"理想点 (Ideal Point): [{ideal_point[0]:.2f}, {ideal_point[1]:.2f}]")
    print(f"ナディア点 (Nadir Point): [{nadir_point[0]:.2f}, {nadir_point[1]:.2f}]")
    print()
    
    # 各データセットを正規化
    normalized_datasets = {}
    for name, data in datasets.items():
        data = np.array(data)
        # 0除算を避けるため、範囲が0の場合は1に設定
        ranges = nadir_point - ideal_point
        ranges[ranges == 0] = 1.0
        
        normalized_data = (data - ideal_point) / ranges
        normalized_datasets[name] = normalized_data
        
        print(f"{name} データ正規化完了:")
        print(f"  元の範囲: [{np.min(data, axis=0)[0]:.2f}, {np.min(data, axis=0)[1]:.2f}] - [{np.max(data, axis=0)[0]:.2f}, {np.max(data, axis=0)[1]:.2f}]")
        print(f"  正規化後: [{np.min(normalized_data, axis=0)[0]:.3f}, {np.min(normalized_data, axis=0)[1]:.3f}] - [{np.max(normalized_data, axis=0)[0]:.3f}, {np.max(normalized_data, axis=0)[1]:.3f}]")
        print()
    
    return normalized_datasets, ideal_point, nadir_point

def calc_hypervolume(points, ref_point):
    """
    2目的最小化問題におけるハイパーボリュームを計算
    points: パレートフロントの点群 (N x 2)
    ref_point: 参照点 [x_ref, y_ref] (すべての点より劣る座標)
    """
    if len(points) == 0:
        return 0.0
    
    points = np.array(points)
    ref_point = np.array(ref_point)
    
    # 第1目的でソート
    sorted_points = points[np.argsort(points[:, 0])]
    
    hypervolume = 0.0
    prev_x = ref_point[0]
    
    for point in sorted_points:
        # 現在の点と参照点で長方形の面積を計算
        width = point[0] - prev_x
        height = ref_point[1] - point[1]
        
        if width > 0 and height > 0:
            hypervolume += width * height
        
        prev_x = point[0]
    
    return hypervolume

def calc_crowding_distance(points):
    """
    Crowding Distanceを計算（パレートフロント内の多様性指標）
    points: パレートフロントの点群 (N x M) Mは目的数
    返り値: 各点のcrowding distance配列
    """
    if len(points) <= 2:
        return np.full(len(points), float('inf'))
    
    points = np.array(points)
    n_points, n_objectives = points.shape
    distances = np.zeros(n_points)
    
    for obj_idx in range(n_objectives):
        # 目的ごとにソート
        sorted_indices = np.argsort(points[:, obj_idx])
        obj_values = points[sorted_indices, obj_idx]
        
        # 端点は無限大の距離を設定
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # 正規化のため目的値の範囲を計算
        obj_range = obj_values[-1] - obj_values[0]
        
        if obj_range > 0:  # 同一値でない場合のみ計算
            for i in range(1, n_points - 1):
                idx = sorted_indices[i]
                if distances[idx] != float('inf'):  # 端点でない場合
                    # 前後の点との距離を正規化して加算
                    distance = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                    distances[idx] += distance
    
    return distances

def calc_generational_distance(points, ref_points):
    """
    Generational Distance (GD)を計算
    points: 近似パレートフロントの点群 (N x M)
    ref_points: 真のパレートフロントの点群 (K x M)
    返り値: GD値（平均最小距離）
    """
    if len(points) == 0 or len(ref_points) == 0:
        return float('inf')
    
    points = np.array(points)
    ref_points = np.array(ref_points)
    
    min_distances = []
    
    for point in points:
        # 各近似点について、真のフロント上の最近点への距離を計算
        distances = np.sqrt(np.sum((ref_points - point) ** 2, axis=1))
        min_distance = np.min(distances)
        min_distances.append(min_distance)
    
    # 平均距離を返す
    return np.mean(min_distances)

def calc_igd_plus(points, ref_points):
    """
    IGD+を計算（最小化問題用）
    points: 近似パレートフロントの点群 (N x M)
    ref_points: 真のパレートフロントの点群 (K x M)
    返り値: IGD+値（平均最小距離）
    """
    if len(points) == 0 or len(ref_points) == 0:
        return float('inf')
    
    points = np.array(points)
    ref_points = np.array(ref_points)
    
    min_distances = []
    
    for ref_point in ref_points:
        # 各真のフロント点について、近似フロント上の最近点への距離を計算
        distances = []
        
        for point in points:
            # IGD+では、pointがref_pointをweakly dominateする場合、
            # modified distanceを使用（最小化問題用）
            if np.all(point <= ref_point):  # pointがref_pointをweakly dominate
                # dominated caseの距離計算
                distance = np.sqrt(np.sum(np.maximum(0, point - ref_point) ** 2))
            else:
                # non-dominated caseの通常のユークリッド距離
                distance = np.sqrt(np.sum((point - ref_point) ** 2))
            
            distances.append(distance)
        
        min_distance = np.min(distances)
        min_distances.append(min_distance)
    
    # 平均距離を返す
    return np.mean(min_distances)  

def calc_epsilon_indicator(points, ref_points):
    """
    Epsilon Indicatorを計算（最小化問題用）
    points: 近似パレートフロントの点群 (N x M)
    ref_points: 真のパレートフロントの点群 (K x M)  
    返り値: Epsilon Indicator値（最小の倍率値）
    """
    if len(points) == 0 or len(ref_points) == 0:
        return float('inf')
    
    points = np.array(points)
    ref_points = np.array(ref_points)
    
    epsilon_values = []
    
    for ref_point in ref_points:
        # 各真のフロント点について、近似フロントで支配するのに必要な最小倍率を計算
        min_epsilon = float('inf')
        
        for point in points:
            # pointがref_pointをepsilon倍で支配できるかチェック（最小化問題）
            # point <= epsilon * ref_point となる最小のepsilonを求める
            if np.all(ref_point > 0):  # 0除算を避ける
                epsilon_needed = np.max(point / ref_point)
            else:
                # ref_pointに0が含まれる場合の処理
                epsilon_ratios = []
                for i in range(len(point)):
                    if ref_point[i] > 0:
                        epsilon_ratios.append(point[i] / ref_point[i])
                    elif point[i] == 0 and ref_point[i] == 0:
                        epsilon_ratios.append(1.0)  # 両方0の場合は1とする
                    elif point[i] > 0:
                        epsilon_ratios.append(float('inf'))
                
                if len(epsilon_ratios) > 0:
                    epsilon_needed = max(epsilon_ratios)
                else:
                    epsilon_needed = float('inf')
            
            if not np.isinf(epsilon_needed):
                min_epsilon = min(min_epsilon, epsilon_needed)
        
        # 有限の値のみを記録
        if not np.isinf(min_epsilon):
            epsilon_values.append(min_epsilon)
    
    # 最大のepsilon値を返す（最も支配が困難な点の倍率）
    if len(epsilon_values) > 0:
        return np.max(epsilon_values)
    else:
        return float('inf')
    
def main():
    """
    全ての点群に対して評価指標を計算し、わかりやすく表示
    """
    print("=" * 80)
    print("パレートフロント評価指標の計算結果（正規化版）")
    print("=" * 80)
    
    # データセットの定義
    original_datasets = {
        "PCN": pcn_data,
        "NSGA2": nsga_pareto,
        "All": all_solutions,
        "Yasuda": yasuda
    }
    
    # データを正規化
    print("1. データ正規化")
    print("-" * 60)
    datasets, ideal_point, nadir_point = normalize_data(original_datasets)
    
    # 正規化後の参照点設定（ナディア点にマージンを加える）
    ref_point = [1, 1]  # 正規化後なので各軸の最大値1.0に10%マージン
    
    print(f"正規化後参照点 (Normalized Reference Point): [{ref_point[0]:.1f}, {ref_point[1]:.1f}]")
    print()
    
    # 各パレートフロントの抽出（正規化後データで）
    pareto_fronts = {}
    for name, data in datasets.items():
        pareto_fronts[name] = pareto_front("min", data)
    
    # 2. パレートフロント情報の表示
    print("2. パレートフロント情報")
    print("-" * 60)
    print(f"{'データセット名':<20} {'全点数':<8} {'パレート点数':<12} {'削減率(%)':<10}")
    print("-" * 60)
    
    for name, data in datasets.items():
        total_points = len(data)
        pareto_points = len(pareto_fronts[name])
        reduction_rate = (1 - pareto_points / total_points) * 100
        print(f"{name:<20} {total_points:<8} {pareto_points:<12} {reduction_rate:<10.1f}")
    
    print()
    
    # 3. ハイパーボリューム計算
    print("3. ハイパーボリューム (HyperVolume) - 正規化後")
    print("-" * 50)
    print(f"{'データセット名':<20} {'HV値':<15}")
    print("-" * 50)
    
    hv_results = {}
    for name, pf in pareto_fronts.items():
        hv = calc_hypervolume(pf, ref_point)
        hv_results[name] = hv
        print(f"{name:<20} {hv:<15.4f}")
    
    print()
    
    # 4. クラウディング距離
    print("4. クラウディング距離 (Crowding Distance) - 正規化後")
    print("-" * 70)
    print(f"{'データセット名':<20} {'平均':<10} {'最小':<10} {'最大':<10} {'標準偏差':<10}")
    print("-" * 70)
    
    for name, pf in pareto_fronts.items():
        if len(pf) > 0:
            cd = calc_crowding_distance(pf)
            # inf値を除外して統計計算
            cd_finite = cd[np.isfinite(cd)]
            if len(cd_finite) > 0:
                mean_cd = np.mean(cd_finite)
                min_cd = np.min(cd_finite)
                max_cd = np.max(cd_finite)
                std_cd = np.std(cd_finite)
                print(f"{name:<20} {mean_cd:<10.4f} {min_cd:<10.4f} {max_cd:<10.4f} {std_cd:<10.4f}")
            else:
                print(f"{name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print()
    
    # 5. 世代距離（All_Night2を真のフロントとして使用）
    print("5. 世代距離 (Generational Distance) - Allを真のフロントとして (正規化後)")
    print("-" * 70)
    print(f"{'データセット名':<20} {'GD値':<15}")
    print("-" * 70)
    
    true_front = pareto_fronts["All"]
    for name, pf in pareto_fronts.items():
        if name != "All":  # 自分自身との比較は除外
            gd = calc_generational_distance(pf, true_front)
            print(f"{name:<20} {gd:<15.4f}")
    
    print()
    
    # 6. IGD+（逆世代距離プラス）
    print("6. IGD+ (Inverted Generational Distance Plus) - Allを真のフロントとして (正規化後)")
    print("-" * 80)
    print(f"{'データセット名':<20} {'IGD+値':<15}")
    print("-" * 80)
    
    for name, pf in pareto_fronts.items():
        if name != "All":  # 自分自身との比較は除外
            igd_plus = calc_igd_plus(pf, true_front)
            print(f"{name:<20} {igd_plus:<15.4f}")
    
    print()
    
    # 7. Epsilon Indicator
    print("7. Epsilon Indicator - Allを真のフロントとして (正規化後)")
    print("-" * 70)
    print(f"{'データセット名':<20} {'Epsilon値':<15}")
    print("-" * 70)
    
    for name, pf in pareto_fronts.items():
        if name != "All":  # 自分自身との比較は除外
            epsilon = calc_epsilon_indicator(pf, true_front)
            print(f"{name:<20} {epsilon:<15.4f}")
    
    print()
    
    # 8. 総合評価
    print("8. 総合評価")
    print("-" * 40)
    print("【ハイパーボリューム順位】（大きい方が良い）")
    hv_ranking = sorted(hv_results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, hv) in enumerate(hv_ranking, 1):
        print(f"{i}位: {name} (HV = {hv:.4f})")
    
    print("\n【推奨】")
    print(f"最高HV: {hv_ranking[0][0]} - 最も広い領域を支配")
    if len(hv_ranking) > 1:
        print(f"2位HV: {hv_ranking[1][0]} - バランスの取れた解集合")

    # 正規化されたデータでプロット
    plt.figure(figsize=(10, 8))
    
    # 正規化されたパレートフロントをプロット
    for name, pf in pareto_fronts.items():
        if len(pf) > 0:
            plt.scatter(pf[:, 0], pf[:, 1], label=f'{name} (Pareto)', marker='o', s=50, alpha=0.7)
    
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('Pareto Front')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.1, 1.2)
    
    # 参照点をプロット
    plt.scatter(ref_point[0], ref_point[1], marker='*', s=200, c='red', label='reference point')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('eval_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n正規化後のプロットを 'eval_normalized.png' として保存しました")
    print(f"元の座標系での理想点: [{ideal_point[0]:.2f}, {ideal_point[1]:.2f}]")
    print(f"元の座標系でのナディア点: [{nadir_point[0]:.2f}, {nadir_point[1]:.2f}]")

if __name__ == "__main__":
    main()
