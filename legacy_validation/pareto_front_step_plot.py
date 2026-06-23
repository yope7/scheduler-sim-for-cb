import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import json
import os

def hv_polygon_post(front, ref):
    """
    パレートフロント用のハイパーボリューム領域ポリゴンを生成（右→下ステップ）
    
    Args:
        front: パレートフロントの点群 (N, 2)
        ref: 参照点 [x_ref, y_ref]
    
    Returns:
        np.array: ポリゴンの頂点座標
    """
    # x軸で昇順ソート
    f = front[np.argsort(front[:,0])]
    
    # 重複するx座標を除去（必要に応じて）
    unique_mask = np.r_[True, np.diff(f[:,0]) > 0]
    f = f[unique_mask]
    
    # ポリゴンの頂点を構築
    verts = [(ref[0], ref[1]), (f[0,0], ref[1])]   # 右端から左へ水平
    
    for i in range(len(f)):
        verts.append((f[i,0], f[i,1]))            # 前線の点
        if i < len(f)-1:
            verts.append((f[i+1,0], f[i,1]))      # 右へ水平→次で落ちる（右→下）
    
    verts += [(ref[0], f[-1,1]), (ref[0], ref[1])]
    return np.array(verts)

def plot_pareto_step_comparison(pf1, pf2, ref_point, title="Pareto Front Comparison", 
                               labels=["Pareto Front 1", "Pareto Front 2"],
                               colors=['blue', 'red'], hatches=['//', '\\\\']):
    """
    2つのパレートフロントをステップ線とハッチ領域で比較可視化
    
    Args:
        pf1, pf2: パレートフロント点群 (N, 2)
        ref_point: 参照点 [x_ref, y_ref]
        title: グラフタイトル
        labels: 凡例ラベル
        colors: 線の色
        hatches: ハッチパターン
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. ステップ線（右→下）
    ax.step(pf1[:,0], pf1[:,1], where='post', linewidth=2.2, 
            color=colors[0], label=labels[0], zorder=4)
    ax.step(pf2[:,0], pf2[:,1], where='post', linewidth=2.2, 
            color=colors[1], label=labels[1], zorder=4)
    
    # 2. ハッチ領域も右→下で一致させる
    poly1 = Polygon(hv_polygon_post(pf1, ref_point), closed=True, 
                    fill=True, alpha=0.15, hatch=hatches[0], 
                    facecolor=colors[0], edgecolor='none', zorder=1)
    poly2 = Polygon(hv_polygon_post(pf2, ref_point), closed=True, 
                    fill=True, alpha=0.15, hatch=hatches[1], 
                    facecolor=colors[1], edgecolor='none', zorder=2)
    
    ax.add_patch(poly1)
    ax.add_patch(poly2)
    
    # 3. 散布図でパレート点を強調
    ax.scatter(pf1[:,0], pf1[:,1], c=colors[0], s=80, 
               alpha=0.8, edgecolor='black', linewidth=1, zorder=5)
    ax.scatter(pf2[:,0], pf2[:,1], c=colors[1], s=80, 
               alpha=0.8, edgecolor='black', linewidth=1, zorder=5)
    
    # 4. 参照点をマーク
    ax.scatter(ref_point[0], ref_point[1], c='black', s=120, 
               marker='x', linewidth=3, label='Reference Point', zorder=6)
    
    # 5. グラフの装飾
    ax.set_xlabel('Objective 1 (Cost)', fontsize=14)
    ax.set_ylabel('Objective 2 (Waiting Time)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # 軸範囲の調整
    all_points = np.vstack([pf1, pf2])
    x_margin = (np.max(all_points[:,0]) - np.min(all_points[:,0])) * 0.1
    y_margin = (np.max(all_points[:,1]) - np.min(all_points[:,1])) * 0.1
    
    ax.set_xlim(np.min(all_points[:,0]) - x_margin, ref_point[0] + x_margin)
    ax.set_ylim(np.min(all_points[:,1]) - y_margin, ref_point[1] + y_margin)
    
    plt.tight_layout()
    return fig, ax

def plot_single_pareto_step(pf, ref_point, title="Pareto Front", 
                           label="Pareto Front", color='blue', hatch='//'):
    """
    単一のパレートフロントをステップ線とハッチ領域で可視化
    
    Args:
        pf: パレートフロント点群 (N, 2)
        ref_point: 参照点 [x_ref, y_ref]
        title: グラフタイトル
        label: 凡例ラベル
        color: 線の色
        hatch: ハッチパターン
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. ステップ線（右→下）
    ax.step(pf[:,0], pf[:,1], where='post', linewidth=2.5, 
            color=color, label=label, zorder=4)
    
    # 2. ハッチ領域
    poly = Polygon(hv_polygon_post(pf, ref_point), closed=True, 
                   fill=True, alpha=0.2, hatch=hatch, 
                   facecolor=color, edgecolor='none', zorder=1)
    ax.add_patch(poly)
    
    # 3. 散布図でパレート点を強調
    ax.scatter(pf[:,0], pf[:,1], c=color, s=100, 
               alpha=0.9, edgecolor='black', linewidth=1.5, zorder=5)
    
    # 4. 参照点をマーク
    ax.scatter(ref_point[0], ref_point[1], c='black', s=150, 
               marker='x', linewidth=4, label='Reference Point', zorder=6)
    
    # 5. グラフの装飾
    ax.set_xlabel('Objective 1 (Cost)', fontsize=14)
    ax.set_ylabel('Objective 2 (Waiting Time)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # 軸範囲の調整
    x_margin = (np.max(pf[:,0]) - np.min(pf[:,0])) * 0.1
    y_margin = (np.max(pf[:,1]) - np.min(pf[:,1])) * 0.1
    
    ax.set_xlim(np.min(pf[:,0]) - x_margin, ref_point[0] + x_margin)
    ax.set_ylim(np.min(pf[:,1]) - y_margin, ref_point[1] + y_margin)
    
    plt.tight_layout()
    return fig, ax

def load_pareto_data_from_json(json_file_path):
    """
    JSONファイルからパレートフロントデータを読み込む
    
    Args:
        json_file_path: JSONファイルのパス
        
    Returns:
        np.array: パレートフロント点群 (N, 2)
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # JSONの構造に応じて調整が必要
    if 'pareto_front' in data:
        pareto_points = data['pareto_front']
    elif 'pareto_results' in data:
        pareto_points = [[r['cost'], r['waiting_time']] for r in data['pareto_results']]
    else:
        # データ構造を推測
        keys = list(data.keys())
        print(f"利用可能なキー: {keys}")
        # 最初のキーのデータを使用
        first_key = keys[0]
        if isinstance(data[first_key], list):
            pareto_points = data[first_key]
        else:
            raise ValueError("パレートフロントデータが見つかりません")
    
    return np.array(pareto_points)

def load_pareto_data_from_npz(npz_file_path, key='pareto_front'):
    """
    NPZファイルからパレートフロントデータを読み込む
    
    Args:
        npz_file_path: NPZファイルのパス
        key: データのキー名
        
    Returns:
        np.array: パレートフロント点群 (N, 2)
    """
    data = np.load(npz_file_path)
    
    # 利用可能なキーを確認
    available_keys = list(data.keys())
    print(f"利用可能なキー: {available_keys}")
    
    if key in data:
        return data[key]
    elif 'pareto_costs' in data and 'pareto_waiting_times' in data:
        # コストと待ち時間が別々に保存されている場合
        costs = data['pareto_costs']
        waiting_times = data['pareto_waiting_times']
        return np.column_stack([costs, waiting_times])
    else:
        # 最初のキーを使用
        first_key = available_keys[0]
        return data[first_key]

def plot_real_data_comparison():
    """実際のデータを使用した比較プロット"""
    
    # データディレクトリの確認
    data_dir = "data"
    distributed_dir = "distributed_pareto_results"
    
    if os.path.exists(data_dir):
        print(f"データディレクトリが見つかりました: {data_dir}")
        for file in os.listdir(data_dir):
            if file.endswith(('.npz', '.json')):
                print(f"  - {file}")
    
    if os.path.exists(distributed_dir):
        print(f"分散結果ディレクトリが見つかりました: {distributed_dir}")
        json_files = [f for f in os.listdir(distributed_dir) if f.endswith('.json')]
        print(f"  JSONファイル数: {len(json_files)}")
        if json_files:
            print(f"  例: {json_files[:3]}")
    
    # サンプルファイルがあれば読み込んでプロット
    sample_npz = "data/accumulated_pareto_fronts.npz"
    if os.path.exists(sample_npz):
        try:
            print(f"\nNPZファイルを読み込み中: {sample_npz}")
            pf_data = load_pareto_data_from_npz(sample_npz)
            
            # 参照点を自動設定
            ref_point = [np.max(pf_data[:,0]) * 1.1, np.max(pf_data[:,1]) * 1.1]
            
            fig, ax = plot_single_pareto_step(
                pf_data, ref_point,
                title="Real Pareto Front with Step Lines",
                label="Accumulated Pareto Front",
                color='#4169E1',
                hatch='//'
            )
            
            fig.savefig('real_pareto_step.png', dpi=300, bbox_inches='tight')
            print("実データのプロットを保存しました: real_pareto_step.png")
            
        except Exception as e:
            print(f"NPZファイルの読み込みエラー: {e}")
    
    # 分散結果のJSONファイルも試す
    if os.path.exists(distributed_dir):
        json_files = [f for f in os.listdir(distributed_dir) if f.endswith('.json')]
        if json_files:
            try:
                sample_json = os.path.join(distributed_dir, json_files[0])
                print(f"\nJSONファイルを読み込み中: {sample_json}")
                pf_data = load_pareto_data_from_json(sample_json)
                
                ref_point = [np.max(pf_data[:,0]) * 1.1, np.max(pf_data[:,1]) * 1.1]
                
                fig, ax = plot_single_pareto_step(
                    pf_data, ref_point,
                    title="Distributed Pareto Results with Step Lines",
                    label="Distributed Results",
                    color='#DC143C',
                    hatch='\\\\'
                )
                
                fig.savefig('distributed_pareto_step.png', dpi=300, bbox_inches='tight')
                print("分散結果のプロットを保存しました: distributed_pareto_step.png")
                
            except Exception as e:
                print(f"JSONファイルの読み込みエラー: {e}")

def demo_step_plot():
    """デモンストレーション用のサンプルプロット"""
    
    # サンプルデータの生成
    np.random.seed(42)
    
    # パレートフロント1（より良い性能）
    pf1_x = np.array([1.0, 2.5, 4.0, 6.5, 9.0])
    pf1_y = np.array([9.0, 6.5, 4.5, 2.5, 1.0])
    pf1 = np.column_stack([pf1_x, pf1_y])
    
    # パレートフロント2（劣る性能）
    pf2_x = np.array([1.5, 3.0, 5.0, 7.0, 9.5])
    pf2_y = np.array([9.5, 7.5, 5.5, 3.5, 1.5])
    pf2 = np.column_stack([pf2_x, pf2_y])
    
    # 参照点（右上）
    ref_point = [10.0, 10.0]
    
    # 比較プロット
    fig1, ax1 = plot_pareto_step_comparison(
        pf1, pf2, ref_point,
        title="Pareto Front Comparison with Step Lines",
        labels=["Algorithm A", "Algorithm B"],
        colors=['#2E8B57', '#DC143C'],  # SeaGreen, Crimson
        hatches=['//', '\\\\']
    )
    
    # 単独プロット
    fig2, ax2 = plot_single_pareto_step(
        pf1, ref_point,
        title="Single Pareto Front with Step Lines",
        label="Optimized Solutions",
        color='#4169E1',  # RoyalBlue
        hatch='//'
    )
    
    # 保存
    fig1.savefig('pareto_comparison_step.png', dpi=300, bbox_inches='tight')
    fig2.savefig('pareto_single_step.png', dpi=300, bbox_inches='tight')
    
    print("デモプロットを生成しました:")
    print("- pareto_comparison_step.png")
    print("- pareto_single_step.png")
    
    plt.show()

if __name__ == "__main__":
    print("=== パレートフロント ステップ線プロット ===")
    print("\n1. デモプロットの生成")
    demo_step_plot()
    
    print("\n2. 実データの処理")
    plot_real_data_comparison() 