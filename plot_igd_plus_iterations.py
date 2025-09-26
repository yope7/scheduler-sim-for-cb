import matplotlib.pyplot as plt
import numpy as np
from extract_pareto_values import main

def plot_igd_plus_iterations():
    """iterationごとのIGD+値をプロットする"""
    
    # データを取得
    all_iteration_data, igd_plus_list, igd_plus_array, true_pareto = main()
    
    # iteration番号を抽出（数値のみ）
    iteration_numbers = []
    for iteration_dir in all_iteration_data.keys():
        # "iteration_XXX" から数値部分を抽出
        iteration_num = int(iteration_dir.split('_')[1])
        iteration_numbers.append(iteration_num)
    
    # iteration番号とIGD+値をソート
    sorted_data = sorted(zip(iteration_numbers, igd_plus_list))
    sorted_iterations, sorted_igd_plus = zip(*sorted_data)
    
    # プロット設定
    plt.figure(figsize=(12, 8))
    
    # IGD+値をプロット
    plt.plot(sorted_iterations, sorted_igd_plus, 'b-o', linewidth=2, markersize=6, 
             label='IGD+ Values', alpha=0.8)
    
    # グリッドとラベル
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('IGD+ Value', fontsize=12)
    plt.title('IGD+ Values across Iterations', fontsize=14, fontweight='bold')
    
    # 統計情報をテキストボックスで表示
    stats_text = f"""Statistics:
Mean: {np.mean(sorted_igd_plus):.4f}
Std: {np.std(sorted_igd_plus):.4f}
Min: {np.min(sorted_igd_plus):.4f}
Max: {np.max(sorted_igd_plus):.4f}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 最小値と最大値を強調
    min_idx = np.argmin(sorted_igd_plus)
    max_idx = np.argmax(sorted_igd_plus)
    
    plt.scatter(sorted_iterations[min_idx], sorted_igd_plus[min_idx], 
               color='green', s=100, marker='*', label=f'Best (Iter {sorted_iterations[min_idx]})')
    plt.scatter(sorted_iterations[max_idx], sorted_igd_plus[max_idx], 
               color='red', s=100, marker='x', label=f'Worst (Iter {sorted_iterations[max_idx]})')
    
    # レジェンド
    plt.legend()
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    plt.savefig('igd_plus_iterations.png', dpi=300, bbox_inches='tight')
    print(f"\nプロットを保存しました: igd_plus_iterations.png")
    
    # 表示
    plt.show()
    
    # 詳細な分析結果を出力
    print("\n" + "="*60)
    print("詳細な分析結果:")
    print("="*60)
    
    print(f"最良のIGD+値: {np.min(sorted_igd_plus):.6f} (Iteration {sorted_iterations[min_idx]})")
    print(f"最悪のIGD+値: {np.max(sorted_igd_plus):.6f} (Iteration {sorted_iterations[max_idx]})")
    
    # 改善傾向の分析
    first_half = sorted_igd_plus[:len(sorted_igd_plus)//2]
    second_half = sorted_igd_plus[len(sorted_igd_plus)//2:]
    
    print(f"\n前半のIGD+平均値: {np.mean(first_half):.6f}")
    print(f"後半のIGD+平均値: {np.mean(second_half):.6f}")
    
    if np.mean(second_half) < np.mean(first_half):
        print("→ 全体的に改善傾向が見られます")
    else:
        print("→ 全体的な改善傾向は明確ではありません")
    
    # 上位5つの結果
    top_5_indices = np.argsort(sorted_igd_plus)[:5]
    print(f"\n上位5つの結果 (IGD+が小さい順):")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"{i}. Iteration {sorted_iterations[idx]}: IGD+ = {sorted_igd_plus[idx]:.6f}")
    
    return sorted_iterations, sorted_igd_plus

if __name__ == "__main__":
    iterations, igd_values = plot_igd_plus_iterations() 