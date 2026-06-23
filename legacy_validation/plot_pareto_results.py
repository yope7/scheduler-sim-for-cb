import matplotlib.pyplot as plt
import numpy as np

# データを手動で入力（テキストファイルから抽出）
costs = [74595.00, 83947.00, 62986.00, 47046.00, 39822.00, 106738.00, 59777.00, 62217.00, 
         120160.00, 101918.00, 83536.00, 83187.00, 85353.00, 62418.00, 73165.00, 95082.00, 
         83917.00, 78474.00, 61646.00, 87426.00, 75767.00, 87824.00, 90193.00, 99618.00, 
         86752.00, 86105.00, 62557.00, 83081.00, 95683.00, 92469.00, 75193.00, 70381.00]

wait_times = [34.56, 26.44, 58.50, 50.72, 57.00, 12.19, 41.31, 40.75, 1.00, 4.50, 42.84, 
              27.81, 15.78, 50.09, 43.38, 13.22, 20.34, 32.50, 39.78, 14.09, 33.06, 19.94, 
              18.75, 25.50, 20.12, 34.00, 42.62, 29.66, 18.94, 28.88, 40.41, 31.75]

# パレート最適解のインデックス（1-indexedから0-indexedに変換）
pareto_optimal_indices = [0, 1]  # No.1とNo.2がパレート最適

# プロットの設定
plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'DejaVu Sans'  # 日本語対応のフォント

# 全解をプロット
plt.scatter(costs, wait_times, c='lightblue', s=100, alpha=0.7, label='全解', edgecolors='black', linewidth=0.5)

# パレート最適解を強調
pareto_costs = [costs[i] for i in pareto_optimal_indices]
pareto_wait_times = [wait_times[i] for i in pareto_optimal_indices]
plt.scatter(pareto_costs, pareto_wait_times, c='red', s=150, alpha=0.9, label='パレート最適解', 
            edgecolors='darkred', linewidth=2, zorder=5)

# 各点に番号をラベル付け
for i, (cost, wait_time) in enumerate(zip(costs, wait_times)):
    plt.annotate(f'{i+1}', (cost, wait_time), xytext=(5, 5), textcoords='offset points', 
                 fontsize=8, alpha=0.7)

# 軸ラベルとタイトル
plt.xlabel('実際のコスト', fontsize=14)
plt.ylabel('実際の待ち時間', fontsize=14)
plt.title('分散パレート探索結果: 実際のコスト vs 実際の待ち時間\n(実行時刻: 2025-08-19 13:19:04)', fontsize=16)

# 凡例
plt.legend(fontsize=12, loc='upper right')

# グリッド
plt.grid(True, alpha=0.3)

# 軸の範囲を調整
plt.xlim(min(costs) * 0.95, max(costs) * 1.05)
plt.ylim(min(wait_times) * 0.95, max(wait_times) * 1.05)

# レイアウトを調整
plt.tight_layout()

# 保存
plt.savefig('pareto_results_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('pareto_results_plot.pdf', bbox_inches='tight')

print("プロットを保存しました:")
print("- pareto_results_plot.png (高解像度PNG)")
print("- pareto_results_plot.pdf (PDF)")

# 表示
plt.show()

# 統計情報も表示
print(f"\n統計情報:")
print(f"コスト範囲: {min(costs):.2f} - {max(costs):.2f}")
print(f"待ち時間範囲: {min(wait_times):.2f} - {max(wait_times):.2f}")
print(f"パレート最適解数: {len(pareto_optimal_indices)}")
print(f"全解数: {len(costs)}") 