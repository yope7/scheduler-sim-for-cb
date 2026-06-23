import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# データの定義
weights_wait = [0.000, 0.032, 0.065, 0.097, 0.129, 0.161, 0.194, 0.226, 0.258, 0.290, 
                0.323, 0.355, 0.387, 0.419, 0.452, 0.484, 0.516, 0.548, 0.581, 0.613, 
                0.645, 0.677, 0.710, 0.742, 0.774, 0.806, 0.839, 0.871, 0.903, 0.935, 
                0.968, 1.000]

total_cost = [12165.00, 29017.00, 14144.00, 3710.00, 3276.00, 52906.00, 20872.00, 9457.00, 
              14458.00, 9000.00, 12222.00, 20311.00, 19280.00, 8719.00, 16426.00, 44169.00, 
              5863.00, 37680.00, 32174.00, 36823.00, 85401.00, 52604.00, 58786.00, 47091.00, 
              50262.00, 95616.00, 84822.00, 210744.00, 186693.00, 281060.00, 395806.00, 501130.00]

total_wait_time = [1780.59, 1752.78, 1730.63, 1780.49, 1781.86, 1705.48, 1786.57, 1774.61, 
                   1733.45, 1793.22, 1760.42, 1777.06, 1717.79, 1774.55, 1724.15, 1687.87, 
                   1792.56, 1745.22, 1718.95, 1675.28, 1611.39, 1720.38, 1721.02, 1696.45, 
                   1683.72, 1558.29, 1710.69, 1287.48, 1503.80, 1180.25, 1090.34, 879.06]
total_wait_time = [i / 128 for i in total_wait_time]


# フィギュアとサブプロットの作成
fig, ax1 = plt.subplots(figsize=(12, 8))

# 左軸（待ち時間）のプロット
color1 = 'tab:blue'
ax1.set_xlabel('Weight', fontsize=36)
ax1.set_ylabel('Average Job Wait Time', fontsize=36)
line1 = ax1.plot(weights_wait, total_wait_time, color=color1, marker='o', linewidth=2, markersize=6, label='Average Job Wait Time')
ax1.tick_params(axis='y', labelsize=18)


# 右軸（コスト）のプロット
ax2 = ax1.twinx()
#数字にカンマを付与

ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
color2 = 'tab:red'
ax2.set_ylabel('Total Cost', fontsize=24)
line2 = ax2.plot(weights_wait, total_cost, color=color2, marker='s', linewidth=2, markersize=6, label='Total Cost')
ax2.tick_params(axis='y', labelsize=18)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
# グリッドの設定
ax1.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# タイトルの設定
# plt.title('Total Cost and Job Wait Time vs Weight Transition', fontsize=24)

# 凡例の設定
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=18)

# 軸の範囲設定
ax1.set_xlim(-0.02, 1.02)

# レイアウトの調整
plt.tight_layout()

# PDFで保存
plt.savefig('pareto_weights_plot.pdf', dpi=300, bbox_inches='tight', format='pdf')
print("プロットが 'pareto_weights_plot.pdf' として保存されました。")

# プロットの表示
plt.show() 