#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ハイパーボリューム結果のグラフ化
"""

import matplotlib.pyplot as plt
import numpy as np

# データ
x_values = [10, 20, 40, 60, 80, 100]
y_values = [0.734108, 0.742617, 0.752900, 0.751596, 0.749434,  0.753298]
y_values_1000 = [0.760836]
x_values_1000 = [1000]

# グラフの設定
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, 'bo-', linewidth=2, markersize=8, label='normalized hypervolume')

# データポイントに値を表示
for i, (x, y) in enumerate(zip(x_values, y_values)):
    plt.annotate(f'{y:.6f}', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

# グラフの装飾
plt.xlabel('iteration', fontsize=12)
plt.ylabel('normalized hypervolume', fontsize=12)
plt.title('Relationship between iterations and normalized hypervolume', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# x軸のスケール調整（1000の点が見やすくなるように）
plt.yscale('log')
plt.xlim(5, 120)

# y軸の範囲調整
plt.ylim(0, 0.80)

# グリッド線の調整
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.grid(True, which="minor", ls=":", alpha=0.2)

plt.tight_layout()
plt.savefig('res_0819/hypervolume_vs_iterations.png', dpi=300, bbox_inches='tight')
plt.show()

print("グラフを保存しました: res_0819/hypervolume_vs_iterations.png") 