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
                    if all(point2[k] <= point1[k] for k in range(len(point1))) and any(point2[k] < point1[k] for k in range(len(point1))):
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
                    if all(point2[k] >= point1[k] for k in range(len(point1))) and any(point2[k] > point1[k] for k in range(len(point1))):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto.append(point1)
        return np.array(pareto)

nsga= np.array([
    [139274.00, 19.97],
    [29951.00, 232.06],
    [32328.00, 227.12],
    [128213.00, 25.28],
    [43455.00, 198.31],
    [100624.00, 39.06],
    [37385.00, 198.78],
    [45726.00, 180.78],
    [95391.00, 71.78],
    [51338.00, 179.38],
    [51399.00, 159.62],
    [98015.00, 47.66],
    [71597.00, 99.38],
    [75408.00, 82.66],
    [74774.00, 92.88],
    [51582.00, 156.75],
    [85735.00, 77.28],
    [95544.00, 56.81],
    [90426.00, 76.09],
    [56753.00, 128.25],
    [68265.00, 111.69],
])



plt.scatter(nsga[:, 0], nsga[:, 1], label='nsga')








plt.xlabel('cost')
plt.ylabel('makespan')
plt.title('Pareto Front')
plt.legend()
plt.show()
plt.savefig('points/32-256.png')