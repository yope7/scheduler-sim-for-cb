# NSGA-II C言語実装

NSGA-IIアルゴリズムのコア部分をC言語で実装し、pybind11でPythonバインディングを提供します。

## ビルド方法

仮想環境を使用する場合：

```bash
cd src/agents/c_nsga2
# 仮想環境をアクティベート
source /path/to/venv/bin/activate  # または適切な仮想環境
pip install -e .
```

デバッグモードでビルドする場合：

```bash
DEBUG=true pip install -e .
```

システムパッケージを使用する場合（推奨されません）：

```bash
pip install -e . --break-system-packages
```

## 実装された関数

- `dominates`: 支配関係の判定
- `non_dominated_sort`: 非支配ソート
- `calculate_crowding_distance`: 混雑度計算
- `tournament_selection`: トーナメント選択
- `single_point_crossover`: 一点交叉
- `two_point_crossover`: 二点交叉
- `uniform_crossover`: 一様交叉
- `mutation`: 突然変異
- `eliminate_duplicates`: 重複個体の排除

## 使用方法

```python
import numpy as np
import nsga2_core

# 目的関数値の行列 (n_pop, n_obj)
objectives = np.array([
    [100.0, 10.0],
    [110.0, 8.0],
    [105.0, 12.0],
    # ...
], dtype=np.float64)

# 非支配ソート
ranks = nsga2_core.non_dominated_sort(objectives)

# 混雑度計算
crowding_distances = nsga2_core.calculate_crowding_distance(objectives)
```

## パフォーマンス

C言語実装により、以下の処理が高速化されます：

1. **非支配ソート**: O(N²)の支配関係計算が高速化
2. **混雑度計算**: ソートと距離計算が高速化
3. **交叉・突然変異**: メモリコピーが最適化

## 注意事項

- NumPy配列はC連続（C_CONTIGUOUS）である必要があります
- 目的関数値は`np.float64`型で提供してください
- 染色体は`np.int32`型で提供してください

