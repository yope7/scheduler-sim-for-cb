# クイックスタートガイド

## ビルドとインストール

```bash
# 1. 依存関係をインストール
cd src/envs/c_scheduling_env
pip install pybind11 numpy

# 2. C言語実装をビルドしてインストール
pip install -e .
```

## テスト

```bash
# 単体テストを実行
python test_c_implementation.py
```

## 使用方法

```python
import numpy as np
from scheduling_env_core import WindowCache, find_allocation_position

# ウィンドウの状態を準備
H, W = 10, 100
window_status = np.zeros((H, W), dtype=np.int32)

# キャッシュを構築
cache = WindowCache(window_status, H, W)

# 割り当て位置を探索
position, waiting_time = find_allocation_position(
    cache, job_width=5, job_height=3,
    when_submitted=0, current_time=10
)

if position is not None:
    print(f"位置が見つかりました: {position}, 待ち時間: {waiting_time}")
```

## 既知の問題

- `time_transition`関数が配列を変更しない問題があります（調査中）

## パフォーマンス

- **find_allocation_position**: 0.004ms/回（1000回の平均）
- 既存のPython実装と比較して、2-5倍の高速化が期待されます

