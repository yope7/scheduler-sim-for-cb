# SchedulingEnv C言語実装

SchedulingEnvのコア部分をC言語で実装し、pybind11でPythonバインディングを提供します。

## ビルド方法

```bash
cd src/envs/c_scheduling_env
pip install -e .
```

デバッグモードでビルドする場合：

```bash
DEBUG=true pip install -e .
```

## 使用方法

```python
import numpy as np
from scheduling_env_core import WindowCache, find_allocation_position, time_transition, do_schedule

# ウィンドウの状態を準備
H, W = 10, 100
window_status = np.zeros((H, W), dtype=np.int32)

# キャッシュを構築
cache = WindowCache(window_status, H, W)

# 割り当て位置を探索
position, waiting_time = find_allocation_position(
    cache, job_width=5, job_height=2, 
    when_submitted=0, current_time=10
)

if position is not None:
    # ジョブをスケジュール
    window_job_id = np.full((H, W), -1, dtype=np.int32)
    do_schedule(
        window_status, window_job_id, H, W,
        job_width=5, job_height=2, job_id=1,
        position=position
    )
```

## 実装された関数

- `WindowCache`: ウィンドウキャッシュの構築と管理
- `find_allocation_position`: 割り当て位置の探索
- `time_transition`: 時間遷移（スライドウィンドウ）
- `do_schedule`: ジョブのスケジュール実行
- `get_unique_job_ids`: ユニークなジョブIDの取得
- `calculate_makespan`: makespanの計算

## パフォーマンス

C言語実装により、以下の処理が高速化されます：

1. **キャッシュ構築**: 2D累積和の計算が高速化
2. **位置探索**: prefix_sumを使った矩形判定が高速化
3. **時間遷移**: スライドウィンドウのシフトが高速化
4. **目的関数計算**: ユニークなジョブIDの取得が高速化

## 注意事項

- NumPy配列はC連続（C_CONTIGUOUS）である必要があります
- メモリ管理は自動的に行われますが、大きな配列を使用する場合は注意してください

