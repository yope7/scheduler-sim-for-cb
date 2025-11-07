# Numbaキャッシュデバッグ分析

## ログ分析結果

### 1回目の実行
```
[cache] index saved to '...time_transition_njit-103.py312.nbi'
[cache] data saved to '...time_transition_njit-103.py312.1.nbc'
[cache] index saved to '...get_unique_job_ids_njit-27.py312.nbi'
[cache] data saved to '...get_unique_job_ids_njit-27.py312.1.nbc'
[cache] index saved to '...calculate_makespan_batch_njit-65.py312.nbi'
[cache] data saved to '...calculate_makespan_batch_njit-65.py312.1.nbc'
```
→ すべて初回コンパイル（正常）

### 2回目の実行
```
[cache] index loaded from '...time_transition_njit-103.py312.nbi'
[cache] index saved to '...time_transition_njit-103.py312.nbi'
[cache] data saved to '...time_transition_njit-103.py312.1.nbc'
[cache] index loaded from '...get_unique_job_ids_njit-27.py312.nbi'
[cache] index loaded from '...get_unique_job_ids_njit-27.py312.nbi'
[cache] index saved to '...get_unique_job_ids_njit-27.py312.nbi'
[cache] data saved to '...get_unique_job_ids_njit-27.py312.1.nbc'
[cache] index loaded from '...calculate_makespan_batch_njit-65.py312.nbi'
[cache] index loaded from '...calculate_makespan_batch_njit-65.py312.nbi'
[cache] index saved to '...calculate_makespan_batch_njit-65.py312.nbi'
[cache] data saved to '...calculate_makespan_batch_njit-65.py312.1.nbc'
```

## 問題点

1. **`get_unique_job_ids_njit`が複数回コンパイルされている**
   - `index loaded`が2回出ている = 異なるシグネチャで呼ばれている
   - その後も`index saved`が出ている = 新しいシグネチャが検出されている

2. **`calculate_makespan_batch_njit`が複数回コンパイルされている**
   - 同様に`index loaded`が2回出ている

## 原因の可能性

### 原因1: 配列のshapeが実行ごとに変わる

**`get_unique_job_ids_njit`**:
- `history_matrix`のshapeが実行ごとに異なる可能性
- `cloud_window_history_full`のサイズが実行時間によって変わる

**`calculate_makespan_batch_njit`**:
- `onpre_matrix`と`cloud_matrix`のshapeが実行ごとに異なる可能性
- 履歴の長さが実行時間によって変わる

### 原因2: dtypeが変わる

- `np.asarray(..., dtype=np.int32)`を使っているが、元の配列のdtypeが異なる場合
- ただし、これは`asarray`で型変換されているので問題ないはず

## 解決策

### 解決策1: 明示的なシグネチャを指定（推奨）

shapeを明示的に指定することで、異なるshapeでも同じシグネチャとして扱える場合があります：

```python
@njit("int32[:](int32[:,:], int32)", cache=True, fastmath=True)
def get_unique_job_ids_njit(history_matrix, max_job_id=10000):
    ...
```

ただし、これは実際にはshapeの違いもシグネチャに含まれるため、完全な解決にはならない可能性があります。

### 解決策2: shapeが変わることを受け入れる

実行時間が異なれば履歴の長さも変わるため、これは避けられない可能性があります。
しかし、同じ実行内で同じshapeで呼ばれる場合はキャッシュが効くはずです。

### 解決策3: ログレベルを上げて詳細を確認

`NUMBA_DEBUG_CACHE=2`やより詳細なログレベルを設定して、MISSの理由を詳しく確認する。

