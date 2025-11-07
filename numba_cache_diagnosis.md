# Numbaキャッシュ問題の診断と対処

## ログ分析結果

### 問題点

2回目の実行でも一部の関数が再コンパイルされています：

1. **`get_unique_job_ids_njit`**: 
   - `index loaded`が2回 = 異なるシグネチャ（shape）で呼ばれている
   - その後も`index saved` = 新しいシグネチャが検出されている

2. **`calculate_makespan_batch_njit`**:
   - 同様に`index loaded`が2回 = 異なるshapeで呼ばれている

## 原因の特定

### 原因1: 配列のshapeが実行ごとに変わる

**問題**: `cloud_window_history_full`と`on_premise_window_history_full`のshapeが実行時間によって異なる

```python
history_matrix = np.asarray(self.cloud_window_history_full, dtype=np.int32)  # shapeが変わる
onpre_matrix = np.asarray(self.on_premise_window_history_full, dtype=np.int32)  # shapeが変わる
```

**影響**:
- 実行1: `history_matrix.shape = (10, 100)` → シグネチャA
- 実行2: `history_matrix.shape = (10, 200)` → シグネチャB（再コンパイル）

### 原因2: 同じ実行内で異なるshapeで呼ばれている

`calc_objective_values`が呼ばれるたびに、`cloud_window_history_full`のサイズが異なる可能性があります。

## これは正常な動作です

実行時間が異なれば履歴の長さも変わるため、これは**避けられない**可能性があります。
ただし、**同じ実行内で同じshapeで呼ばれる場合はキャッシュが効くはず**です。

## 確認方法

1. 同じ実行内で同じshapeで呼ばれているかを確認
2. キャッシュのHIT/MISSログを確認
3. 同じshapeで呼ばれている場合は、`index loaded`が表示されるはず

## 改善案（オプション）

もしshapeが異なることを受け入れたくない場合：

1. **履歴のサイズを固定**: 最大サイズで固定長配列を使用（メモリ消費増）
2. **明示的なシグネチャを指定**: ただし、shapeが異なるとやはり別シグネチャになる
3. **現状維持**: 実行時間が異なればshapeも変わるのは仕方ない

## 推奨

**現状維持を推奨**。実行時間が異なれば履歴の長さも変わるのは仕方ないです。
重要なのは、**同じ実行内で同じshapeで呼ばれる場合はキャッシュが効く**ことです。

