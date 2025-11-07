# Numbaキャッシュ問題の診断結果と修正案

## 問題の特定

`cache=True`を付けても毎回コンパイルが走る問題について、以下を確認しました。

### 問題のある関数: `do_schedule_njit`

**原因**:
1. **Object modeになっている**: `len(position)`でタプルを扱い、`node_allocation`がネストしたリスト
2. **使われていない**: 実際には`do_schedule`メソッドで直接配列操作をしている
3. **キャッシュ対象外**: object modeの関数はキャッシュされない

**該当コード**:
```python
@njit(cache=True, fastmath=True)
def do_schedule_njit(..., position, current_time):
    if len(position) == 2:  # ❌ object mode
        i, a = position
    else:
        i, a, node_allocation = position  # ❌ リストを含むタプル
        for col_offset in range(len(node_allocation)):  # ❌ リストのlen()
```

### 修正可能な関数

以下の関数は問題ありませんが、型の一貫性を保証するために改善できます：

1. **`get_unique_job_ids_njit`**: ✅ nopythonモードで動作
2. **`calculate_makespan_batch_njit`**: ✅ nopythonモードで動作
3. **`time_transition_njit`**: ✅ nopythonモードで動作（ただしdtype固定が必要）
4. **`first_fit_position_njit`**: ✅ nopythonモードで動作（ただしdtype固定が必要）

## 修正方針

### 1. `do_schedule_njit`を削除または修正

- **推奨**: 使われていないため削除
- **代替**: 必要なら`position`を2つのint引数`(i, a)`に分割した関数を作成

### 2. 型を固定する

呼び出し側で必ず`np.asarray(..., dtype=np.int32)`で型を固定：
- `time_transition_njit`の呼び出し
- `calculate_makespan_batch_njit`の呼び出し
- その他の関数呼び出し

### 3. 明示的なシグネチャを指定（オプション）

より確実にするなら、各関数に明示的なシグネチャを指定：
```python
@njit("(int32[:,:], int32[:,:], bool_, bool_)", cache=True, fastmath=True)
def time_transition_njit(...):
```

## 診断方法

以下のコマンドでキャッシュのHIT/MISSを確認：

```bash
export NUMBA_DEBUG_CACHE=1
python your_script.py
```

ログで確認：
- `HIT`: キャッシュが使われている
- `MISS`: キャッシュが使われていない（理由も表示）
- `object mode`: object modeになっている（キャッシュ対象外）

## 次のステップ

1. `do_schedule_njit`を削除または修正
2. 呼び出し側でdtypeを固定
3. `NUMBA_DEBUG_CACHE=1`でキャッシュが効くことを確認

