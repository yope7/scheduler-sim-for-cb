# Numbaキャッシュ問題の診断と修正

## 問題点

`cache=True`を付けても毎回コンパイルが走る問題。

## 診断結果

### 1. `do_schedule_njit`がobject modeになっている

**問題**: 定義されているが使用されていない関数がobject modeを引き起こしている可能性

```python
@njit(cache=True, fastmath=True)
def do_schedule_njit(..., position, current_time):
    if len(position) == 2:  # ❌ positionがタプルでlen()を使っている
        i, a = position  # ❌ タプルのアンパック
    else:
        i, a, node_allocation = position  # ❌ リストを含むタプル
        for col_offset in range(len(node_allocation)):  # ❌ リストのlen()
```

**原因**: 
- `len(position)`はタプル/リストを扱うためobject modeになる
- `node_allocation`がネストしたリストで、Numbaのnopythonモードではサポートされていない

**対処**: 
- この関数は使われていないようなので、削除するか
- 使う場合は、`position`を2つのint引数に分割する

### 2. 型が固定されていない可能性

各関数の引数のdtypeが毎回変わる可能性：

```python
# 問題: dtypeが固定されていない
history_matrix = self.cloud_window_history_full  # dtype不明

# 修正: dtypeを固定
history_matrix = np.asarray(self.cloud_window_history_full, dtype=np.int32)
```

### 3. 明示的なシグネチャが指定されていない

シグネチャを明示的に指定することで、型の一貫性を保証できます。

## 修正案

### 修正1: `do_schedule_njit`を削除（使われていない場合）

この関数は実際には使われていないようなので、削除します。

### 修正2: 明示的なシグネチャを指定

各関数に明示的なシグネチャを指定します。

### 修正3: dtypeを固定

呼び出し側で必ず`np.asarray(..., dtype=...)`で型を固定します。

## 診断方法

```bash
export NUMBA_DEBUG_CACHE=1
python your_script.py
```

ログで以下を確認：
- `HIT`: キャッシュが使われている
- `MISS`: キャッシュが使われていない（理由も表示される）
- `object mode`: object modeになっている（キャッシュ対象外）

