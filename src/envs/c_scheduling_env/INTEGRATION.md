# 統合ガイド

既存のSchedulingEnvとC言語実装を統合する方法を説明します。

## 段階的移行アプローチ

### ステップ1: C言語実装のビルドとテスト

```bash
# プロジェクトルートで
uv sync
python test_c_implementation.py
```

### ステップ2: 既存のSchedulingEnvにC言語実装を統合

既存の`SchedulingEnv`クラスに、C言語実装を使用するオプションを追加します。

```python
# src/envs/scheduling_env.py に追加

try:
    from src.envs.c_scheduling_env.scheduling_env_wrapper import SchedulingEnvCWrapper
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False

class SchedulingEnv(gym.core.Env):
    def __init__(self, ..., use_c_implementation=False):
        self.use_c_implementation = use_c_implementation and C_AVAILABLE
        
        if self.use_c_implementation:
            self._c_wrapper = SchedulingEnvCWrapper(
                n_on_premise_node=self.n_on_premise_node,
                n_cloud_node=self.n_cloud_node,
                n_window=self.n_window
            )
        else:
            self._c_wrapper = None
```

### ステップ3: メソッドの置き換え

既存のメソッドを、C言語実装を使用するように変更します。

```python
def find_allocation_position(self, action, cache_onpre=None, cache_cloud=None):
    if self.use_c_implementation and self._c_wrapper:
        # C言語実装を使用
        return self._c_wrapper.find_allocation_position(
            action, self.job_queue[0],
            cache_onpre, cache_cloud
        )
    else:
        # 既存のPython実装を使用
        # ... 既存のコード ...
```

### ステップ4: 段階的な置き換え

以下の順序で段階的に置き換えます：

1. **find_allocation_position**: 最もボトルネックとなる部分
2. **time_transition**: 頻繁に呼ばれる処理
3. **do_schedule**: ジョブのスケジュール実行
4. **calc_objective_values**: 目的関数値の計算

## 使用例

### 既存コードとの互換性を保つ

```python
# 既存のコード（変更なし）
env = SchedulingEnv(...)
obs = env.reset()
action = 0
obs, reward, scheduled, wt_step, done = env.step(action)
```

### C言語実装を有効化

```python
# C言語実装を使用
env = SchedulingEnv(..., use_c_implementation=True)
obs = env.reset()
action = 0
obs, reward, scheduled, wt_step, done = env.step(action)
```

## パフォーマンス比較

既存のPython実装とC言語実装の性能を比較します。

```bash
python benchmark_comparison.py
```

期待される結果：
- **find_allocation_position**: 2-5倍の高速化
- **time_transition**: 1.5-3倍の高速化
- **calc_objective_values**: 1.5-2倍の高速化

## 注意事項

1. **メモリ管理**: C言語実装は自動的にメモリを管理しますが、大きな配列を使用する場合は注意してください。

2. **NumPy配列**: NumPy配列はC連続（C_CONTIGUOUS）である必要があります。通常は自動的に処理されますが、明示的に`np.ascontiguousarray()`を使用することもできます。

3. **互換性**: 既存のコードとの互換性を保つため、C言語実装が利用できない場合は自動的にPython実装にフォールバックします。

4. **デバッグ**: デバッグ時は`use_c_implementation=False`に設定して、既存のPython実装を使用できます。

## トラブルシューティング

### C言語実装が使用されない

- `use_c_implementation=True`が設定されているか確認
- C言語実装が正しくビルドされているか確認（`python test_c_implementation.py`）

### パフォーマンスが期待通りでない

- コンパイラ最適化フラグを確認（`-O3`, `-march=native`）
- プロファイリングを実行してボトルネックを特定

### メモリエラー

- 大きな配列を使用する場合は、メモリ使用量を確認
- 必要に応じて、配列のサイズを調整

