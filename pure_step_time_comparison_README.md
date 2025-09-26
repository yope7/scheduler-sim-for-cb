# 純粋Step時間計測機能

この機能は、環境インスタンスの作成・初期化時間を除外し、純粋に `env.step()` の処理時間のみを計測するためのものです。

## 概要

従来の時間計測では、以下の処理時間が含まれていました：
1. 環境インスタンスの作成 (`SchedulingEnv(...)`)
2. 環境のリセット (`env.reset()`)
3. メインループ処理 (`env.step()`)
4. 環境の最終化 (`env.finalize_window_history()`)
5. 目的関数値の計算 (`env.calc_objective_values()`)

新しい純粋step時間計測では、1と2を除外し、3-5の処理時間のみを計測します。

## 使用方法

### NSGA2Agent

```python
from src.agents.nsga2_agent import NSGA2Agent

# エージェントの初期化
nsga2_agent = NSGA2Agent(pop_size=100, num_generations=50)

# 純粋step時間計測による評価
nsga2_agent.evaluate_population_pure_step_time(env, n_jobs=4)
```

### ExhaustiveSearchAgentDistributed

```python
from src.agents.all_agent_distributed import ExhaustiveSearchAgentDistributed

# エージェントの初期化
exhaustive_agent = ExhaustiveSearchAgentDistributed(num_workers=4)

# 純粋step時間計測による全探索
result = exhaustive_agent.run_exhaustive_search_pure_step_time(env, nb_jobs=10)
```

## 出力例

### NSGA2Agent
```
個体 1 の純粋step時間: 0.1234秒
個体 2 の純粋step時間: 0.1345秒
個体 3 の純粋step時間: 0.1189秒
個体 4 の純粋step時間: 0.1423秒
個体 5 の純粋step時間: 0.1298秒
純粋step時間の統計: 平均=0.1299秒, 最小=0.1189秒, 最大=0.1423秒
```

### ExhaustiveSearchAgentDistributed
```
アクションセット 1 の純粋step時間: 0.1234秒
アクションセット 2 の純粋step時間: 0.1345秒
アクションセット 3 の純粋step時間: 0.1189秒
...

=== 純粋step時間の統計 ===
平均step時間: 0.1299秒
最小step時間: 0.1189秒
最大step時間: 0.1423秒
標準偏差: 0.0089秒
```

## 比較方法

両方のエージェントで同じ環境設定を使用して純粋step時間を計測し、以下の点を比較できます：

1. **平均step時間**: 処理の効率性
2. **時間のばらつき**: 処理の安定性
3. **最大・最小時間**: 処理時間の範囲

## 注意事項

- 環境インスタンスの作成・初期化時間は除外されます
- 純粋な計算処理時間のみが計測されます
- 並列処理のオーバーヘッドは含まれません
- 同じ環境設定で比較する必要があります

## 期待される結果

純粋step時間計測により、両エージェントの処理時間が近い値になるはずです。もし大きな差がある場合は、以下の原因が考えられます：

1. 環境の内部実装の違い
2. データ構造やアルゴリズムの違い
3. メモリ管理の違い
4. 最適化レベルの違い 