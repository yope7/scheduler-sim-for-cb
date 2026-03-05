# イテレーション20, 50での解悪化の調査レポート

## 概要
`execution_20260219_193020` の Pareto フロント可視化において、イテレーション 20 と 50 で解が明らかに悪化している現象について、`distributed_pcn.py` および関連コードを調査した結果をまとめる。

## 発見した潜在的な問題点

### 1. **_nlargest の距離計算バグ（重要）**

**場所:** `src/agents/pcn_agent.py` 1337-1340行目

```python
dist = np.min(np.linalg.norm(nd_returns - nd_returns[i], axis=1))
combined_score = crowding_distances[i] / (dist + 1e-8)
```

**問題:** `nd_returns - nd_returns[i]` の計算で、自分自身との距離（常に0）が含まれるため、`dist` が常に 0 になる。結果として `combined_score` は実質 `crowding_distances[i] * 1e8` となり、距離ベースの多様性が無効化されている。

**影響:** 評価・探索方向の選択が crowding distance のみに依存し、パレートフロント上の多様な点が選ばれにくくなる可能性がある。

### 2. **np.random と np_random の混在**

**場所:** `src/agents/pcn_agent.py` 955行目

```python
t = np.random.randint(0, episode_length)  # グローバル np.random を使用
```

**問題:** バッチサンプリングは `self.np_random.choice` を使用しているが、エピソード内ステップ選択は `np.random.randint` を使用。`JobGenerator.generate_jobs_set()` が `np.random.seed(0)` を呼ぶため、グローバル乱数状態が他と干渉する可能性がある。

### 3. **評価のサンプリング方向の変動**

評価時、`evaluate()` は `_nlargest(n)` でバッファ内の「最良」エピソードを取得し、その (return, horizon) を目標としてポリシーを実行する。つまり**評価する方向がバッファの内容に依存**している。

- バッファの組成が変わる → 評価方向が変わる → 見かけ上「悪化」する可能性
- ただし、バッファサイズが 10000 に達する前（iter 20: ~1400, iter 50: ~2400）は FIFO による追い出しは発生していない

### 4. **経験再生バッファのヒープ優先度**

`_add_episode` では `(1, (step, hash), transitions)` をヒープに追加。全エピソードで priority=1 のため、実質的には `(step, hash)` で順序が決まる。step は `global_step` で、learn() 呼び出し時に固定される。古いエピソード（小さい step）が先に追い出される設計で、論理的には妥当。

## 推奨修正

### 修正1: _nlargest の距離計算（自分自身を除外）

```python
# 修正前
dist = np.min(np.linalg.norm(nd_returns - nd_returns[i], axis=1))

# 修正後: 自分自身との距離(0)を除外
dists_to_others = np.linalg.norm(nd_returns - nd_returns[i], axis=1)
dists_to_others[i] = np.inf  # 自分自身を除外
dist = np.min(dists_to_others)
```

### 修正2: 一貫した乱数生成器の使用

```python
# 修正前
t = np.random.randint(0, episode_length)

# 修正後
t = self.np_random.integers(0, episode_length)
# または np.random を使う場合: t = int(np.random.randint(0, episode_length))
# ただし np_random に統一することを推奨
```

### 修正3（オプション）: 評価方向の固定化

評価の再現性を高めるため、評価時に使用する (return, horizon) の集合を固定する、または評価専用のサンプリング戦略を導入することを検討できる。

## 結論

- **修正1** はバグであり、多様な解の選択に影響する可能性が高い。実施を推奨する。
- **修正2** は乱数の一貫性のため推奨。
- iter 20, 50 に特化した周期性の原因は特定できなかったが、上記の修正により全体的な安定性が向上する可能性がある。
- カタストロフィック forgetting や学習の不安定性も候補として残る。修正後も悪化が続く場合は、学習率スケジュールやエリート保持の導入を検討する価値がある。
