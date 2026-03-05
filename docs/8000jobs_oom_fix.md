# 8000ジョブ環境でのOOM対策

## 問題の原因

### 1. メモリOOM（各Actor約29GB）

**根本原因**: Actorが全50エピソードをメモリに保持してからReplayBufferへ一括送信する設計。

- 1エピソード（8000ステップ）: 観測38,440次元 × 2（obs+next_obs）× 4bytes × 8000 ≈ **2.3GB**
- 50エピソード保持: **約115GB/Actor**
- 16 Actor並列: **約1.8TB** → 503GBノードでOOM

### 2. Rayオブジェクトストア・ディスクOOM（get_all_episodes）

**根本原因**: Learner.learn()がReplayBufferから全エピソードを一括取得 → 200エピソード×2.3GB≈**492GB**のオブジェクトがRayオブジェクトストア（17GB）とスピル先ディスクを圧迫。

- `Shared memory store full, falling back to allocating from filesystem: 492415201688`
- `/tmp` が満杯 → `No space left on device`, `File too large`

### 3. ValueError（buffer_size=0）

フェーズ1でOOMにより全Actorが強制終了 → エピソード0件 → Learner.update()で
`np_random.choice(0, ...)` が `ValueError: a must be a positive integer` を発生。

## 実施した修正

### 1. ストリーミング送信（`distributed_pcn.py`）

- **EPISODES_STREAM_BATCH=5**: N_JOBS≥2000のとき、5エピソードごとにReplayBufferへ送信
- ピークメモリ: 50エピソード → 5エピソード分に削減（約23GB/Actor → 約2.3GB/Actor）

### 2. Learnerの空バッファ対応（`distributed_pcn.py`）

- `Learner.update()`: buffer_size=0のとき学習をスキップし `(0.0, None)` を返す

### 3. フェーズ2のスキップ（`distributed_pcn.py`）

- total_episodes=0のとき、フェーズ2を実行せず明確なエラーメッセージを表示して終了

### 4. Learner.learn()のバッチ取得（`distributed_pcn.py`）

- **get_episodes_batch(max_episodes)**: ReplayBufferから最大5件ずつ取得
- 一括取得（492GB）を回避し、5件≈12GBずつ処理

### 5. 大規模ジョブ時のスケーリング（`distributed_pcn.py`）

| N_JOBS | N_ACTORS | INITIAL_EPISODES | REPLAY_BUFFER_MAX_SIZE |
|--------|----------|------------------|------------------------|
| ≥6000  | min(16,8) | min(50,10)      | 30                     |
| ≥4000  | min(16,12) | min(50,20)     | 30                     |

## プロファイリング

```bash
python scripts/profile_memory_8000jobs.py
```

## 8000ジョブでの実行例

```bash
# デフォルト（ストリーミング＋スケーリングが自動適用）
python -m src.distributed.distributed_pcn

# 環境変数でジョブ数指定
DISTRIBUTED_PCN_JOBS=8000 python -m src.distributed.distributed_pcn
```

## Rayの一時ディレクトリ（ディスク不足時）

`/tmp` が満杯になる場合、十分な空き容量があるディレクトリを指定:

```bash
export RAY_TMPDIR=/path/to/large/disk/ray_tmp  # 500GB以上の空きを推奨
python -m src.distributed.distributed_pcn
```

## 追加のチューニング

メモリがまだ不足する場合:

1. **EPISODES_STREAM_BATCH** を 3 に減らす（`distributed_pcn.py` 66行付近）
2. **LEARN_EPISODES_BATCH_SIZE** を 3 に減らす（オブジェクトストア負荷軽減）
3. **EPISODES_STREAM_THRESHOLD** を 1000 に下げて、より早くストリーミングを有効化
4. **N_ACTORS** を 4〜6 に手動で設定
5. **INITIAL_EPISODES** を 5〜8 に減らす
