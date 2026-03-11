# 分散PCN プロファイリング分析レポート

## 実行環境
- クイックモード: N_ITERATIONS=5, N_ACTORS=12, INITIAL_EPISODES=100
- 観測空間: 256×30 + 1024×30 + 40 = 38,440 要素 (float32 ≈ 153KB/観測)

## 最新プロファイリング結果（2026-03-11）

### cProfile + 詳細タイミング + py-spy による計測

| フェーズ | 時間 | 割合 |
|----------|------|------|
| Phase1 (初期エピソード収集) | 32s | 32% |
| Phase2 (教師あり学習) | 13s | 13% |
| Phase3 (改良経験の実現) | 29s | 29% |
| **合計** | **~100s** | 100% |

### Phase1 の Learner 内訳
| 処理 | 時間 | 備考 |
|------|------|------|
| **get_episodes** | **12s** | 1200エピソードの Ray シリアライゼーション・転送 |
| add_episodes | 0.3s | 軽微 |
| update | 1.7s | GPU学習 |

### Phase3 の 1イテレーションあたり
| 処理 | 時間 |
|------|------|
| Actor実行 (12エピソード) | 2.1s |
| Learner get_episodes | 0.02-0.5s |
| Learner update | 1.3-1.7s |

### Actor 内訳（env.step）
- **1.1-1.5ms/step**（32 steps/エピソード、約40ms/エピソード）
- 過去の 7.5ms/step から大幅改善済み（get_observation 最適化の効果）

### py-spy サンプル数 Top（アプリ関連）
| サンプル数 | 関数 | 備考 |
|------------|------|------|
| 16,109 | _run_episode | Actor のエピソード実行 |
| 13,015 | array2string / _array_str_implementation | **NumPy 配列→文字列変換** |
| 4,284 | update | Learner の GPU 学習 |
| 2,482 | step | env.step |
| 1,524 | get_training_batch | バッチ取得 |
| 1,415 | time_transition | C 拡張 |
| 952 | _make_env | 環境作成 |
| 795 | update_weights_ref | 重み更新 |
| 564 | serialize / _serialize_to_msgpack | Ray シリアライゼーション |

### 新規発見: array2string のボトルネック
- **原因**: Phase1 で `hash(str(obs))` が各エピソード開始時に呼ばれる（`distributed_pcn.py:483`）
- 観測は 38,440 要素の NumPy 配列。`str(obs)` が `array2string` を呼び、大規模配列の文字列化で重い
- **発生箇所**: `episode_seed = ... + hash(str(obs))`（random_actions=True 時のみ、Phase1 の 1200 エピソード）
- **対策**: `obs.tobytes()` や `hash(obs.tobytes())` など、文字列化しないハッシュ方法に変更

---

## ボトルネック特定結果（過去データ含む）

### 1. フェーズ3（メイン学習ループ）の内訳

| 処理 | 時間 | 割合 |
|------|------|------|
| Actor実行 | 2.1s/iter | ~40% |
| Learner実行 | 1.3-1.7s/iter | ~50% |

### 2. Actor内訳（1エピソードあたり）
- **env.step ループ**: ~40ms (32 steps, **1.2ms/step**)
- **_choose_commands**: 0.008-0.26s（初回のみ遅い、2回目以降はキャッシュで高速）
- **get_weights**: エピソード開始前に1回（並列実行のため影響小）

### 3. Learner内訳
- **get_episodes**: Phase1 で **12s**（1200エピソード）、Phase3 で 0.02-0.5s
- **add_episodes**: 0.001-0.025s（軽微）
- **update (GPU学習)**: **1.3-1.7s**（主要ボトルネック）

## ボトルネックの優先度（2026-03-11 更新）

### 最優先: Phase1 get_episodes (12s)
- 1200エピソードの Ray シリアライゼーション・転送
- エピソードデータの圧縮、Object Store の最適化

### 最優先: hash(str(obs)) → array2string (py-spy で 13k サンプル)
- Phase1 の各エピソード開始時に `hash(str(obs))` で 38K 配列を文字列化
- `obs.tobytes()` や `hash(obs.tobytes())` に変更

### 高: Learner update (1.3-1.7s/iter)
- GPU学習が1イテレーションの約40%を占有
- N_UPDATES=5 のため5回のパラメータ更新

### 中: env.step (1.2ms/step)
- 既に最適化済み。大規模時は 4.5ms/step まで悪化

### 低: _choose_commands
- 初回0.26s、2回目以降0.008s（Learner側の_nlargest）

## 推奨最適化（ハイパラメータ変更を除く）

1. **env.step**: 観測の遅延評価・部分更新、C側での観測作成
2. **Ray**: エピソードデータの圧縮、Object Storeの事前確保
3. **並列度**: Actor数を増やしてenv.stepの並列化を強化

※ N_UPDATES, BATCH_SIZE等のハイパラメータ変更は高速化に含めない

## 実装済み最適化（2026-02-18）

### 1. get_observation の高速化 (SchedulingEnvCacheOptimized)
- C連続配列を直接使用し、構造化配列を経由しない
- 事前確保バッファで np.concatenate の中間コピーを削減
- **効果**: env.step が 7.5ms/step → 2.7ms/step（small規模）

### 2. Actor-Learner非同期オーバーラップ
- Learner(i)とActor(i+1)を並列実行し、待ち時間を隠蔽
- デフォルト有効（`DISTRIBUTED_PCN_ASYNC_OVERLAP=0`で無効化して従来の逐次モードに戻す）
- パイプライン: Actor(0)→[Learner(0)∥Actor(1)]→[Learner(1)∥Actor(2)]→...

### 3. プロファイリングツール
- `scripts/profile_distributed_pcn.py --quick`: 詳細タイミング付き短時間実行
- `--cprofile`: cProfileでプロファイリング
- `--py-spy`: サンプリングプロファイリング（子プロセス含む）

## スケーリングベンチマーク結果（N_UPDATES=5 固定）

| スケール | N_JOBS | 観測 | Wall | Phase1 | Phase2 | Phase3 | env.step | Actor/iter | Learner/iter | get_episodes |
|----------|--------|------|------|-------|--------|--------|----------|-------------|--------------|--------------|
| small | 32 | 38,440 | 41s | 6.4s | 3.8s | 16s | 2.7ms | 1.2s | 0.93s | 0.18s |
| medium | 64 | 76,840 | 80s | 9.5s | 5.6s | 44s | 2.8ms | 2.6s | 1.94s | 0.57s |
| large | 128 | 76,840 | 125s | 13s | 5.8s | 83s | 4.5ms | 3.5s | 2.97s | 0.81s |

- スケールが大きくなるほど Phase3（改良経験の実現）が支配的
- 大規模時: get_episodes が 0.8s → Ray シリアライゼーションがボトルネック候補

## GPU利用の確認

### 現状の設計
- **Learner**: `device='cuda'` で初期化。Ray が GPU を認識している場合 `num_gpus=1` で Learner に割り当て
- **Actor**: 意図的に `device='cpu'`（重みの取得・推論のみ、env.step は CPU）
- **PCN**: AMP (autocast + GradScaler) 使用、モデル・データは `.to(device)` で GPU 転送

### 潜在的な問題
1. **Learner の `_get_available_device`**: `torch.cuda.is_available()` をチェックせず常に `'cuda'` を返す。CUDA が無い環境ではクラッシュの可能性
2. **Ray の GPU 認識**: `ray.init()` 時に `num_gpus` を指定しないと、Ray が GPU をクラスターリソースとして認識しない。その場合 Learner は `num_gpus` なしで起動し、PyTorch が直接 GPU を使用（他プロセスと競合の可能性）

### GPU確認方法
```bash
# 学習中に別ターミナルで
watch -n 1 python scripts/check_gpu_usage.py
# または nvidia-smi で GPU 使用率・メモリを確認
```

## プロファイリングの実行方法

```bash
# 詳細タイミング（Actor/Learner 内訳）
DISTRIBUTED_PCN_PROFILE=1 uv run python scripts/profile_distributed_pcn.py --quick

# cProfile（メインプロセスのみ、Ray 待ちが大半）
uv run python scripts/profile_distributed_pcn.py --cprofile -o profile_pcn.prof

# py-spy（全プロセス・C拡張含むサンプリング）
py-spy record --format speedscope --output pcn.speedscope.json --duration 100 --subprocesses -- uv run python -m src.distributed.distributed_pcn
# 結果は https://www.speedscope.app/ で可視化
```

## 必要な最適化（ハイパラメータ変更を除く）

### 優先度 最優先
1. **hash(str(obs)) の削除** (py-spy で 13k サンプル)
   - `distributed_pcn.py:483`: `episode_seed = ... + hash(str(obs))` を `hash(obs.tobytes())` 等に変更
   - Phase1 の 1200 エピソードで 38K 配列の文字列化が発生

2. **Phase1 get_episodes (12s)**
   - 1200エピソードの Ray シリアライゼーション・転送
   - エピソードデータの圧縮、Object Store の最適化

### 優先度 高
3. **Learner update (1.3-1.7s/iter)**
   - GPU学習のボトルネック。N_UPDATES の調整はハイパラ変更のため除外

4. **env.step の観測作成** (large で 4.5ms/step)
   - C 拡張内での観測作成（Python 呼び出し削減）
   - 観測の遅延評価・差分更新

### 優先度 中
5. **Learner の CUDA フォールバック**
   - `_get_available_device` で `torch.cuda.is_available()` をチェックし、無い場合は `'cpu'` を返す

6. ~~**Actor と Learner のオーバーラップ**~~ ✓ 実装済み

### 優先度 低
7. **get_weights の共有**
   - ObjectRef による重み共有は既に実装済み。Actor 数が多い場合の競合確認
