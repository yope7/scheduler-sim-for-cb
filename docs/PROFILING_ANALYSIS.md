# 分散PCN プロファイリング分析レポート

## 実行環境
- クイックモード: N_ITERATIONS=3, N_ACTORS=4, INITIAL_EPISODES=10
- 観測空間: 256×30 + 1024×30 + 40 = 38,440 要素 (float32 ≈ 153KB/観測)

## ボトルネック特定結果

### 1. フェーズ3（メイン学習ループ）の内訳

| 処理 | 時間 | 割合 |
|------|------|------|
| Actor実行 | 1.0-1.4s/iter | ~50% |
| Learner実行 | 1.0-1.3s/iter | ~50% |

### 2. Actor内訳（1エピソードあたり）
- **env.step ループ**: ~0.24s (32 steps, **7.5ms/step**)
- **_choose_commands**: 0.008-0.26s（初回のみ遅い、2回目以降はキャッシュで高速）
- **get_weights**: エピソード開始前に1回（並列実行のため影響小）

### 3. Learner内訳
- **get_episodes**: 0.02-0.67s（バッファサイズに依存、40エピソードで0.5s）
- **add_episodes**: 0.001-0.025s（軽微）
- **update (GPU学習)**: **0.67-1.07s**（主要ボトルネック）

## ボトルネックの優先度

### 高: env.step (7.5ms/step)
- 32ジョブ × 32ステップ/エピソード × 16 Actor × 2エピソード = 32,768 steps/iter
- 並列化されているが、1 Actor あたり 64 steps × 7.5ms = 480ms
- **原因**: 観測作成（38K要素の配列）、C拡張の呼び出しオーバーヘッド

### 高: Learner update (0.8s)
- GPU学習が1イテレーションの約40%を占有
- N_UPDATES=5 のため5回のパラメータ更新

### 中: get_episodes (Ray シリアライゼーション)
- 40エピソード × 32遷移 × (観測×2 + 報酬等) ≈ 数百MBのシリアライズ
- フェーズ1で0.5s、フェーズ3では4エピソードで0.05sと小さい

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

## 必要な最適化（ハイパラメータ変更を除く）

### 優先度 高
1. **Ray get_episodes のシリアライゼーション** (large で 0.8s)
   - エピソードデータの圧縮（観測の量子化、重複排除）
   - Object Store の zero-copy 転送（Arrow 等）
   - バッチサイズの最適化（送信単位の見直し）

2. **env.step の観測作成** (large で 4.5ms/step)
   - C 拡張内での観測作成（Python 呼び出し削減）
   - 観測の遅延評価・差分更新

### 優先度 中
3. **Learner の CUDA フォールバック**
   - `_get_available_device` で `torch.cuda.is_available()` をチェックし、無い場合は `'cpu'` を返す

4. ~~**Actor と Learner のオーバーラップ**~~ ✓ 実装済み
   - Learner(i) と Actor(i+1) を並列実行して待ち時間を隠蔽

### 優先度 低
5. **get_weights の共有**
   - ObjectRef による重み共有は既に実装済み。Actor 数が多い場合の競合確認
