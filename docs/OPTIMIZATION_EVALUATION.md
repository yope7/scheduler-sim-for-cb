# 分散PCN 最適化の評価と今後の余地

## 1. 評価方法

### 1.1 時間計測
```bash
# クイックモード（3イテレーション）でプロファイリング
DISTRIBUTED_PCN_QUICK=1 DISTRIBUTED_PCN_PROFILE=1 python -m src.distributed.distributed_pcn
```

### 1.2 スケールベンチマーク
```bash
python scripts/scale_benchmark.py --scale large
```

### 1.3 評価指標
| 指標 | 説明 | 目標 |
|------|------|------|
| 総経過時間 | フェーズ1〜3の合計 | 短いほど良い |
| env.step | 1ステップあたりの時間 | < 5ms/step (large) |
| Actor/iter | 1イテレーションあたりのActor実行時間 | 並列化で短縮 |
| Learner/iter | 1イテレーションあたりのLearner実行時間 | GPU効率化 |
| _choose_commands | 目標値選択の待ち時間 | < 1s（16sスパイクは異常） |

---

## 2. 現状の評価（2026-02-18時点）

### 実装済み最適化
| 最適化 | 効果 |
|--------|------|
| C側観測作成 | env.step: 7.5ms → 2.7〜4.5ms/step |
| Actor-Learner非同期オーバーラップ | 待ち時間の隠蔽（効果は負荷に依存） |
| add_batch完了待機 | データ整合性の確保 |

### 直近計測（クイックモード）
- **総経過時間**: 約112秒
- **Phase3**: 約76秒（67.9%）
- **env.step**: 3.6〜3.8ms（Phase1）、15〜18ms（Phase3）
- **問題**: `_choose_commands` が 16秒超のスパイクを記録

---

## 3. 更なる最適化の余地

### 優先度 高

#### 3.1 _choose_commands のボトルネック解消
**現象**: ActorがLearnerの`_choose_commands`を呼ぶと、16秒超の待ちが発生することがある。

**原因候補**:
1. **Learnerの直列化**: 4 Actorが同時に`_choose_commands`を呼ぶと、Learnerは1件ずつ処理。他が待機。
2. **experience_replay の肥大化**: `_nlargest`が全バッファ（最大10000エピソード）を走査。`get_non_dominated_inds`がO(n²)。
3. **Learner更新との競合**: オーバーラップ時にLearnerが`learn`中だと、Actorの`_choose_commands`がブロックされる。

**対策案**:
- ~~**キャッシュ**: 同一イテレーション内で目標値をキャッシュし、Actor間で共有~~ ✓ 実装済み
- **_choose_commands_batch**: 1回のLearner呼び出しで複数の異なる目標値を取得し、各Actorに割り当て（多様性を維持）
- **_nlargestの高速化**: サンプリングで候補を絞る、または非支配解計算をNumba/JAXで高速化

#### 3.2 Ray get_episodes のシリアライゼーション
- 大規模時: 0.2〜0.3s/回
- 対策: 観測の量子化（float32→int8）、Arrow形式、バッチ単位の最適化

### 優先度 中

#### 3.3 env.step のさらなる短縮
- 現状: largeで15〜18ms/step（Phase3）。Phase1では3.7msと速い。
- 差分: Phase3では`_choose_commands`後の推論などが含まれる可能性。
- 対策: 観測の遅延評価、差分更新の拡張

#### 3.4 Learner の CUDA フォールバック
- `_get_available_device`で`torch.cuda.is_available()`を確認済み
- CPU環境でのクラッシュ防止として重要

### 優先度 低

#### 3.5 Actor 数のチューニング
- 現状: 4 Actor（クイックモード）
- Learnerがボトルネックの場合、Actor増加は効果が限定的
- `_choose_commands`の競合が解消されれば、Actor増加の効果が期待できる

---

## 4. 推奨アクション

1. **即時**: `_choose_commands`のプロファイリング
   - Learner側で`_nlargest`の実行時間を計測
   - experience_replayのサイズと相関を確認

2. **短期**: 目標値のキャッシュ・共有
   - 1イテレーションあたり1回だけ`_choose_commands`を実行し、全Actorで共有
   - 4回→1回の呼び出しに削減

3. **中期**: `_nlargest`の高速化
   - バッファが大きい場合のサンプリング
   - 非支配解計算のベクトル化・JITコンパイル
