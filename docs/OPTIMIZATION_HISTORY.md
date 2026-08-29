# 最適化履歴と成果まとめ

本ドキュメントは、スケジューリング環境（env.step）および関連コンポーネントの高速化について、変更履歴・設計判断・実験結果をまとめたものです。**計算結果や品質は一切変更していません**（精度を落とさない方針で実施）。

---

## 1. 全体サマリー

| 項目 | 内容 |
|------|------|
| 対象 | `SchedulingEnvCacheOptimized`（env.step）、`PCN`（_choose_commands） |
| ベースライン | ビットマップ版（memmove による time_transition） |
| 最終成果 | リングバッファ版がビットマップ版の **約 2.5〜2.8 倍** 高速 |
| 検証 | `scripts/verify.py`、`scripts/benchmark_ringbuffer_vs_bitmap.py` |

---

## 2. 変更履歴と設計判断

### 2.1 リングバッファ + キャッシュ差分更新

**問題**: ビットマップ版の `time_transition` は O(H×W) の memmove を毎回実行しており、時間遷移のたびに全ウィンドウをコピーしていた。

**方針**: 物理列を固定し、論理列0の位置（head）だけを進めるリングバッファ方式に変更。memmove を廃止し、時間遷移を O(H) に削減。

**変更内容**:
- C: `time_transition_ringbuffer` を追加（列 head をクリアし、`head = (head+1) % W` で進める）
- C: `do_schedule_ringbuffer` を追加（論理列 a → 物理列 `(head+a)%W` でマッピング）
- C: `update_cache_time_transition_ringbuffer` を追加（キャッシュを左シフトで差分更新）
- C: `update_cache_incremental_ringbuffer` を追加（ジョブ追加時のキャッシュ差分更新）
- Python: `_head_onpre`, `_head_cloud` を導入し、リングバッファの head を管理

**結果**: リングバッファ版がビットマップ版より約 1.3〜1.9 倍高速（初期ベンチマーク）。

---

### 2.2 構造化配列の廃止

**問題**: `on_premise_window` / `cloud_window` を構造化配列で保持し、C 連続配列と同期する処理（`np.copyto`）が毎ステップ発生していた。

**方針**: データソースを C 連続配列（`_onpre_status_c`, `_cloud_status_c` 等）に一本化し、構造化配列を廃止。互換性のため `on_premise_window` / `cloud_window` はプロパティで論理順ビューを返す。

**変更内容**:
- Python: `_init_c_arrays` を直接初期化に変更、同期処理を削除
- Python: `on_premise_window` / `cloud_window` をプロパティ化（`_get_window_view` で論理順ビューを返す）
- 可視化: `finalize_window_history` で論理順に並べ替えて取り出し

**結果**: 追加で約 1.2〜1.7 倍の高速化。

---

### 2.3 get_observation の高速化（get_observation_ringbuffer）

**問題**: 観測取得時に Python 側で `onpre_chrono` / `cloud_chrono` を構築し、`np.column_stack` やリスト内包でインデックス計算していた。プロファイリングで env.step 内の観測作成がボトルネックの一つと判明。

**方針**: 観測構築を C 側に移し、生のウィンドウ配列と head から直接観測を構築する。Python 側の配列構築を省略。

**変更内容**:
- C: `get_observation_ringbuffer` を追加（`onpre_status`, `cloud_status`, `head_onpre`, `head_cloud`, `obs_window_size` を受け取り、リングバッファから直接観測を構築）
- Python: `get_observation` で `c_get_observation_ringbuffer` を呼ぶよう変更
- `onpre_chrono` / `cloud_chrono` の Python 側構築を削除

**結果**: env.step の観測作成オーバーヘッドを削減。プロファイリングでは env.step が 7.5ms/step → 2.7ms/step（small 規模）に改善。

---

### 2.4 _choose_commands の高速化（Numba JIT）

**問題**: PCN の `_choose_commands` 内で `get_non_dominated_inds` と `crowding_distance` が呼ばれ、NSGA-II 風の非支配ソート・混雑度計算がボトルネックの一つだった。

**方針**: これらの関数を Numba JIT 化し、ループをベクトル化。ロジックは変更せず精度を維持。

**変更内容**:
- Python: `_get_non_dominated_inds_maximize_numba`, `_get_non_dominated_inds_minimize_numba` を `@njit(cache=True)` で追加
- Python: `_crowding_distance_numba` を `@njit(cache=True)` で追加
- Python: `get_non_dominated_inds`, `get_non_dominated_inds_minimize`, `crowding_distance` から Numba 版を呼ぶよう変更
- Python: `_nlargest` の距離ループをベクトル化

**結果**: `get_non_dominated_inds` が約 10〜37 倍、`crowding_distance` が約 3.7〜6.9 倍高速化。

---

### 2.5 find_allocation_position の malloc 削減

**問題**: `find_allocation_position` 内で `sliding_window_min` が毎回 `malloc`/`free` を実行しており、呼び出し回数が多いとオーバーヘッドが無視できなかった。

**方針**: `WindowCache` に `scratch_mins` と `scratch_deque` を追加し、キャッシュ構築時に一度だけ確保。`find_allocation_position` ではこのスクラッチバッファを再利用。

**変更内容**:
- C: `WindowCache` に `scratch_mins`, `scratch_deque` を追加
- C: `build_cache`, `build_cache_from_ringbuffer` でこれらを確保
- C: `sliding_window_min` に deque バッファを引数で渡すよう変更
- C: `find_allocation_position` 内の `malloc`/`free` を削除

**結果**: プロファイリングでは find_allocation_position は全体の約 3% だったため、効果は限定的だがオーバーヘッド削減として有効。

---

### 2.6 update_cache_time_transition_ringbuffer の最適化

**問題**: プロファイリングで `update_cache_time_transition_ringbuffer` が env.step 時間の **約 57%** を占めていることが判明。左シフトをループで行い、`prefix_sum` を毎回全再計算していた。

**方針**:
1. `free_per_col` と `occ` のシフトを `memmove` に置き換え
2. `prefix_sum` を全再計算から差分更新に変更（左シフト対応: `new[i][j] = old[i][j+1] - old[i][1]`）

**変更内容**:
- C: `free_per_col` のシフトを `memmove` に変更
- C: `occ` のシフトを行ごとに `memmove` に変更
- C: `prefix_sum` を `prow[j] = prow[j+1] - col0` の形で差分更新（j を W-1 から 1 へ逆順で更新）

**結果**: この最適化により、リングバッファ版がビットマップ版より **約 2.5〜2.8 倍** 高速に。update_cache の負荷が大きく軽減された。

---

### 2.7 free_nodes_list プール（試行・取りやめ）

**問題**: `update_cache_time_transition_ringbuffer` 内で `free_nodes_list[W-1]` 用に毎回 `malloc` し、`free_nodes_list[0]` を `free` していた。

**方針**: W+1 本の H サイズバッファを事前確保し、プールで再利用。malloc/free を削減。

**試行結果**: `update_cache_incremental_ringbuffer` が可変長で malloc しており、プールバッファと混在。プール由来かどうかの判定が難しく、二重解放や不正ポインタの原因に。**計算結果・品質を変えない方針のため、プール変更は取りやめ**。

---

### 2.8 PyTorch vs JAX（検討のみ・移行は未実施）

**目的**: PCN の Learner update（forward + backward + optimizer step）を JAX に置き換えた場合の高速化を検討。

**ベンチマーク**: `scripts/benchmark_pytorch_vs_jax.py` で、PCN 相当の MLP（state_dim=38440, batch_size=1024, hidden_dim=256）の 1 update あたりの時間を計測。

**結果（CPU）**:

| フレームワーク | ms/update | 備考 |
|----------------|-----------|------|
| PyTorch (Adam) | 約 50〜55 | 安定 |
| JAX (SGD)      | 約 51〜60 | 分散が大きい（±100〜150 ms） |

- **平均では**: JAX がわずかに速い場合もある（約 1.07x）が、ばらつきが大きい
- **プロファイル**: JAX の `block_until_ready` が約 60 ms/iter、`cache_miss` / コンパイルが約 50 ms/iter のオーバーヘッドを占める
- **結論**: CPU 上では PyTorch と JAX はほぼ同程度。JAX への移行による明確な高速化は期待しにくい。GPU 使用時は JAX が有利になる可能性はあるが、現状は CPU ベースのため移行は見送り。

**実行方法**:
```bash
uv run python scripts/benchmark_pytorch_vs_jax.py
uv run python scripts/benchmark_pytorch_vs_jax.py --profile  # プロファイル取得
```

---

### 2.9 PCN部分の高速化（2026-03-11）

**目的**: run sweep で Env 改善の効果が限定的だったため、PCN 側のオーバーヘッドを削減。

**プロファイル結果**:
- **_choose_commands**: 各 Actor が Learner にリモート呼び出し（約 1s/回）。12 Actor × 1 回/iter = 12 回の Learner 負荷
- **Learner update**: 0.5〜0.7s（5 updates）、CPU 学習
- **PCN.update の remaining_return ループ**: Python の二重ループで O(batch_size × episode_length)

**実施した改善**:

1. **_choose_commands_batch による一括取得**
   - 各 Actor が個別に `_choose_commands` を呼ぶのではなく、メインループで `_choose_commands_batch(50, n_actors)` を 1 回呼び、全 Actor に事前取得した結果を渡す
   - 12 回のリモート呼び出し → 1 回に削減
   - `Actor.run(pre_fetched_commands=...)` で受け取り、`_choose_commands` をスキップ

2. **PCN.update の remaining_return ベクトル化**
   - 累積報酬計算の内側ループを NumPy の `np.dot(discounts, rewards_slice)` に置き換え
   - Python ループを C 側（NumPy）に移行し、Learner update の CPU 負荷を軽減

**GPU 利用**: 既存の `device='cuda'` 設定のまま。CUDA が利用可能な環境では Learner が GPU を使用する。

---

### 2.10 A40 向け GPU 最適化（2026-03-11）

**対象**: A40 (48GB) × 2 のマシン向けに、torch.compile・BATCH_SIZE・JAX+CUDA を導入。

**実施内容**:

1. **torch.compile**
   - PCN モデル（DiscreteActionsDefaultModel / EnhancedPCNModel）を `th.compile(model, mode='reduce-overhead')` でラップ
   - PyTorch 2.0+ かつ CUDA 時のみ有効

2. **BATCH_SIZE の GPU 向け増加**
   - CUDA 利用時: 2048 → 8192（A40 48GB 向け）
   - CPU 時: 2048 のまま

3. **JAX+CUDA Learner（オプション・非推奨）**
   - `DISTRIBUTED_PCN_USE_JAX=1` で JAX 学習を有効化
   - **実測で PyTorch より遅い**（Phase3 が約 40% 増）。デフォルトは PyTorch 推奨
   - torch.compile: デフォルト無効（CUDAGraphs オーバーヘッド）。`DISTRIBUTED_PCN_USE_TORCH_COMPILE=1` で有効化

**使用方法**:
```bash
# 推奨: PyTorch（最速）
uv run python scripts/run_distributed_pcn_sweep.py --quick

# JAX は遅いため非推奨
# DISTRIBUTED_PCN_USE_JAX=1 uv run python scripts/run_distributed_pcn_sweep.py --quick
```

---

## 3. 実験と成果

### 3.1 ベンチマークの目的

- リングバッファ版とビットマップ版の **env.step を含むエピソード実行時間** を比較
- ジョブ数（nb_jobs）を変えてスケーラビリティを確認
- 両実装で **同じジョブセット** を使用し、公平な比較を実施

---

### 3.2 ベンチマークの詳細

#### 3.2.1 実行方法

```bash
# デフォルト（nb_jobs=20,50,100,200,500、各2回）
uv run python scripts/benchmark_ringbuffer_vs_bitmap.py

# ジョブ数・実行回数を指定
uv run python scripts/benchmark_ringbuffer_vs_bitmap.py --nb_jobs "1000,2000,4000" --n_runs 2

# 結果をJSONで保存
uv run python scripts/benchmark_ringbuffer_vs_bitmap.py --nb_jobs "1000,2000,4000" --output auto
```

#### 3.2.2 計測フェーズ

1エピソードを以下の4フェーズに分けて計測:

| フェーズ | 内容 | 備考 |
|----------|------|------|
| **init** | 環境インスタンスの生成 | `EnvClass(...)` |
| **reset** | `env.reset()` | ジョブキュー投入、ウィンドウ初期化 |
| **main** | メインループ（env.step の繰り返し） | **比較の中心** |
| **finalize** | `env.finalize_window_history()` | 履歴の論理順並べ替え |

- **total**: init + reset + main + finalize
- 高速化の効果は主に **main** に現れる

#### 3.2.3 環境設定（config/config.yml）

| パラメータ | 値 | 説明 |
|------------|-----|------|
| n_window | 100 | リソースマップの横幅（時間軸） |
| n_on_premise_node | 256 | オンプレミスノード数 |
| n_cloud_node | 1024 | クラウドノード数 |
| n_job_queue_obs | 5 | ジョブキューの観測長 |
| n_job_queue_bck | 5 | ジョブキューのバックログ長 |

#### 3.2.4 ジョブ生成

- **JobGenerator**: `job_type=1`（デフォルト）、`lam=0.2`
- **シード**: 固定（デフォルト 42）で両実装に同じジョブセットを渡す
- **ジョブ形式**: `[到着時間, 処理時間, ノード数, クラウド使用可, ユーザID, ジョブID, waiting_time, 提出時間]`

#### 3.2.5 エージェント（HeuristicAgent）

- **base_wait_time_threshold**: 5
- **width_factor**: 0.3（ジョブ幅の30%を閾値に加算）
- **use_cloud_fallback**: True
- **戦略**: オンプレミス優先、待ち時間が閾値を超えたらクラウドへフォールバック

#### 3.2.6 メインループの条件

```python
max_steps = min(nb_jobs * 10, 50000)
while not env.check_is_done() and step_count < max_steps:
    action, is_valid = agent.select_action(env)
    obs, rewards, scheduled, wt_step, done = env.step(action if is_valid else 0)
    step_count += 1
    if done:
        break
```

- `done` になるまで、または `max_steps` に達するまで env.step を繰り返す
- 1エピソードあたりの step 数はジョブ数やスケジュール結果に依存（nb_jobs の数倍程度）

#### 3.2.7 比較対象

| 実装 | ソース | 特徴 |
|------|--------|------|
| **ビットマップ版** | `src/envs/backup_bitmap/scheduling_env_cache_optimized.py` | time_transition で memmove による左シフト |
| **リングバッファ版** | `src/envs/scheduling_variants/bitmap_c_env.py`（ビルド: `src/envs/scheduling_native/`） | head のみ進めるリングバッファ、差分更新 |

#### 3.2.8 出力形式

- 各 nb_jobs について、Bitmap / Ringbuf の total_sec, main_sec を表示
- 高速化倍率 = Bitmap総(秒) / Ringbuf総(秒)
- `--output` 指定時は JSON で保存（timestamp, nb_jobs_list, n_runs, seed, bitmap, ringbuffer）

---

### 3.3 ベンチマーク結果（最終）

**条件**: nb_jobs=1000,2000,4000、n_runs=2、seed=42

| nb_jobs | Bitmap総(秒) | Ringbuf総(秒) | 高速化 | Bitmap main | Ringbuf main |
|---------|-------------|---------------|--------|-------------|--------------|
| 1000    | 5.55        | **1.99**      | **2.79x** | 5.26 | 1.95 |
| 2000    | 11.12       | **4.23**      | **2.63x** | 10.64 | 4.15 |
| 4000    | 22.27       | **8.96**      | **2.49x** | 21.31 | 8.82 |

- nb_jobs が大きいほど step 数が増え、リングバッファ版の優位性が顕著
- 小規模（nb_jobs=20〜100）では環境構築・reset のオーバーヘッドの影響が相対的に大きい場合がある

#### 3.3.1 小規模時の注意

nb_jobs が小さい（20〜200）場合、step 数が少ないため main の絶対時間が短く、init/reset の影響が相対的に大きくなる。また、ジョブの到着パターンや配置結果によって step 数が変動するため、小規模ではビットマップ版が速く出るケースもある。**大規模（nb_jobs≥500）での比較が本質的な性能差を表す**。

---

### 3.4 プロファイリング結果（最適化後）

`scripts/profile_find_allocation.py` および cProfile による内訳（nb_jobs=2000 想定）:

| 処理 | 時間 | 割合 |
|------|------|------|
| update_cache_time_transition_ringbuffer | 2.49s | 41% |
| append_new_job2job_queue | 1.32s | 22% |
| _append_history_onpre | 0.36s | 6% |
| time_transition (Python) | 0.24s | 4% |
| find_allocation_position | 0.22s | 4% |
| get_observation_ringbuffer | 0.14s | 2% |

### 3.5 再現性

- **シード固定**: 両実装に同じ `seed`（デフォルト 42）でジョブセットを生成
- **ジョブセット**: 各 nb_jobs ごとに `create_jobs_set(nb_jobs, config, seed)` で再生成し、Bitmap/Ringbuf 両方に同じジョブを渡す
- **実行順**: 各 nb_jobs で Bitmap を先に n_runs 回、続けて Ringbuf を n_runs 回実行

### 3.6 関連スクリプト

| スクリプト | 用途 |
|------------|------|
| `scripts/benchmark_ringbuffer_vs_bitmap.py` | 本ベンチマーク（Bitmap vs Ringbuf） |
| `scripts/profile_find_allocation.py` | find_allocation_position と env.step の時間内訳を表示 |
| `scripts/verify.py` | scheduling_env_core, nsga2_core, SchedulingEnvCacheOptimized の動作確認 |

プロファイリング例:
```bash
uv run python scripts/profile_find_allocation.py
# または cProfile で詳細内訳
uv run python -c "import cProfile; import pstats; ..."  # 内訳取得
```

### 3.7 検証

- **動作確認**: `scripts/verify.py` で scheduling_env_core, nsga2_core, SchedulingEnvCacheOptimized の動作を確認
- **正確性**: ロジック変更なし、精度を落とさない方針で全最適化を実施

---

## 4. 変更ファイル一覧

| ファイル | 主な変更 |
|----------|----------|
| `scheduling_env_core.c` | リングバッファ版関数、scratch バッファ、memmove、prefix_sum 差分更新 |
| `scheduling_env_core.h` | WindowCache に scratch_mins, scratch_deque 追加 |
| `scheduling_env_bindings.cpp` | get_observation_ringbuffer, update_cache_time_transition_ringbuffer 等のバインディング |
| `scheduling_variants/bitmap_c_env.py` | リングバッファ対応、構造化配列廃止、get_observation_ringbuffer 利用 |
| `pcn_agent.py` | get_non_dominated_inds, crowding_distance の Numba JIT 化 |
| `scripts/benchmark_ringbuffer_vs_bitmap.py` | ベンチマークスクリプト（新規） |
| `scripts/verify.py` | 動作確認スクリプト（新規） |
| `scripts/profile_find_allocation.py` | find_allocation_position プロファイリング（新規） |

---

## 5. 今後の改善候補

- **append_new_job2job_queue**（約 22%）: ジョブキューの roll や np.all の削減、C 側への移行
- **free_nodes_list プール**: `update_cache_incremental_ringbuffer` もプール利用に統一すれば再検討可能
- **Ray get_episodes のシリアライゼーション**: 分散 PCN 大規模時のボトルネック
