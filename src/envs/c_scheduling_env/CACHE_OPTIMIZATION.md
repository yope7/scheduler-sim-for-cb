# キャッシュ再構築ロジックの最適化

## 問題点

### 現在の実装の問題
1. **再構築の頻度が高すぎる**
   - 1 stepあたり約30回のキャッシュチェック/再構築が発生
   - `time_transition`の後、必ずキャッシュを無効化
   - `find_allocation_position`内でもキャッシュを再構築

2. **全面再構築**
   - 毎回キャッシュを完全に再構築
   - O(HW)の計算量（H: ノード数、W: ウィンドウサイズ）
   - 変更部分だけを更新する差分更新がない

3. **キャッシュの再利用が不十分**
   - 同じstep内でキャッシュを再利用していない
   - 不要な再構築が発生

## 最適化の方針

### 1. 再構築の頻度制御（優先度: 高）

**方針:**
- 本当にウィンドウが変更された時だけ再構築
- 変更フラグを導入して、変更されたウィンドウのみを無効化

**実装:**
```python
class SchedulingEnvCacheOptimized(SchedulingEnvOptimized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 変更フラグを導入
        self._window_changed_onpre = False
        self._window_changed_cloud = False
    
    def _invalidate_window_cache(self, on_premise=True, cloud=True):
        # 変更フラグを設定（実際に変更されたウィンドウのみ）
        if on_premise:
            self._window_changed_onpre = True
        if cloud:
            self._window_changed_cloud = True
    
    def _rebuild_cache_if_needed(self, use_cloud: bool):
        # 本当に必要な時だけ再構築（変更フラグとバージョンチェック）
        if (self._window_changed_onpre or 
            self._cache_onpre_c is None or 
            self._cache_version_onpre != current_version):
            # 再構築
            self._cache_onpre_c = WindowCache(...)
            self._window_changed_onpre = False  # フラグをリセット
```

**効果:**
- 不要な再構築を削減
- キャッシュの再利用を最大化

### 2. キャッシュの再利用（優先度: 高）

**方針:**
- 同じstep内ではキャッシュを再利用
- `find_allocation_position`にキャッシュを必ず渡す

**実装:**
```python
def step(self, action_raw):
    # ループ外で一度だけキャッシュを取得
    cache_onpre = None
    cache_cloud = None
    cache_needs_refresh = True
    
    while True:
        # キャッシュが必要な場合のみ再取得
        if cache_needs_refresh or cache_onpre is None or cache_cloud is None:
            cache_onpre = self._rebuild_cache_if_needed(use_cloud=False)
            cache_cloud = self._rebuild_cache_if_needed(use_cloud=True)
            cache_needs_refresh = False
        
        # find_allocation_positionを呼び出す（キャッシュを必ず渡す）
        position, wt_real = self.find_allocation_position(
            action, cache_onpre=cache_onpre, cache_cloud=cache_cloud
        )
```

**効果:**
- 同じstep内でのキャッシュ再利用
- 不要な再構築を削減

### 3. 無効化の最適化（優先度: 中）

**方針:**
- 変更されたウィンドウのみを無効化
- `time_transition`の後、必ずキャッシュを無効化しない

**実装:**
```python
def time_transition(self, slide_on_premise=True, slide_cloud=True):
    # 時間遷移を実行
    if slide_on_premise:
        c_time_transition(...)
        # 変更フラグを設定（実際に変更されたウィンドウのみ）
        self._window_changed_onpre = True
    
    if slide_cloud:
        c_time_transition(...)
        # 変更フラグを設定（実際に変更されたウィンドウのみ）
        self._window_changed_cloud = True
    
    # キャッシュを無効化（変更されたウィンドウのみ）
    self._invalidate_window_cache(on_premise=slide_on_premise, cloud=slide_cloud)
```

**効果:**
- 不要な無効化を削減
- キャッシュの再利用を最大化

## 期待される効果

### パフォーマンス改善
- **再構築の頻度**: 1 stepあたり30回 → 1-2回（大幅削減）
- **再構築のコスト**: O(HW) → O(HW)（現状維持、ただし頻度削減）
- **全体的なパフォーマンス**: 20-30%改善を期待

### 次のステップ
1. **差分更新の実装**（優先度: 高）
   - 全面再構築から差分更新へ
   - 変更部分だけを更新
   - O(HW) → O(変更ジョブ数)

2. **Prefix Sum等の構造化**（優先度: 中）
   - 2Dマップ＋Prefix Sum等の構造化
   - キャッシュの効率的な更新

3. **プロファイルの改善**（優先度: 低）
   - C内部の詳細を見る
   - sampling profilerの併用

## 使用方法

```python
from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized

# キャッシュ最適化版の環境を作成
env = SchedulingEnvCacheOptimized(
    max_step, n_window, n_on_premise_node, n_cloud_node,
    n_job_queue_obs, n_job_queue_bck,
    weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action,
    jobs_set, None, flag=0
)

# 通常のSchedulingEnvと同じように使用
observation = env.reset()
observation, rewards, scheduled, wt_step, done = env.step(action)
```

## テスト方法

```bash
# キャッシュ最適化版のテスト
python test_large_scale_timing_cache_optimized.py --nb_jobs 1000 --use_heuristic

# プロファイリング
python test_large_scale_timing_cache_optimized.py --nb_jobs 1000 --profile --profile_output cache_optimized.prof
```

