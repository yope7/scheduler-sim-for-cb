"""
キャッシュ再構築ロジックを最適化したSchedulingEnv
- 再構築の頻度制御: 本当に必要な時だけ再構築
- キャッシュの再利用: 同じstep内ではキャッシュを再利用
- 差分更新の準備: 将来的に差分更新に対応
"""
import numpy as np
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.envs.c_scheduling_env.scheduling_env_optimized import SchedulingEnvOptimized

try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position as c_find_allocation_position,
        time_transition as c_time_transition,
        do_schedule as c_do_schedule,
        get_unique_job_ids as c_get_unique_job_ids,
        calculate_makespan as c_calculate_makespan,
        update_cache_incremental as c_update_cache_incremental,
        update_cache_time_transition as c_update_cache_time_transition
    )
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    raise ImportError("C言語実装が利用できません。ビルドしてください。")


class SchedulingEnvCacheOptimized(SchedulingEnvOptimized):
    """
    キャッシュ再構築ロジックを最適化したSchedulingEnv
    
    最適化ポイント:
    1. 再構築の頻度制御: 本当にウィンドウが変更された時だけ再構築
    2. キャッシュの再利用: 同じstep内ではキャッシュを再利用
    3. 無効化の最適化: 変更されたウィンドウのみを無効化
    """
    
    def __init__(self, *args, **kwargs):
        """初期化（キャッシュ最適化版）"""
        super().__init__(*args, **kwargs)
        
        # キャッシュ変更フラグ（ウィンドウが実際に変更されたかどうか）
        self._window_changed_onpre = False
        self._window_changed_cloud = False
        
        print("キャッシュ再構築ロジックを最適化した環境を初期化しました")
    
    def _invalidate_window_cache(self, on_premise=True, cloud=True):
        """ウィンドウキャッシュを無効化（最適化版）"""
        # 変更フラグを設定（実際に変更されたウィンドウのみ）
        if on_premise:
            self._window_changed_onpre = True
        if cloud:
            self._window_changed_cloud = True
        
        # 親クラスの無効化を呼び出す
        super()._invalidate_window_cache(on_premise=on_premise, cloud=cloud)
    
    def _rebuild_cache_if_needed(self, use_cloud: bool):
        """キャッシュを再構築（最適化版: 差分更新を使用）"""
        self._ensure_cache_initialized()
        
        if not use_cloud:
            # オンプレミスのキャッシュ
            current_version = getattr(self, '_version_onpre', 0)
            
            # 初回構築またはバージョン不一致の場合は全面再構築
            if (self._cache_onpre_c is None or 
                self._cache_version_onpre != current_version):
                
                # C連続配列を直接使用（データ変換不要）
                self._cache_onpre_c = WindowCache(
                    self._onpre_status_c, self.n_on_premise_node, self.n_window
                )
                self._cache_version_onpre = current_version
                self._window_changed_onpre = False  # フラグをリセット
            elif self._window_changed_onpre:
                # 差分更新を使用（変更フラグが立っている場合）
                c_update_cache_time_transition(
                    self._cache_onpre_c, self._onpre_status_c
                )
                self._window_changed_onpre = False  # フラグをリセット
            
            # 互換性のためPythonキャッシュも返す（最小限のデータ）
            if not hasattr(self, '_cache_onpre') or self._cache_onpre.get('version', -1) != current_version:
                self._cache_onpre = {
                    'version': current_version,
                    'free_per_col': np.array([self.n_on_premise_node] * self.n_window, dtype=np.int32),
                    'prefix_sum': np.zeros((self.n_on_premise_node+1, self.n_window+1), dtype=np.int32),
                    'free_nodes_list': [np.arange(self.n_on_premise_node) for _ in range(self.n_window)],
                    'shape': (self.n_on_premise_node, self.n_window),
                    'occ': np.zeros((self.n_on_premise_node, self.n_window), dtype=np.int32)
                }
            return self._cache_onpre
        else:
            # クラウドのキャッシュ
            current_version = getattr(self, '_version_cloud', 0)
            
            # 初回構築またはバージョン不一致の場合は全面再構築
            if (self._cache_cloud_c is None or 
                self._cache_version_cloud != current_version):
                
                # C連続配列を直接使用（データ変換不要）
                self._cache_cloud_c = WindowCache(
                    self._cloud_status_c, self.n_cloud_node, self.n_window
                )
                self._cache_version_cloud = current_version
                self._window_changed_cloud = False  # フラグをリセット
            elif self._window_changed_cloud:
                # 差分更新を使用（変更フラグが立っている場合）
                c_update_cache_time_transition(
                    self._cache_cloud_c, self._cloud_status_c
                )
                self._window_changed_cloud = False  # フラグをリセット
            
            # 互換性のためPythonキャッシュも返す（最小限のデータ）
            if not hasattr(self, '_cache_cloud') or self._cache_cloud.get('version', -1) != current_version:
                self._cache_cloud = {
                    'version': current_version,
                    'free_per_col': np.array([self.n_cloud_node] * self.n_window, dtype=np.int32),
                    'prefix_sum': np.zeros((self.n_cloud_node+1, self.n_window+1), dtype=np.int32),
                    'free_nodes_list': [np.arange(self.n_cloud_node) for _ in range(self.n_window)],
                    'shape': (self.n_cloud_node, self.n_window),
                    'occ': np.zeros((self.n_cloud_node, self.n_window), dtype=np.int32)
                }
            return self._cache_cloud
    
    def find_allocation_position(self, action, cache_onpre=None, cache_cloud=None):
        """割り当て位置を探索（最適化版: キャッシュを必ず引数で受け取る）"""
        method = action[0]
        use_cloud = action[1]
        job = self.job_queue[0]
        
        if method == 0:
            job = self.job_queue[0]

        job_width = int(job[0])
        job_height = int(job[1])
        when_submitted = int(job[-1])
        current_time = self.time

        # job が 0 なら早期リターン
        if job[0] == 0 and job[1] == 0:
            return None, np.inf

        # 使用するウィンドウの選択とキャッシュ取得（最適化版）
        # キャッシュは必ず引数で受け取る（再構築を避ける）
        if not use_cloud:
            max_h, max_w = self.n_on_premise_node, self.n_window
            
            # キャッシュが渡されていない場合のみ再構築（通常は渡される）
            # ただし、Cキャッシュは常に使用する（Pythonキャッシュは互換性のため）
            current_version = getattr(self, '_version_onpre', 0)
            if self._cache_onpre_c is None or self._cache_version_onpre != current_version:
                self._cache_onpre_c = WindowCache(
                    self._onpre_status_c, max_h, max_w
                )
                self._cache_version_onpre = current_version
            cache_c = self._cache_onpre_c
        else:
            max_h, max_w = self.n_cloud_node, self.n_window
            
            # キャッシュが渡されていない場合のみ再構築（通常は渡される）
            # ただし、Cキャッシュは常に使用する（Pythonキャッシュは互換性のため）
            current_version = getattr(self, '_version_cloud', 0)
            if self._cache_cloud_c is None or self._cache_version_cloud != current_version:
                self._cache_cloud_c = WindowCache(
                    self._cloud_status_c, max_h, max_w
                )
                self._cache_version_cloud = current_version
            cache_c = self._cache_cloud_c

        # ジョブサイズが大きすぎる場合は早期リターン
        if job_width > max_w or job_height > max_h:
            return None, np.inf
        
        # C言語実装で位置を探索
        position, waiting_time = c_find_allocation_position(
            cache_c, job_width, job_height, when_submitted, current_time
        )
        
        return position, waiting_time
    
    def time_transition(self, slide_on_premise=True, slide_cloud=True):
        """時間遷移（最適化版: 差分更新を使用）"""
        # 時間を1進める
        self.time += 1
        self.update_window_history()

        # C連続配列を直接使用（データ変換不要）
        if slide_on_premise:
            # C実装で直接変更（in-place）
            c_time_transition(
                self._onpre_status_c, self._onpre_job_id_c,
                self.n_on_premise_node, self.n_window, True
            )
            # 構造化配列に同期
            self.on_premise_window['status'] = self._onpre_status_c
            self.on_premise_window['job_id'] = self._onpre_job_id_c
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_onpre_c is not None:
                c_update_cache_time_transition(
                    self._cache_onpre_c, self._onpre_status_c
                )
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_onpre = True
        
        if slide_cloud:
            # C実装で直接変更（in-place）
            c_time_transition(
                self._cloud_status_c, self._cloud_job_id_c,
                self.n_cloud_node, self.n_window, True
            )
            # 構造化配列に同期
            self.cloud_window['status'] = self._cloud_status_c
            self.cloud_window['job_id'] = self._cloud_job_id_c
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_cloud_c is not None:
                c_update_cache_time_transition(
                    self._cache_cloud_c, self._cloud_status_c
                )
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_cloud = True

        # 新しいジョブをジョブキューに追加
        self.append_new_job2job_queue()
        
        # キャッシュを無効化（変更されたウィンドウのみ）
        self._invalidate_window_cache(on_premise=slide_on_premise, cloud=slide_cloud)
        # キャッシュオブジェクトは保持（次回再構築時に使用）
    
    def do_schedule(self, action, job, position):
        """ジョブをスケジュール（最適化版: 差分更新を使用）"""
        self.jobs_processed_count += 1
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        when_submitted = int(job[-1])
        use_cloud = action[1]
        
        # C連続配列を直接使用（データ変換不要）
        if not use_cloud:
            # 位置情報を取得（差分更新用）
            if isinstance(position, tuple):
                if len(position) == 2:
                    # 連続割り当て
                    i, a = position
                    i_start, i_end = i, i + job_height
                    a_start, a_end = a, a + job_width
                elif len(position) == 3:
                    # 分散割り当て
                    i, a, node_allocation = position
                    i_start, i_end = 0, self.n_on_premise_node  # 分散割り当ては全行に影響
                    a_start, a_end = a, a + job_width
                else:
                    i_start, i_end = 0, self.n_on_premise_node
                    a_start, a_end = 0, self.n_window
            else:
                i_start, i_end = 0, self.n_on_premise_node
                a_start, a_end = 0, self.n_window
            
            # C実装で直接変更（in-place）
            c_do_schedule(
                self._onpre_status_c, self._onpre_job_id_c,
                self.n_on_premise_node, self.n_window,
                job_width, job_height, job_id,
                position
            )
            # 構造化配列に同期
            self.on_premise_window['status'] = self._onpre_status_c
            self.on_premise_window['job_id'] = self._onpre_job_id_c
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_onpre_c is not None:
                c_update_cache_incremental(
                    self._cache_onpre_c, self._onpre_status_c,
                    i_start, i_end, a_start, a_end
                )
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_onpre = True
            
            self._invalidate_window_cache(on_premise=True, cloud=False)
        else:
            # 位置情報を取得（差分更新用）
            if isinstance(position, tuple):
                if len(position) == 2:
                    # 連続割り当て
                    i, a = position
                    i_start, i_end = i, i + job_height
                    a_start, a_end = a, a + job_width
                elif len(position) == 3:
                    # 分散割り当て
                    i, a, node_allocation = position
                    i_start, i_end = 0, self.n_cloud_node  # 分散割り当ては全行に影響
                    a_start, a_end = a, a + job_width
                else:
                    i_start, i_end = 0, self.n_cloud_node
                    a_start, a_end = 0, self.n_window
            else:
                i_start, i_end = 0, self.n_cloud_node
                a_start, a_end = 0, self.n_window
            
            # C実装で直接変更（in-place）
            c_do_schedule(
                self._cloud_status_c, self._cloud_job_id_c,
                self.n_cloud_node, self.n_window,
                job_width, job_height, job_id,
                position
            )
            # 構造化配列に同期
            self.cloud_window['status'] = self._cloud_status_c
            self.cloud_window['job_id'] = self._cloud_job_id_c
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_cloud_c is not None:
                c_update_cache_incremental(
                    self._cache_cloud_c, self._cloud_status_c,
                    i_start, i_end, a_start, a_end
                )
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_cloud = True
            
            self._invalidate_window_cache(on_premise=False, cloud=True)
        
        waiting_time = self.time - when_submitted
        self.waiting_times.append(waiting_time)
        
        return waiting_time
    
    def step(self, action_raw):
        """ステップ実行（最適化版: キャッシュの再利用を最適化）"""
        # ループ外で一度だけキャッシュを取得（ループ内での重複取得を避ける）
        cache_onpre = None
        cache_cloud = None
        cache_needs_refresh = True  # 初回は必ず取得
        
        while True:
            scheduled = False
            valid_action_cache = {}
            time_reward_new = 0
            time = self.time
            allocated_job = self.job_queue[0]
            action = self.get_converted_action(action_raw)   
            wt_step = 0
            is_valid = False
            
            # キャッシュが必要な場合のみ再取得（最適化版: 変更フラグをチェック）
            if cache_needs_refresh or cache_onpre is None or cache_cloud is None:
                cache_onpre = self._rebuild_cache_if_needed(use_cloud=False)
                cache_cloud = self._rebuild_cache_if_needed(use_cloud=True)
                cache_needs_refresh = False
            
            # find_allocation_positionを呼び出す（キャッシュを必ず渡す）
            position, wt_real = self.find_allocation_position(
                action, cache_onpre=cache_onpre, cache_cloud=cache_cloud
            )
            
            if position is None:
                if np.all(allocated_job == 0):
                    job_none = True
                    self.time_transition(True, True)
                    # time_transition後は変更フラグが設定されるため、次回再構築時に使用
                    cache_needs_refresh = True
                else:
                    job_none = False
                    if action[1] == 0:
                        self.time_transition(True, False)
                    else:
                        self.time_transition(False, True)
                    # time_transition後は変更フラグが設定されるため、次回再構築時に使用
                    cache_needs_refresh = True

                var_reward = 0
                var_after = 0
                wt_step = 0
                std_mean_before = 0
                std_mean_after = 0
                std_reward = 0
                continue
            else:
                job_none = False
                job = self.job_queue[0]
                is_valid = True

                if action[1] == 0:
                    if (0,1) in valid_action_cache:
                        position_parallel, wt_parallel = valid_action_cache[(0,1)]
                    else:
                        # 既に取得したキャッシュを再利用
                        position_parallel, wt_parallel = self.find_allocation_position(
                            [0,1], cache_onpre=cache_onpre, cache_cloud=cache_cloud
                        )
                        valid_action_cache[(0,1)] = (position_parallel, wt_parallel)
                if action[1] == 1:
                    if (0,0) in valid_action_cache:
                        position_parallel, wt_parallel = valid_action_cache[(0,0)]
                    else:
                        # 既に取得したキャッシュを再利用
                        position_parallel, wt_parallel = self.find_allocation_position(
                            [0,0], cache_onpre=cache_onpre, cache_cloud=cache_cloud
                        )
                        valid_action_cache[(0,0)] = (position_parallel, wt_parallel)

                time_reward_new = wt_real

                if action[0] == 0:
                    wt_step = self.do_schedule(action, job, position)
                    scheduled = True
                    self.job_queue = np.roll(self.job_queue, -1, axis=0)
                    self.job_queue[-1] = 0
                    self.rear_job_queue -= 1

                    observation = self.get_observation()
                    cost = self.compute_cost(action, allocated_job, is_valid)
                    done = self.check_is_done()

                    rewards = np.array([-time_reward_new, -cost], dtype=np.float64)
                    self.step_count += 1
                    return observation, rewards, scheduled, wt_step, done
    
    def reset(self):
        """環境のリセット（最適化版）"""
        # 親クラスのリセットを実行
        observation = super().reset()
        
        # 変更フラグをリセット
        self._window_changed_onpre = False
        self._window_changed_cloud = False
        
        # C連続配列を再初期化
        self._init_c_arrays()
        
        # キャッシュをリセット
        self._cache_onpre_c = None
        self._cache_cloud_c = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        
        return observation

