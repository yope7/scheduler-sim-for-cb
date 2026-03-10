"""
C言語実装に最適化したSchedulingEnv（キャッシュ最適化版 + リングバッファ）

- リングバッファ: time_transition を O(HW)→O(H) に削減（memmove 不要）
- C言語実装を直接使用して高速化
- キャッシュ差分更新: update_cache_time_transition_ringbuffer / update_cache_incremental_ringbuffer
- 構造化配列廃止: C連続配列のみ保持、on_premise_window/cloud_window はプロパティで論理順ビューを提供
- 可視化: finalize_window_history で論理順に並べ替えて取り出し

旧ビットマップ実装のバックアップ: src/envs/backup_bitmap/
"""
import numpy as np
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.envs.scheduling_env import SchedulingEnv

try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position as c_find_allocation_position,
        time_transition as c_time_transition,
        time_transition_ringbuffer as c_time_transition_ringbuffer,
        do_schedule as c_do_schedule,
        do_schedule_ringbuffer as c_do_schedule_ringbuffer,
        get_unique_job_ids as c_get_unique_job_ids,
        calculate_makespan as c_calculate_makespan,
        update_cache_incremental as c_update_cache_incremental,
        update_cache_time_transition as c_update_cache_time_transition,
        update_cache_time_transition_ringbuffer as c_update_cache_time_transition_ringbuffer,
        update_cache_incremental_ringbuffer as c_update_cache_incremental_ringbuffer,
        rebuild_cache_if_needed as c_rebuild_cache_if_needed,
        get_observation as c_get_observation,
        get_observation_ringbuffer as c_get_observation_ringbuffer,
    )
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    raise ImportError("C言語実装が利用できません。ビルドしてください。")


class SchedulingEnvCacheOptimized(SchedulingEnv):
    """
    キャッシュ再構築ロジックを最適化したSchedulingEnv
    
    最適化ポイント:
    1. 再構築の頻度制御: 本当にウィンドウが変更された時だけ再構築
    2. キャッシュの再利用: 同じstep内ではキャッシュを再利用
    3. 無効化の最適化: 変更されたウィンドウのみを無効化
    """
    
    def __init__(self, *args, **kwargs):
        """初期化（C言語実装最適化版）"""
        super().__init__(*args, **kwargs)
        
        # C言語実装用のキャッシュ
        self._cache_onpre_c = None
        self._cache_cloud_c = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        
        # ウィンドウの状態をC連続配列として保持（最適化）
        self._onpre_status_c = None
        self._onpre_job_id_c = None
        self._cloud_status_c = None
        self._cloud_job_id_c = None
        
        # 初期化時にC連続配列を作成
        self._init_c_arrays()
        
        # キャッシュ変更フラグ（ウィンドウが実際に変更されたかどうか）
        self._window_changed_onpre = False
        self._window_changed_cloud = False
        
        # get_observation用の事前計算済みサイズ（観測作成の高速化）
        self._obs_onpre_size = self.n_on_premise_node * self.obs_window_size
        self._obs_cloud_size = self.n_cloud_node * self.obs_window_size
        self._obs_job_size = 8 * 5
        self._obs_total_size = self._obs_onpre_size + self._obs_cloud_size + self._obs_job_size
        
        # リングバッファ用: 最古列の物理インデックス（論理列0=物理列head）
        self._head_onpre = 0
        self._head_cloud = 0

    def _get_window_view(self, status_c, job_id_c, head, H, W):
        """C配列から論理順のビューを返す（構造化配列互換）"""
        class _WindowView:
            __slots__ = ("_status", "_job_id", "_head", "_W", "_H")

            def __init__(self, status, job_id, head, H, W):
                self._status = status
                self._job_id = job_id
                self._head = head
                self._W = W
                self._H = H

            def __getitem__(self, key):
                if key == "status":
                    return np.column_stack([
                        self._status[:, (self._head + c) % self._W]
                        for c in range(self._W)
                    ]).astype(np.int32)
                if key == "job_id":
                    return np.column_stack([
                        self._job_id[:, (self._head + c) % self._W]
                        for c in range(self._W)
                    ]).astype(np.int32)
                raise KeyError(key)

        return _WindowView(status_c, job_id_c, head, H, W)

    @property
    def on_premise_window(self):
        """構造化配列互換: 論理順のビュー（status/job_id）"""
        return self._get_window_view(
            self._onpre_status_c, self._onpre_job_id_c,
            self._head_onpre, self.n_on_premise_node, self.n_window
        )

    @on_premise_window.setter
    def on_premise_window(self, _):
        pass  # 無視（C配列がソース）

    @property
    def cloud_window(self):
        """構造化配列互換: 論理順のビュー（status/job_id）"""
        return self._get_window_view(
            self._cloud_status_c, self._cloud_job_id_c,
            self._head_cloud, self.n_cloud_node, self.n_window
        )

    @cloud_window.setter
    def cloud_window(self, _):
        pass  # 無視（C配列がソース）

    def _invalidate_window_cache(self, on_premise=True, cloud=True):
        """ウィンドウキャッシュを無効化（最適化版）"""
        # 変更フラグを設定（実際に変更されたウィンドウのみ）
        if on_premise:
            self._window_changed_onpre = True
        if cloud:
            self._window_changed_cloud = True
        
        # 親クラスの無効化を呼び出す
        super()._invalidate_window_cache(on_premise=on_premise, cloud=cloud)
    
    def _init_c_arrays(self):
        """C連続配列を初期化（構造化配列なし・直接初期化）"""
        # オンプレミスのウィンドウをC連続配列として保持
        if self._onpre_status_c is None or self._onpre_status_c.shape != (self.n_on_premise_node, self.n_window):
            self._onpre_status_c = np.zeros(
                (self.n_on_premise_node, self.n_window), dtype=np.int32
            )
            self._onpre_job_id_c = np.full(
                (self.n_on_premise_node, self.n_window), -1, dtype=np.int32
            )
        else:
            self._onpre_status_c.fill(0)
            self._onpre_job_id_c.fill(-1)
        
        # クラウドのウィンドウをC連続配列として保持
        if self._cloud_status_c is None or self._cloud_status_c.shape != (self.n_cloud_node, self.n_window):
            self._cloud_status_c = np.zeros(
                (self.n_cloud_node, self.n_window), dtype=np.int32
            )
            self._cloud_job_id_c = np.full(
                (self.n_cloud_node, self.n_window), -1, dtype=np.int32
            )
        else:
            self._cloud_status_c.fill(0)
            self._cloud_job_id_c.fill(-1)
    
    def _rebuild_cache_if_needed(self, use_cloud: bool):
        """キャッシュを再構築（リングバッファ版: build_cache_from_ringbuffer を使用）"""
        self._ensure_cache_initialized()
        
        if not use_cloud:
            current_version = getattr(self, '_version_onpre', 0)
            cache_version = getattr(self, '_cache_version_onpre', -1)
            window_changed = getattr(self, '_window_changed_onpre', False)
            
            if (self._cache_onpre_c is not None and 
                cache_version == current_version and 
                not window_changed):
                return self._cache_onpre_c
            
            # リングバッファからキャッシュを構築
            self._cache_onpre_c = WindowCache(
                self._onpre_status_c, self.n_on_premise_node, self.n_window,
                self._head_onpre
            )
            self._cache_version_onpre = current_version
            self._window_changed_onpre = False
            return self._cache_onpre_c
        else:
            current_version = getattr(self, '_version_cloud', 0)
            cache_version = getattr(self, '_cache_version_cloud', -1)
            window_changed = getattr(self, '_window_changed_cloud', False)
            
            if (self._cache_cloud_c is not None and 
                cache_version == current_version and 
                not window_changed):
                return self._cache_cloud_c
            
            # リングバッファからキャッシュを構築
            self._cache_cloud_c = WindowCache(
                self._cloud_status_c, self.n_cloud_node, self.n_window,
                self._head_cloud
            )
            self._cache_version_cloud = current_version
            self._window_changed_cloud = False
            return self._cache_cloud_c
    
    def find_allocation_position(self, action):
        """割り当て位置を探索（最適化版: Cキャッシュを直接使用）"""
        use_cloud = action[1]
        job = self.job_queue[0]

        job_width = int(job[0])
        job_height = int(job[1])
        when_submitted = int(job[-1])
        current_time = self.time

        # job が 0 なら早期リターン
        if job[0] == 0 and job[1] == 0:
            return None, np.inf

        # 使用するウィンドウの選択とキャッシュ取得（最適化版）
        if not use_cloud:
            max_h, max_w = self.n_on_premise_node, self.n_window
            cache_c = self._cache_onpre_c
        else:
            max_h, max_w = self.n_cloud_node, self.n_window
            cache_c = self._cache_cloud_c

        # キャッシュ未構築ならここで再構築してから使う
        if cache_c is None:
            cache_c = self._rebuild_cache_if_needed(use_cloud=use_cloud)

        # ジョブサイズが大きすぎる場合は早期リターン
        if job_width > max_w or job_height > max_h:
            return None, np.inf
        
        # C言語実装で位置を探索
        position, waiting_time = c_find_allocation_position(
            cache_c, job_width, job_height, when_submitted, current_time
        )
        
        return position, waiting_time
    
    def time_transition(self, slide_on_premise=True, slide_cloud=True):
        """時間遷移（リングバッファ版: O(H)のみ、memmove不要）"""
        self.time += 1
        
        if slide_on_premise:
            self._onpre_status_c = np.ascontiguousarray(self._onpre_status_c, dtype=np.int32)
            self._onpre_job_id_c = np.ascontiguousarray(self._onpre_job_id_c, dtype=np.int32)
            self._onpre_status_c.setflags(write=True)
            self._onpre_job_id_c.setflags(write=True)
            self._append_history_onpre(self._onpre_job_id_c[:, self._head_onpre].copy())
        if slide_cloud:
            self._cloud_status_c = np.ascontiguousarray(self._cloud_status_c, dtype=np.int32)
            self._cloud_job_id_c = np.ascontiguousarray(self._cloud_job_id_c, dtype=np.int32)
            self._cloud_status_c.setflags(write=True)
            self._cloud_job_id_c.setflags(write=True)
            self._append_history_cloud(self._cloud_job_id_c[:, self._head_cloud].copy())
        
        if slide_on_premise:
            _, _, self._head_onpre = c_time_transition_ringbuffer(
                self._onpre_status_c, self._onpre_job_id_c,
                self.n_on_premise_node, self.n_window, self._head_onpre
            )
            if self._cache_onpre_c is not None:
                c_update_cache_time_transition_ringbuffer(self._cache_onpre_c)
            else:
                self._window_changed_onpre = True
        if slide_cloud:
            _, _, self._head_cloud = c_time_transition_ringbuffer(
                self._cloud_status_c, self._cloud_job_id_c,
                self.n_cloud_node, self.n_window, self._head_cloud
            )
            if self._cache_cloud_c is not None:
                c_update_cache_time_transition_ringbuffer(self._cache_cloud_c)
            else:
                self._window_changed_cloud = True

        self.append_new_job2job_queue()
    
    def _append_history_onpre(self, col_data):
        """オンプレミス履歴に1列を追加"""
        if self._hist_len_onpre >= self._hist_cap_onpre:
            new_cap = self._hist_cap_onpre * 2
            new_buf = np.empty((self.n_on_premise_node, new_cap), dtype=int)
            new_buf[:, :self._hist_cap_onpre] = self._hist_onpre_buf
            new_buf[:, self._hist_cap_onpre:new_cap] = -1
            self._hist_onpre_buf = new_buf
            self._hist_cap_onpre = new_cap
        self._hist_onpre_buf[:, self._hist_len_onpre] = col_data
        self._hist_len_onpre += 1
    
    def _append_history_cloud(self, col_data):
        """クラウド履歴に1列を追加"""
        if self._hist_len_cloud >= self._hist_cap_cloud:
            new_cap = self._hist_cap_cloud * 2
            new_buf = np.empty((self.n_cloud_node, new_cap), dtype=int)
            new_buf[:, :self._hist_cap_cloud] = self._hist_cloud_buf
            new_buf[:, self._hist_cap_cloud:new_cap] = -1
            self._hist_cloud_buf = new_buf
            self._hist_cap_cloud = new_cap
        self._hist_cloud_buf[:, self._hist_len_cloud] = col_data
        self._hist_len_cloud += 1
    
    def do_schedule(self, action, job, position):
        """ジョブをスケジュール（リングバッファ版）"""
        self.jobs_processed_count += 1
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        when_submitted = int(job[-1])
        use_cloud = action[1]
        
        if isinstance(position, tuple) and len(position) == 2:
            i, a = position
            i_start, i_end = i, i + job_height
            a_start, a_end = a, a + job_width
        elif isinstance(position, tuple) and len(position) == 3:
            i, a, _ = position
            i_start, i_end = 0, self.n_on_premise_node if not use_cloud else self.n_cloud_node
            a_start, a_end = a, a + job_width
        else:
            i_start, i_end = 0, self.n_on_premise_node if not use_cloud else self.n_cloud_node
            a_start, a_end = 0, self.n_window
        
        if not use_cloud:
            c_do_schedule_ringbuffer(
                self._onpre_status_c, self._onpre_job_id_c,
                self.n_on_premise_node, self.n_window,
                job_width, job_height, job_id,
                position, self._head_onpre
            )
            if self._cache_onpre_c is not None:
                c_update_cache_incremental_ringbuffer(
                    self._cache_onpre_c, self._onpre_status_c,
                    i_start, i_end, a_start, a_end, self._head_onpre
                )
            else:
                self._window_changed_onpre = True
        else:
            c_do_schedule_ringbuffer(
                self._cloud_status_c, self._cloud_job_id_c,
                self.n_cloud_node, self.n_window,
                job_width, job_height, job_id,
                position, self._head_cloud
            )
            if self._cache_cloud_c is not None:
                c_update_cache_incremental_ringbuffer(
                    self._cache_cloud_c, self._cloud_status_c,
                    i_start, i_end, a_start, a_end, self._head_cloud
                )
            else:
                self._window_changed_cloud = True
        
        waiting_time = self.time - when_submitted
        self.waiting_times.append(waiting_time)
        
        return waiting_time
    
    def step(self, action_raw):
        """ステップ実行（最適化版: キャッシュ再構築を最小化）"""
        # 初回のみキャッシュを構築（必要に応じて）
        use_cloud_initial = None
        
        while True:
            scheduled = False
            action = self.get_converted_action(action_raw)   
            allocated_job = self.job_queue[0]
            use_cloud = action[1]
            
            # 初回またはキャッシュが無効な場合のみ再構築
            if use_cloud_initial is None or use_cloud != use_cloud_initial:
                # 必要なキャッシュのみ再構築（キャッシュが有効かチェック済み）
                self._rebuild_cache_if_needed(use_cloud=use_cloud)
                use_cloud_initial = use_cloud
            
            # find_allocation_positionを呼び出す（Cキャッシュを直接使用）
            position, wt_real = self.find_allocation_position(action)
            
            if position is None:
                if np.all(allocated_job == 0):
                    self.time_transition(True, True)
                    # time_transitionで差分更新済みなので、再構築不要
                    # ただし、次回アクションで異なるuse_cloudの場合は再構築が必要
                    use_cloud_initial = None
                else:
                    if action[1] == 0:
                        self.time_transition(True, False)
                    else:
                        self.time_transition(False, True)
                    # time_transitionで差分更新済みなので、再構築不要
                    # ただし、次回アクションで異なるuse_cloudの場合は再構築が必要
                    use_cloud_initial = None
                continue
            else:
                job = self.job_queue[0]
                is_valid = True
                time_reward_new = wt_real

                if action[0] == 0:
                    wt_step = self.do_schedule(action, job, position)
                    scheduled = True
                    self.job_queue = np.roll(self.job_queue, -1, axis=0)
                    self.job_queue[-1] = 0
                    self.rear_job_queue -= 1
                    
                    # do_scheduleで差分更新済みなので、次回アクションで異なるuse_cloudの場合のみ再構築が必要
                    # 同じuse_cloudの場合は再構築不要

                    observation = self.get_observation()
                    cost = self.compute_cost(action, allocated_job, is_valid)
                    done = self.check_is_done()

                    rewards = np.array([-time_reward_new, -cost], dtype=np.float64)
                    self.step_count += 1
                    return observation, rewards, scheduled, wt_step, done
    
    def reset(self):
        """環境のリセット（最適化版）"""
        observation = super().reset()
        
        self._head_onpre = 0
        self._head_cloud = 0
        self._window_changed_onpre = False
        self._window_changed_cloud = False
        self._init_c_arrays()
        self._cache_onpre_c = None
        self._cache_cloud_c = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        
        return observation
    
    def get_observation(self):
        """観測取得（リングバッファ版: C側で直接構築、Python側の配列構築を省略）"""
        job_queue_f64 = np.ascontiguousarray(
            self.job_queue[:5].astype(np.float64),
            dtype=np.float64
        )
        return c_get_observation_ringbuffer(
            self._onpre_status_c,
            self._cloud_status_c,
            job_queue_f64,
            self.n_on_premise_node,
            self.n_cloud_node,
            self.n_window,
            self._head_onpre,
            self._head_cloud,
            self.obs_window_size,
        )
    
    def init_window(self):
        """ウィンドウの初期化（構造化配列なし・C配列のみ）"""
        self._init_c_arrays()
    
    def finalize_window_history(self):
        """ウィンドウ全体を履歴に追加（リングバッファ版: 論理順に並べ替えてから連結）"""
        hist_onpre = self._hist_onpre_buf[:, :self._hist_len_onpre]
        hist_cloud = self._hist_cloud_buf[:, :self._hist_len_cloud]
        # リングバッファを論理順（時刻の古い順）に並べ替え
        window_onpre_chrono = np.column_stack([
            self._onpre_job_id_c[:, (self._head_onpre + i) % self.n_window]
            for i in range(self.n_window)
        ])
        window_cloud_chrono = np.column_stack([
            self._cloud_job_id_c[:, (self._head_cloud + i) % self.n_window]
            for i in range(self.n_window)
        ])
        self.on_premise_window_history_full = np.hstack((hist_onpre, window_onpre_chrono))
        self.cloud_window_history_full = np.hstack((hist_cloud, window_cloud_chrono))
        self.on_premise_window_history_full = np.delete(self.on_premise_window_history_full, 0, axis=1)
        self.cloud_window_history_full = np.delete(self.cloud_window_history_full, 0, axis=1)
        cost, _, _ = self.calc_objective_values(calc_makespan=False, calc_avg_waiting_time=False)
        self.cost = cost

