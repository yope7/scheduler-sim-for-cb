"""
C言語実装に最適化したSchedulingEnv（キャッシュ最適化版）
- C言語実装を直接使用して高速化
- キャッシュ再構築ロジックを最適化
- 再構築の頻度制御: 本当に必要な時だけ再構築
- キャッシュの再利用: 同じstep内ではキャッシュを再利用
- 差分更新を使用してキャッシュを効率的に更新
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
        do_schedule as c_do_schedule,
        get_unique_job_ids as c_get_unique_job_ids,
        calculate_makespan as c_calculate_makespan,
        update_cache_incremental as c_update_cache_incremental,
        update_cache_time_transition as c_update_cache_time_transition,
        rebuild_cache_if_needed as c_rebuild_cache_if_needed,
        get_observation as c_get_observation,
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
        """C連続配列を初期化（最適化）"""
        # オンプレミスのウィンドウをC連続配列として保持
        if self._onpre_status_c is None or self._onpre_status_c.shape != (self.n_on_premise_node, self.n_window):
            self._onpre_status_c = np.zeros(
                (self.n_on_premise_node, self.n_window), dtype=np.int32
            )
            self._onpre_job_id_c = np.full(
                (self.n_on_premise_node, self.n_window), -1, dtype=np.int32
            )
        else:
            # 既存の配列をクリア
            self._onpre_status_c.fill(0)
            self._onpre_job_id_c.fill(-1)
        
        # 構造化配列からC連続配列に同期
        np.copyto(self._onpre_status_c, self.on_premise_window['status'])
        np.copyto(self._onpre_job_id_c, self.on_premise_window['job_id'])
        
        # クラウドのウィンドウをC連続配列として保持
        if self._cloud_status_c is None or self._cloud_status_c.shape != (self.n_cloud_node, self.n_window):
            self._cloud_status_c = np.zeros(
                (self.n_cloud_node, self.n_window), dtype=np.int32
            )
            self._cloud_job_id_c = np.full(
                (self.n_cloud_node, self.n_window), -1, dtype=np.int32
            )
        else:
            # 既存の配列をクリア
            self._cloud_status_c.fill(0)
            self._cloud_job_id_c.fill(-1)
        
        # 構造化配列からC連続配列に同期
        np.copyto(self._cloud_status_c, self.cloud_window['status'])
        np.copyto(self._cloud_job_id_c, self.cloud_window['job_id'])
    
    def _rebuild_cache_if_needed(self, use_cloud: bool):
        """キャッシュを再構築（C実装版: バージョンチェックと差分更新を含む、最適化版）"""
        self._ensure_cache_initialized()
        
        if not use_cloud:
            # オンプレミスのキャッシュ
            current_version = getattr(self, '_version_onpre', 0)
            cache_version = getattr(self, '_cache_version_onpre', -1)
            window_changed = getattr(self, '_window_changed_onpre', False)
            
            # キャッシュが有効で、変更もない場合は再構築不要
            if (self._cache_onpre_c is not None and 
                cache_version == current_version and 
                not window_changed):
                return self._cache_onpre_c
            
            # C実装版の再構築関数を呼び出し
            cache_obj = self._cache_onpre_c if hasattr(self, '_cache_onpre_c') and self._cache_onpre_c is not None else None
            result = c_rebuild_cache_if_needed(
                cache_obj,
                self._onpre_status_c,
                self.n_on_premise_node,
                self.n_window,
                current_version,
                cache_version,
                window_changed
            )
            
            # 結果を取得
            self._cache_onpre_c, new_cache_version, new_window_changed = result
            self._cache_version_onpre = new_cache_version
            self._window_changed_onpre = new_window_changed
            
            return self._cache_onpre_c
        else:
            # クラウドのキャッシュ
            current_version = getattr(self, '_version_cloud', 0)
            cache_version = getattr(self, '_cache_version_cloud', -1)
            window_changed = getattr(self, '_window_changed_cloud', False)
            
            # キャッシュが有効で、変更もない場合は再構築不要
            if (self._cache_cloud_c is not None and 
                cache_version == current_version and 
                not window_changed):
                return self._cache_cloud_c
            
            # C実装版の再構築関数を呼び出し
            cache_obj = self._cache_cloud_c if hasattr(self, '_cache_cloud_c') and self._cache_cloud_c is not None else None
            result = c_rebuild_cache_if_needed(
                cache_obj,
                self._cloud_status_c,
                self.n_cloud_node,
                self.n_window,
                current_version,
                cache_version,
                window_changed
            )
            
            # 結果を取得
            self._cache_cloud_c, new_cache_version, new_window_changed = result
            self._cache_version_cloud = new_cache_version
            self._window_changed_cloud = new_window_changed
            
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
        """時間遷移（最適化版: 差分更新を使用、不要な再構築を削減）"""
        # 時間を1進める
        self.time += 1
        
        # C拡張が要求する書込み可能なC連続int32を保証
        if slide_on_premise:
            self._onpre_status_c = np.ascontiguousarray(self._onpre_status_c, dtype=np.int32)
            self._onpre_job_id_c = np.ascontiguousarray(self._onpre_job_id_c, dtype=np.int32)
            self._onpre_status_c.setflags(write=True)
            self._onpre_job_id_c.setflags(write=True)
        if slide_cloud:
            self._cloud_status_c = np.ascontiguousarray(self._cloud_status_c, dtype=np.int32)
            self._cloud_job_id_c = np.ascontiguousarray(self._cloud_job_id_c, dtype=np.int32)
            self._cloud_status_c.setflags(write=True)
            self._cloud_job_id_c.setflags(write=True)

        # C連続配列を直接使用（データ変換不要）
        if slide_on_premise:
            # C実装で直接変更（in-place）
            c_time_transition(
                self._onpre_status_c, self._onpre_job_id_c,
                self.n_on_premise_node, self.n_window, True
            )
            # update_window_history()で使うため、構造化配列に同期
            self.on_premise_window['status'] = self._onpre_status_c
            self.on_premise_window['job_id'] = self._onpre_job_id_c
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_onpre_c is not None:
                # 差分更新が成功した場合は、変更フラグを立てない（再構築不要）
                c_update_cache_time_transition(
                    self._cache_onpre_c, self._onpre_status_c
                )
                # 差分更新が成功したので、変更フラグは立てない
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_onpre = True
        
        if slide_cloud:
            # C実装で直接変更（in-place）
            c_time_transition(
                self._cloud_status_c, self._cloud_job_id_c,
                self.n_cloud_node, self.n_window, True
            )
            # update_window_history()で使うため、構造化配列に同期
            self.cloud_window['status'] = self._cloud_status_c
            self.cloud_window['job_id'] = self._cloud_job_id_c
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_cloud_c is not None:
                # 差分更新が成功した場合は、変更フラグを立てない（再構築不要）
                c_update_cache_time_transition(
                    self._cache_cloud_c, self._cloud_status_c
                )
                # 差分更新が成功したので、変更フラグは立てない
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_cloud = True

        # update_window_history()で構造化配列を使うため、ここで呼び出す
        self.update_window_history()

        # 新しいジョブをジョブキューに追加
        self.append_new_job2job_queue()
        
        # キャッシュを無効化しない（差分更新で既に更新済み）
        # ただし、キャッシュが存在しない場合のみ変更フラグを設定（上記で処理済み）
    
    def do_schedule(self, action, job, position):
        """ジョブをスケジュール（最適化版: 差分更新を使用、構造化配列への同期を削除）"""
        self.jobs_processed_count += 1
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        when_submitted = int(job[-1])
        use_cloud = action[1]
        
        # C連続配列を直接使用（データ変換不要）
        if not use_cloud:
            # 位置情報を取得（差分更新用）
            if isinstance(position, tuple) and len(position) == 2:
                # 連続割り当て
                i, a = position
                i_start, i_end = i, i + job_height
                a_start, a_end = a, a + job_width
            elif isinstance(position, tuple) and len(position) == 3:
                # 分散割り当て
                i, a, node_allocation = position
                i_start, i_end = 0, self.n_on_premise_node  # 分散割り当ては全行に影響
                a_start, a_end = a, a + job_width
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
            # 構造化配列への同期は削除（get_observation()の直前で同期）
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_onpre_c is not None:
                # 差分更新が成功した場合は、変更フラグを立てない（再構築不要）
                c_update_cache_incremental(
                    self._cache_onpre_c, self._onpre_status_c,
                    i_start, i_end, a_start, a_end
                )
                # 差分更新が成功したので、変更フラグは立てない
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_onpre = True
            
            # キャッシュを無効化しない（差分更新で既に更新済み）
        else:
            # 位置情報を取得（差分更新用）
            if isinstance(position, tuple) and len(position) == 2:
                # 連続割り当て
                i, a = position
                i_start, i_end = i, i + job_height
                a_start, a_end = a, a + job_width
            elif isinstance(position, tuple) and len(position) == 3:
                # 分散割り当て
                i, a, node_allocation = position
                i_start, i_end = 0, self.n_cloud_node  # 分散割り当ては全行に影響
                a_start, a_end = a, a + job_width
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
            # 構造化配列への同期は削除（get_observation()の直前で同期）
            
            # 差分更新を使用（キャッシュが存在する場合）
            if self._cache_cloud_c is not None:
                # 差分更新が成功した場合は、変更フラグを立てない（再構築不要）
                c_update_cache_incremental(
                    self._cache_cloud_c, self._cloud_status_c,
                    i_start, i_end, a_start, a_end
                )
                # 差分更新が成功したので、変更フラグは立てない
            else:
                # キャッシュが存在しない場合は変更フラグを設定
                self._window_changed_cloud = True
            
            # キャッシュを無効化しない（差分更新で既に更新済み）
        
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

                    # get_observation()が呼ばれる直前に構造化配列に同期
                    self.on_premise_window['status'] = self._onpre_status_c
                    self.on_premise_window['job_id'] = self._onpre_job_id_c
                    self.cloud_window['status'] = self._cloud_status_c
                    self.cloud_window['job_id'] = self._cloud_job_id_c
                    
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
        
        # C連続配列を再初期化（reset後にウィンドウが初期化されるため）
        self._init_c_arrays()
        
        # キャッシュをリセット
        self._cache_onpre_c = None
        self._cache_cloud_c = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        
        # 変更フラグをリセット
        self._window_changed_onpre = False
        self._window_changed_cloud = False
        
        return observation
    
    def get_observation(self):
        """観測取得（C側で作成、Pythonのオーバーヘッドを排除）"""
        job_queue_f64 = np.ascontiguousarray(
            self.job_queue[:5].astype(np.float64), dtype=np.float64
        )
        return c_get_observation(
            self._onpre_status_c,
            self._cloud_status_c,
            job_queue_f64,
            self.n_on_premise_node,
            self.n_cloud_node,
            self.n_window,
            self.obs_window_size,
        )
    
    def init_window(self):
        """ウィンドウの初期化（最適化版）"""
        # 親クラスの初期化を実行
        super().init_window()
        
        # C連続配列を再初期化
        self._init_c_arrays()

