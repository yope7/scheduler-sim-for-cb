"""
SchedulingEnvのC言語実装ラッパー
既存のSchedulingEnvと互換性を保ちながら、C言語実装を使用します
"""
import numpy as np
from typing import Optional, Tuple, Union

try:
    from scheduling_env_core import (
        WindowCache,
        find_allocation_position as c_find_allocation_position,
        time_transition as c_time_transition,
        do_schedule as c_do_schedule,
        get_unique_job_ids as c_get_unique_job_ids,
        calculate_makespan as c_calculate_makespan
    )
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
    print("警告: C言語実装が利用できません。Python実装を使用します。")


class SchedulingEnvCWrapper:
    """SchedulingEnvのC言語実装ラッパー"""
    
    def __init__(self, n_on_premise_node: int, n_cloud_node: int, n_window: int):
        """
        初期化
        
        Args:
            n_on_premise_node: オンプレミスノード数
            n_cloud_node: クラウドノード数
            n_window: ウィンドウサイズ
        """
        if not C_AVAILABLE:
            raise ImportError("C言語実装が利用できません。ビルドしてください。")
        
        self.n_on_premise_node = n_on_premise_node
        self.n_cloud_node = n_cloud_node
        self.n_window = n_window
        
        # ウィンドウの状態を初期化
        self.on_premise_window_status = np.zeros(
            (n_on_premise_node, n_window), dtype=np.int32
        )
        self.on_premise_window_job_id = np.full(
            (n_on_premise_node, n_window), -1, dtype=np.int32
        )
        self.cloud_window_status = np.zeros(
            (n_cloud_node, n_window), dtype=np.int32
        )
        self.cloud_window_job_id = np.full(
            (n_cloud_node, n_window), -1, dtype=np.int32
        )
        
        # キャッシュを初期化
        self._cache_onpre = None
        self._cache_cloud = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        
        # 時間とステップカウント
        self.time = 0
        self.step_count = 0
    
    def _rebuild_cache_onpre(self):
        """オンプレミスのキャッシュを再構築"""
        # NumPy配列をC連続に保証
        status = np.ascontiguousarray(
            self.on_premise_window_status, dtype=np.int32
        )
        self._cache_onpre = WindowCache(status, self.n_on_premise_node, self.n_window)
        self._cache_version_onpre += 1
    
    def _rebuild_cache_cloud(self):
        """クラウドのキャッシュを再構築"""
        # NumPy配列をC連続に保証
        status = np.ascontiguousarray(
            self.cloud_window_status, dtype=np.int32
        )
        self._cache_cloud = WindowCache(status, self.n_cloud_node, self.n_window)
        self._cache_version_cloud += 1
    
    def find_allocation_position(
        self,
        action: list,
        job: np.ndarray,
        cache_onpre: Optional[WindowCache] = None,
        cache_cloud: Optional[WindowCache] = None
    ) -> Tuple[Optional[Tuple], float]:
        """
        割り当て位置を探索
        
        Args:
            action: [method, use_cloud]
            job: ジョブ配列 [width, height, ...]
            cache_onpre: オンプレミスのキャッシュ（オプション）
            cache_cloud: クラウドのキャッシュ（オプション）
        
        Returns:
            (position, waiting_time): 位置が見つかった場合は(position, 待ち時間)、
                                     見つからない場合は(None, np.inf)
        """
        use_cloud = action[1]
        job_width = int(job[0])
        job_height = int(job[1])
        when_submitted = int(job[-1])
        
        # ジョブが空の場合
        if job_width == 0 and job_height == 0:
            return None, np.inf
        
        # キャッシュを取得または構築
        if not use_cloud:
            if cache_onpre is None:
                if self._cache_onpre is None:
                    self._rebuild_cache_onpre()
                cache = self._cache_onpre
            else:
                cache = cache_onpre
            max_h, max_w = self.n_on_premise_node, self.n_window
        else:
            if cache_cloud is None:
                if self._cache_cloud is None:
                    self._rebuild_cache_cloud()
                cache = self._cache_cloud
            else:
                cache = cache_cloud
            max_h, max_w = self.n_cloud_node, self.n_window
        
        # ジョブサイズが大きすぎる場合
        if job_width > max_w or job_height > max_h:
            return None, np.inf
        
        # C言語実装で位置を探索
        position, waiting_time = c_find_allocation_position(
            cache,
            job_width,
            job_height,
            when_submitted,
            self.time
        )
        
        if position is None:
            return None, np.inf
        
        return position, waiting_time
    
    def time_transition(self, slide_on_premise: bool = True, slide_cloud: bool = True):
        """
        時間遷移（スライドウィンドウ）
        
        Args:
            slide_on_premise: オンプレミスをスライドするか
            slide_cloud: クラウドをスライドするか
        """
        self.time += 1
        
        # NumPy配列をC連続に保証
        if slide_on_premise:
            status = np.ascontiguousarray(
                self.on_premise_window_status, dtype=np.int32
            )
            job_id = np.ascontiguousarray(
                self.on_premise_window_job_id, dtype=np.int32
            )
            c_time_transition(
                status, job_id,
                self.n_on_premise_node, self.n_window,
                True
            )
            # キャッシュを無効化
            self._cache_onpre = None
        
        if slide_cloud:
            status = np.ascontiguousarray(
                self.cloud_window_status, dtype=np.int32
            )
            job_id = np.ascontiguousarray(
                self.cloud_window_job_id, dtype=np.int32
            )
            c_time_transition(
                status, job_id,
                self.n_cloud_node, self.n_window,
                True
            )
            # キャッシュを無効化
            self._cache_cloud = None
    
    def do_schedule(
        self,
        action: list,
        job: np.ndarray,
        position: Tuple
    ) -> float:
        """
        ジョブのスケジュール実行
        
        Args:
            action: [method, use_cloud]
            job: ジョブ配列 [width, height, ..., job_id, when_submitted]
            position: 位置情報 (i, a) または (i, a, node_allocation)
        
        Returns:
            waiting_time: 待ち時間
        """
        use_cloud = action[1]
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        when_submitted = int(job[-1])
        
        # NumPy配列をC連続に保証
        if not use_cloud:
            status = np.ascontiguousarray(
                self.on_premise_window_status, dtype=np.int32
            )
            job_id_arr = np.ascontiguousarray(
                self.on_premise_window_job_id, dtype=np.int32
            )
            c_do_schedule(
                status, job_id_arr,
                self.n_on_premise_node, self.n_window,
                job_width, job_height, job_id,
                position
            )
            # キャッシュを無効化
            self._cache_onpre = None
        else:
            status = np.ascontiguousarray(
                self.cloud_window_status, dtype=np.int32
            )
            job_id_arr = np.ascontiguousarray(
                self.cloud_window_job_id, dtype=np.int32
            )
            c_do_schedule(
                status, job_id_arr,
                self.n_cloud_node, self.n_window,
                job_width, job_height, job_id,
                position
            )
            # キャッシュを無効化
            self._cache_cloud = None
        
        waiting_time = self.time - when_submitted
        return waiting_time
    
    def get_unique_job_ids(
        self,
        history_matrix: np.ndarray,
        max_job_id: int = 50000
    ) -> np.ndarray:
        """
        ユニークなジョブIDを取得
        
        Args:
            history_matrix: 履歴マトリックス (H x W)
            max_job_id: 最大ジョブID
        
        Returns:
            ユニークなジョブIDの配列
        """
        H, W = history_matrix.shape
        history_contiguous = np.ascontiguousarray(history_matrix, dtype=np.int32)
        return c_get_unique_job_ids(history_contiguous, H, W, max_job_id)
    
    def calculate_makespan(self, window_matrix: np.ndarray) -> int:
        """
        makespanを計算
        
        Args:
            window_matrix: ウィンドウマトリックス (H x W)
        
        Returns:
            makespan（最大列インデックス）
        """
        H, W = window_matrix.shape
        window_contiguous = np.ascontiguousarray(window_matrix, dtype=np.int32)
        return c_calculate_makespan(window_contiguous, H, W)
    
    def reset(self):
        """環境をリセット"""
        self.time = 0
        self.step_count = 0
        
        # ウィンドウを初期化
        self.on_premise_window_status.fill(0)
        self.on_premise_window_job_id.fill(-1)
        self.cloud_window_status.fill(0)
        self.cloud_window_job_id.fill(-1)
        
        # キャッシュをリセット
        self._cache_onpre = None
        self._cache_cloud = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0

