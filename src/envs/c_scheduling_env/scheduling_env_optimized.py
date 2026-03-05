"""
【非推奨】このファイルは非推奨です。
SchedulingEnvCacheOptimizedを使用してください。

このファイルは後方互換性のために残されていますが、
新しいコードではSchedulingEnvCacheOptimizedを使用してください。

C言語実装に最適化したSchedulingEnv
データ変換とメモリコピーを最小限に抑え、C実装を最大限活用
"""
import numpy as np
import gym
from typing import Optional, Tuple, Dict, Any, List
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

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
    raise ImportError("C言語実装が利用できません。ビルドしてください。")

from src.envs.scheduling_env import SchedulingEnv


class SchedulingEnvOptimized(SchedulingEnv):
    """
    【非推奨】このクラスは非推奨です。
    SchedulingEnvCacheOptimizedを使用してください。
    
    C言語実装に最適化したSchedulingEnv
    
    最適化ポイント:
    1. ウィンドウの状態をC連続配列として直接保持
    2. データ変換を最小限に抑える
    3. キャッシュの再利用を最適化
    4. メモリコピーを削減
    """
    
    def __init__(self, *args, **kwargs):
        """初期化（C言語実装に最適化）"""
        super().__init__(*args, **kwargs)
        
        # C言語実装用のキャッシュ
        self._cache_onpre_c = None
        self._cache_cloud_c = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        
        # ウィンドウの状態をC連続配列として保持（最適化）
        # 構造化配列から通常の配列に変換して保持
        self._onpre_status_c = None
        self._onpre_job_id_c = None
        self._cloud_status_c = None
        self._cloud_job_id_c = None
        
        # 初期化時にC連続配列を作成
        self._init_c_arrays()
        
        # print("C言語実装に最適化された環境を初期化しました")
    
    def _init_c_arrays(self):
        """C連続配列を初期化（最適化）"""
        # オンプレミスのウィンドウをC連続配列として保持
        # 既存の配列がある場合は再利用、ない場合は新規作成
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
    
    def _sync_to_c_arrays(self):
        """構造化配列からC連続配列に同期（最適化）"""
        # オンプレミスのウィンドウを同期
        np.copyto(self._onpre_status_c, self.on_premise_window['status'])
        np.copyto(self._onpre_job_id_c, self.on_premise_window['job_id'])
        
        # クラウドのウィンドウを同期
        np.copyto(self._cloud_status_c, self.cloud_window['status'])
        np.copyto(self._cloud_job_id_c, self.cloud_window['job_id'])
    
    def _sync_from_c_arrays(self):
        """C連続配列から構造化配列に同期（最適化）"""
        # オンプレミスのウィンドウを同期
        self.on_premise_window['status'] = self._onpre_status_c
        self.on_premise_window['job_id'] = self._onpre_job_id_c
        
        # クラウドのウィンドウを同期
        self.cloud_window['status'] = self._cloud_status_c
        self.cloud_window['job_id'] = self._cloud_job_id_c
    
    def _rebuild_cache_if_needed(self, use_cloud: bool):
        """C言語実装を使用してキャッシュを構築（最適化版）"""
        self._ensure_cache_initialized()
        
        if not use_cloud:
            # オンプレミスのキャッシュ
            current_version = getattr(self, '_version_onpre', 0)
            if self._cache_onpre_c is None or self._cache_version_onpre != current_version:
                # C連続配列を直接使用（データ変換不要）
                self._cache_onpre_c = WindowCache(
                    self._onpre_status_c, self.n_on_premise_node, self.n_window
                )
                self._cache_version_onpre = current_version
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
            if self._cache_cloud_c is None or self._cache_version_cloud != current_version:
                # C連続配列を直接使用（データ変換不要）
                self._cache_cloud_c = WindowCache(
                    self._cloud_status_c, self.n_cloud_node, self.n_window
                )
                self._cache_version_cloud = current_version
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
        """C言語実装を使用して割り当て位置を探索（最適化版）"""
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

        # 使用するウィンドウの選択とキャッシュ取得（C言語実装を使用）
        if not use_cloud:
            max_h, max_w = self.n_on_premise_node, self.n_window
            # C連続配列を直接使用（データ変換不要）
            current_version = getattr(self, '_version_onpre', 0)
            if self._cache_onpre_c is None or self._cache_version_onpre != current_version:
                self._cache_onpre_c = WindowCache(
                    self._onpre_status_c, max_h, max_w
                )
                self._cache_version_onpre = current_version
            cache_c = self._cache_onpre_c
        else:
            max_h, max_w = self.n_cloud_node, self.n_window
            # C連続配列を直接使用（データ変換不要）
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
        """C言語実装を使用して時間遷移（最適化版）"""
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
        
        if slide_cloud:
            # C実装で直接変更（in-place）
            c_time_transition(
                self._cloud_status_c, self._cloud_job_id_c,
                self.n_cloud_node, self.n_window, True
            )
            # 構造化配列に同期
            self.cloud_window['status'] = self._cloud_status_c
            self.cloud_window['job_id'] = self._cloud_job_id_c

        # 新しいジョブをジョブキューに追加
        self.append_new_job2job_queue()
        
        # キャッシュを無効化
        self._invalidate_window_cache(on_premise=slide_on_premise, cloud=slide_cloud)
        self._cache_onpre_c = None
        self._cache_cloud_c = None
    
    def do_schedule(self, action, job, position):
        """C言語実装を使用してジョブをスケジュール（最適化版）"""
        self.jobs_processed_count += 1
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
        when_submitted = int(job[-1])
        use_cloud = action[1]
        
        # C連続配列を直接使用（データ変換不要）
        if not use_cloud:
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
            self._invalidate_window_cache(on_premise=True, cloud=False)
            self._cache_onpre_c = None
        else:
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
            self._invalidate_window_cache(on_premise=False, cloud=True)
            self._cache_cloud_c = None
        
        waiting_time = self.time - when_submitted
        self.waiting_times.append(waiting_time)
        
        return waiting_time
    
    def reset(self):
        """環境のリセット（最適化版）"""
        # 親クラスのリセットを実行
        observation = super().reset()
        
        # C連続配列を再初期化（reset後にウィンドウが初期化されるため）
        self._init_c_arrays()
        
        # キャッシュをリセット
        self._cache_onpre_c = None
        self._cache_cloud_c = None
        self._cache_version_onpre = 0
        self._cache_version_cloud = 0
        
        return observation
    
    def init_window(self):
        """ウィンドウの初期化（最適化版）"""
        # 親クラスの初期化を実行
        super().init_window()
        
        # C連続配列を再初期化
        self._init_c_arrays()

