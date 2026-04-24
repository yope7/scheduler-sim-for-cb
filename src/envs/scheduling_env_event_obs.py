"""
イベントベース観測のスケジューリング環境

ビットマップ（ウィンドウ占有状態）を観測から撤廃し、
スケジュール済みイベントの開始時間・終了時間・継続時間・配置位置から学習する。

- 内部のスケジューリングロジックは SchedulingEnvCacheOptimized と同一（C実装使用）
- 観測をイベント形式に変更: (start_time, end_time, duration, use_cloud, start_node, job_height) × N + job_queue
- start_node, job_height によりマップの復元が可能
"""
import numpy as np
from gym import spaces

from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized

# 観測に含める最大イベント数（オンプレ+クラウド合計）
N_EVENTS_OBS = 30
# 各イベントの属性数: start_time, end_time, duration, use_cloud, start_node, job_height
EVENT_FEATURES = 6
# ジョブキュー: 5ジョブ × 8属性
JOB_QUEUE_SIZE = 5 * 8


class SchedulingEnvEventObs(SchedulingEnvCacheOptimized):
    """
    イベントベース観測のスケジューリング環境
    
    観測: スケジュール済みイベント(start, end, duration, use_cloud, start_node, job_height) + ジョブキュー
    ビットマップは使用しない。start_node, job_height によりマップ復元が可能。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scheduled_events = []  # (start_time, end_time, duration, use_cloud, start_node, job_height)
        self._obs_events_size = N_EVENTS_OBS * EVENT_FEATURES
        self._obs_total_size = self._obs_events_size + JOB_QUEUE_SIZE
        # 正規化用（0-1にスケール）
        self._norm_time = max(1, self.n_window)
        self._norm_nodes = max(1, max(self.n_on_premise_node, self.n_cloud_node))
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self._obs_total_size,),
            dtype=np.float32
        )

    def reset(self):
        """リセット時にイベント履歴をクリア"""
        obs = super().reset()
        self._scheduled_events = []
        return self.get_observation()

    def do_schedule(self, action, job, position):
        """スケジュール実行 + イベント記録（start_node, job_height を含む）"""
        wt = super().do_schedule(action, job, position)
        job_width = int(job[0])
        job_height = int(job[1])
        use_cloud = action[1]
        start_time = self.time
        end_time = start_time + job_width

        # position から start_node を取得（連続割り当て時は (i, a)、分散割り当て時は (i, a, node_allocation)）
        if isinstance(position, tuple) and len(position) >= 2:
            start_node = int(position[0])
            if len(position) == 3:
                # 分散割り当て: node_allocation は [[col0_nodes], [col1_nodes], ...]、最小ノードを取得
                node_alloc = position[2]
                if node_alloc is not None and len(node_alloc) > 0:
                    flat = [n for col in node_alloc for n in (col if isinstance(col, (list, tuple)) else [col])]
                    if flat:
                        start_node = int(min(flat))
        else:
            start_node = 0

        self._scheduled_events.append((
            start_time, end_time, job_width, float(use_cloud),
            start_node, job_height
        ))
        return wt

    def get_observation(self):
        """
        イベントベース観測を返す（ビットマップ不使用）
        
        観測構成:
        - イベント部分 (N_EVENTS_OBS * 6): 各 (start_norm, end_norm, duration_norm, use_cloud, start_node_norm, job_height_norm)
        - ジョブキュー部分 (40): job_queue[:5].flatten()
        
        マップ復元: 各イベントの (start_node, start_node+job_height) × (start_time, end_time) が占有領域
        """
        # 直近のイベントを取得（現在時刻付近のもの優先）
        obs_events = np.zeros((N_EVENTS_OBS, EVENT_FEATURES), dtype=np.float32)
        if self._scheduled_events:
            # 現在時刻に近いイベントを優先（end_time >= current_time - n_window のもの）
            window_start = max(0, self.time - self.n_window)
            relevant = [
                e for e in self._scheduled_events
                if e[1] >= window_start  # end_time がウィンドウ内
            ]
            # 開始時刻でソート（古い順）
            relevant.sort(key=lambda e: e[0])
            n = min(len(relevant), N_EVENTS_OBS)
            for i in range(n):
                ev = relevant[i]
                s, e, d, uc = ev[0], ev[1], ev[2], ev[3]
                start_node = ev[4] if len(ev) > 4 else 0
                job_height = ev[5] if len(ev) > 5 else 1
                obs_events[i, 0] = np.clip(s / self._norm_time, 0, 1)
                obs_events[i, 1] = np.clip(e / self._norm_time, 0, 1)
                obs_events[i, 2] = np.clip(d / self._norm_time, 0, 1)
                obs_events[i, 3] = uc
                obs_events[i, 4] = np.clip(start_node / self._norm_nodes, 0, 1)
                obs_events[i, 5] = np.clip(job_height / self._norm_nodes, 0, 1)

        # ジョブキュー（既存フォーマット互換・正規化は控えめに）
        job_queue_f32 = self.job_queue[:5].astype(np.float32).flatten()
        # パディング（5ジョブ未満の場合）
        if len(job_queue_f32) < JOB_QUEUE_SIZE:
            pad = np.zeros(JOB_QUEUE_SIZE - len(job_queue_f32), dtype=np.float32)
            job_queue_f32 = np.concatenate([job_queue_f32, pad])

        observation = np.concatenate([
            obs_events.flatten(),
            job_queue_f32
        ]).astype(np.float32)
        return observation
