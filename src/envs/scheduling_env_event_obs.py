"""
イベントベース観測のスケジューリング環境

ビットマップ（ウィンドウ占有状態）を観測から撤廃し、
スケジュール済みイベントの開始時間・終了時間・継続時間のみから学習する。

- 内部のスケジューリングロジックは SchedulingEnvCacheOptimized と同一（C実装使用）
- 観測のみをイベント形式に変更: (start_time, end_time, duration, use_cloud) × N + job_queue
"""
import numpy as np
from gym import spaces

from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized

# 観測に含める最大イベント数（オンプレ+クラウド合計）
N_EVENTS_OBS = 30
# 各イベントの属性数: start_time, end_time, duration, use_cloud
EVENT_FEATURES = 4
# ジョブキュー: 5ジョブ × 8属性
JOB_QUEUE_SIZE = 5 * 8


class SchedulingEnvEventObs(SchedulingEnvCacheOptimized):
    """
    イベントベース観測のスケジューリング環境
    
    観測: スケジュール済みイベント(start, end, duration, use_cloud) + ジョブキュー
    ビットマップは使用しない。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scheduled_events = []  # (start_time, end_time, duration, use_cloud)
        self._obs_events_size = N_EVENTS_OBS * EVENT_FEATURES
        self._obs_total_size = self._obs_events_size + JOB_QUEUE_SIZE
        # 正規化用（0-1にスケール）
        self._norm_time = max(1, self.n_window)
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
        """スケジュール実行 + イベント記録"""
        wt = super().do_schedule(action, job, position)
        job_width = int(job[0])
        use_cloud = action[1]
        start_time = self.time
        end_time = start_time + job_width
        self._scheduled_events.append((start_time, end_time, job_width, float(use_cloud)))
        return wt

    def get_observation(self):
        """
        イベントベース観測を返す（ビットマップ不使用）
        
        観測構成:
        - イベント部分 (N_EVENTS_OBS * 4): 各 (start_norm, end_norm, duration_norm, use_cloud)
        - ジョブキュー部分 (40): job_queue[:5].flatten()
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
                s, e, d, uc = relevant[i]
                obs_events[i, 0] = np.clip(s / self._norm_time, 0, 1)
                obs_events[i, 1] = np.clip(e / self._norm_time, 0, 1)
                obs_events[i, 2] = np.clip(d / self._norm_time, 0, 1)
                obs_events[i, 3] = uc

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
