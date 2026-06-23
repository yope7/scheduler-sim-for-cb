"""
イベントベース観測のスケジューリング環境

ビットマップ（ウィンドウ占有状態）を観測から撤廃し、
スケジュール済みイベントの開始時間・終了時間・継続時間・配置位置から学習する。

- 内部のスケジューリングロジックは SchedulingEnvCacheOptimized と同一（C実装使用）
- 観測をイベント形式に変更: (start_time, end_time, duration, use_cloud, start_node, job_height) × N + job_queue
- start_node, job_height によりマップの復元が可能
- イベント履歴は ``scheduling_env_core.SchedulingEventBuffer``（C メモリ）に保持し、get_observation で numpy 経由のコピーを避ける
"""
import os

import numpy as np
from gym import spaces

from .bitmap_c_env import SchedulingEnvCacheOptimized

try:
    from scheduling_env_core import (
        SchedulingEventBuffer as _CSchedulingEventBuffer,
        get_observation_event as _c_get_observation_event,
    )
except ImportError:
    _CSchedulingEventBuffer = None
    _c_get_observation_event = None

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
        self._absolute_schedule_records = []
        if _CSchedulingEventBuffer is not None:
            self._event_buf = _CSchedulingEventBuffer(4096)
            self._scheduled_events = None
        else:
            self._event_buf = None
            self._scheduled_events = []
        self._obs_events_size = N_EVENTS_OBS * EVENT_FEATURES
        # オプトイン: 現ジョブのオンプレ予測待ち(緊急度)を1次元追加（front-loading対策）
        self._obs_urgency = os.environ.get("SCHEDULER_OBS_URGENCY", "0") == "1"
        # オプトイン: 現ジョブの占有量(pt*nodes)のインスタンス内パーセンタイル順位を1次元追加。
        # trace は少数の巨大ジョブ(giant)の配置が高レバレッジ＝報酬地形の崖。順位は「これは
        # top-k の巨大ジョブか」を相対的に符号化し、決定的なジョブをノイズから分離できるようにする。
        self._obs_occupancy = os.environ.get("SCHEDULER_OBS_OCCUPANCY", "0") == "1"
        self._occ_sorted = None  # reset 時に self.jobs から占有量ソート列をキャッシュ
        self._occ_jobs_id = None
        self._obs_total_size = (
            self._obs_events_size
            + JOB_QUEUE_SIZE
            + (1 if self._obs_urgency else 0)
            + (1 if self._obs_occupancy else 0)
        )
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
        self._absolute_schedule_records = []
        if self._event_buf is not None:
            self._event_buf.reset()
        else:
            self._scheduled_events = []
        return self.get_observation()

    def do_schedule(self, action, job, position):
        """スケジュール実行 + イベント記録（start_node, job_height を含む）"""
        wt = super().do_schedule(action, job, position)
        job_width = int(job[0])
        job_height = int(job[1])
        job_id = int(job[4])
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

        node_cols = None
        if isinstance(position, tuple) and len(position) == 3 and position[2] is not None:
            node_cols = position[2]

        self._absolute_schedule_records.append({
            "job_id": job_id,
            "start": start_time,
            "end": end_time,
            "width": job_width,
            "height": job_height,
            "use_cloud": bool(use_cloud),
            "start_node": start_node,
            "node_cols": node_cols,
        })

        if self._event_buf is not None:
            self._event_buf.append(
                float(start_time),
                float(end_time),
                float(job_width),
                float(use_cloud),
                float(start_node),
                float(job_height),
            )
        else:
            self._scheduled_events.append((
                start_time, end_time, job_width, float(use_cloud),
                start_node, job_height
            ))
        return wt

    def _pack_job_queue_f64(self) -> np.ndarray:
        """観測用ジョブキュー 40 要素（不足はゼロ埋め）"""
        j = np.ascontiguousarray(self.job_queue[:5].astype(np.float64).ravel(), dtype=np.float64)
        if j.size >= JOB_QUEUE_SIZE:
            return j[:JOB_QUEUE_SIZE]
        out = np.zeros(JOB_QUEUE_SIZE, dtype=np.float64)
        out[: j.size] = j
        return out

    def _get_observation_python(self) -> np.ndarray:
        """従来の Python 実装（C 拡張が無い場合のフォールバック）"""
        if self._event_buf is not None:
            events = self._event_buf.events_numpy_copy()
        elif self._scheduled_events:
            events = np.asarray(self._scheduled_events, dtype=np.float64).reshape(-1, 6)
        else:
            events = np.empty((0, 6), dtype=np.float64)

        obs_events = np.zeros((N_EVENTS_OBS, EVENT_FEATURES), dtype=np.float32)
        if events.shape[0] > 0:
            window_start = max(0, self.time - self.n_window)
            mask = events[:, 1] >= window_start
            relevant = events[mask]
            order = np.argsort(relevant[:, 0], kind="mergesort")
            relevant = relevant[order]
            n = min(relevant.shape[0], N_EVENTS_OBS)
            if n > 0:
                chunk = relevant[-n:]
                obs_events[:n, 0] = np.clip((chunk[:, 0] - window_start) / self._norm_time, 0, 1)
                obs_events[:n, 1] = np.clip((chunk[:, 1] - window_start) / self._norm_time, 0, 1)
                obs_events[:n, 2] = np.clip(chunk[:, 2] / self._norm_time, 0, 1)
                obs_events[:n, 3] = chunk[:, 3]
                obs_events[:n, 4] = np.clip(chunk[:, 4] / self._norm_nodes, 0, 1)
                obs_events[:n, 5] = np.clip(chunk[:, 5] / self._norm_nodes, 0, 1)

        job_queue_f32 = self.job_queue[:5].astype(np.float32).flatten()
        if len(job_queue_f32) < JOB_QUEUE_SIZE:
            pad = np.zeros(JOB_QUEUE_SIZE - len(job_queue_f32), dtype=np.float32)
            job_queue_f32 = np.concatenate([job_queue_f32, pad])

        observation = np.concatenate([
            obs_events.flatten(),
            job_queue_f32
        ]).astype(np.float32)
        return observation

    def _front_job_urgency(self) -> float:
        """現ジョブのオンプレ予測待ち時間を log 正規化した緊急度 [0,1]。
        event-native の _find_event_allocation（副作用なしのクエリ）で予測。非対応 env は 0。
        これが観測に無いと、方策は閾値判断ができず予算駆動の front-loading に退化する。"""
        try:
            if not hasattr(self, "_find_event_allocation"):
                return 0.0
            jobs = getattr(self, "jobs", None)
            j = int(getattr(self, "index_next_job", 0))
            if jobs is None or j >= len(jobs):
                return 0.0
            raw = jobs[j]
            job = self._to_queue_job(raw) if hasattr(self, "_to_queue_job") else raw
            arr = int(raw[0])
            _, onp = self._find_event_allocation(job, False, arr)
            pw = max(0, int(onp) - arr)
            return float(np.clip(np.log1p(pw) / 16.0, 0.0, 1.0))
        except Exception:
            return 0.0

    def _front_job_leverage(self) -> float:
        """現ジョブの占有量(pt*nodes)のインスタンス内パーセンタイル順位 [0,1]。
        1.0 に近いほど「全ジョブ中で最大級の巨大ジョブ＝高レバレッジ決定」。非対応 env は 0。"""
        try:
            jobs = getattr(self, "jobs", None)
            j = int(getattr(self, "index_next_job", 0))
            if jobs is None or j >= len(jobs):
                return 0.0
            # 占有量ソート列をインスタンス単位でキャッシュ（job配列が変わったら再計算）
            if self._occ_sorted is None or self._occ_jobs_id != id(jobs):
                arr = np.asarray(jobs, dtype=np.float64)
                occ_all = arr[:, 1] * arr[:, 2]
                self._occ_sorted = np.sort(occ_all)
                self._occ_jobs_id = id(jobs)
            raw = jobs[j]
            occ = float(raw[1]) * float(raw[2])
            n = len(self._occ_sorted)
            if n <= 0:
                return 0.0
            rank = float(np.searchsorted(self._occ_sorted, occ, side="right")) / n
            return float(np.clip(rank, 0.0, 1.0))
        except Exception:
            return 0.0

    def get_observation(self):
        """イベントベース観測（必要なら緊急度・レバレッジを順に末尾へ追加）。"""
        base = self._get_observation_base()
        extras = []
        if getattr(self, "_obs_urgency", False):
            extras.append(self._front_job_urgency())
        if getattr(self, "_obs_occupancy", False):
            extras.append(self._front_job_leverage())
        if not extras:
            return base
        return np.concatenate(
            [np.asarray(base, dtype=np.float32), np.asarray(extras, dtype=np.float32)]
        )

    def _get_observation_base(self):
        """
        イベントベース観測を返す（ビットマップ不使用）

        観測構成:
        - イベント部分 (N_EVENTS_OBS * 6): 各 (start_norm, end_norm, duration_norm, use_cloud, start_node_norm, job_height_norm)
        - ジョブキュー部分 (40): job_queue[:5].flatten()
        
        既定: scheduling_env_core.get_observation_event（C）で構築。
        イベント履歴は SchedulingEventBuffer を渡し list→ndarray の変換を省略。
        マップ復元: 各イベントの (start_node, start_node+job_height) × (start_time, end_time) が占有領域
        """
        if _c_get_observation_event is not None and self._event_buf is not None:
            jq = self._pack_job_queue_f64()
            return _c_get_observation_event(
                self._event_buf,
                int(self.time),
                int(self.n_window),
                float(self._norm_time),
                float(self._norm_nodes),
                N_EVENTS_OBS,
                EVENT_FEATURES,
                JOB_QUEUE_SIZE,
                jq,
            )
        if _c_get_observation_event is not None:
            if self._scheduled_events:
                ev = np.asarray(self._scheduled_events, dtype=np.float64).reshape(-1, 6)
            else:
                ev = np.empty((0, 6), dtype=np.float64)
            jq = self._pack_job_queue_f64()
            return _c_get_observation_event(
                ev,
                int(self.time),
                int(self.n_window),
                float(self._norm_time),
                float(self._norm_nodes),
                N_EVENTS_OBS,
                EVENT_FEATURES,
                JOB_QUEUE_SIZE,
                jq,
            )
        return self._get_observation_python()
