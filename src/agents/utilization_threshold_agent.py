import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal


@dataclass
class UtilizationThresholdPolicy:
    """
    オンプレミス利用率（稼働率）に基づいてクラウドへオフロードする単純ルール。

    - 利用率 > threshold のときクラウド（action=1）
    - それ以外はオンプレ（action=0）

    注意:
    - job_queueのフォーマットは SchedulingEnv.append_new_job2job_queue() 内の np.roll(-1) 後を想定
      [processing_time, n_required_nodes, can_use_cloud, user_id, job_id, waiting_time, ..., submit_time]
    """

    threshold: float  # 0.0〜1.0
    utilization_mode: Literal["col", "first_k", "window"] = "col"
    utilization_column: int = 0  # utilization_mode="col" のとき参照する列（「現在時刻」を0とみなす）
    utilization_k: int = 10  # utilization_mode="first_k" のとき、左端からK列の平均占有率
    strict_greater: bool = True  # 「何%超えたら」なので > をデフォルト

    # 統計（1エピソード単位で集計・リセット推奨）
    stats: Optional[Dict] = None

    def reset_stats(self) -> None:
        self.stats = {
            "n_decisions": 0,
            "n_onprem": 0,
            "n_cloud": 0,
            "n_forced_onprem": 0,  # can_use_cloud=0等でオンプレに固定された回数
            "util_sum": 0.0,
        }

    def _get_onprem_utilization(self, env) -> float:
        status = env.on_premise_window["status"]
        # statusは0/1を想定（占有セル=1）
        if self.utilization_mode == "window":
            return float(np.mean(status != 0))

        if self.utilization_mode == "first_k":
            k = int(self.utilization_k)
            k = max(1, min(k, status.shape[1]))
            return float(np.mean(status[:, :k] != 0))

        # default: "col"
        col = int(self.utilization_column)
        if col < 0:
            col = status.shape[1] + col
        col = int(np.clip(col, 0, status.shape[1] - 1))
        return float(np.mean(status[:, col] != 0))

    def _job_fits(self, job_width: int, job_height: int, max_h: int, max_w: int) -> bool:
        if job_width <= 0 or job_height <= 0:
            return True
        return (job_width <= max_w) and (job_height <= max_h)

    def select_action(self, env) -> Tuple[int, float]:
        """
        Returns:
            (action_raw, utilization)
            action_raw: 0=オンプレ, 1=クラウド（SchedulingEnv.get_converted_actionと整合）
        """
        if self.stats is None:
            self.reset_stats()

        # キューが空ならオンプレ（env.step側で時間遷移して次のジョブを待つ）
        if getattr(env, "rear_job_queue", 0) <= 0 or np.all(env.job_queue[0] == 0):
            util = self._get_onprem_utilization(env)
            self.stats["util_sum"] += util
            self.stats["n_decisions"] += 1
            self.stats["n_onprem"] += 1
            return 0, util

        job = env.job_queue[0]
        job_width = int(job[0])
        job_height = int(job[1])

        # can_use_cloud: 0ならクラウド不可（JobGeneratorの定義に依存）
        can_use_cloud = int(job[2]) if len(job) > 2 else 1

        util = self._get_onprem_utilization(env)
        offload = (util > self.threshold) if self.strict_greater else (util >= self.threshold)

        # まずルールで希望先を決める
        prefer_cloud = offload and (can_use_cloud == 1)

        fits_onprem = self._job_fits(job_width, job_height, env.n_on_premise_node, env.n_window)
        fits_cloud = (can_use_cloud == 1) and self._job_fits(job_width, job_height, env.n_cloud_node, env.n_window)

        # 希望先が物理的に無理なら反対側へ
        if prefer_cloud:
            action = 1 if fits_cloud else (0 if fits_onprem else None)
        else:
            action = 0 if fits_onprem else (1 if fits_cloud else None)

        if action is None:
            raise ValueError(
                f"ジョブがどちらの資源にも収まりません。"
                f"job_width={job_width}, job_height={job_height}, "
                f"onprem(H={env.n_on_premise_node},W={env.n_window}), "
                f"cloud(H={env.n_cloud_node},W={env.n_window}), can_use_cloud={can_use_cloud}"
            )

        # 統計更新
        self.stats["util_sum"] += util
        self.stats["n_decisions"] += 1
        if action == 0:
            self.stats["n_onprem"] += 1
            if can_use_cloud == 0:
                self.stats["n_forced_onprem"] += 1
        else:
            self.stats["n_cloud"] += 1

        return action, util



