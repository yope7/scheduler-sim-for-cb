import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class WaitTimeThresholdPolicy:
    """
    ルール:
    - 「オンプレに割り当てた場合の予測待ち時間」が threshold 以上ならクラウドへ（可能なら）
    - それ以外はオンプレ

    予測待ち時間は env.find_allocation_position([0,0]) が返す waiting_time を使用する。
    （SchedulingEnv実装では、開始時刻 - 提出時刻 に相当）
    """

    threshold: float  # 0〜10など（単位はEnvの時間ステップ）
    strict_greater_equal: bool = True  # 「閾値以上」なのでデフォルトは >=

    stats: Optional[Dict] = None

    def reset_stats(self) -> None:
        self.stats = {
            "n_decisions": 0,
            "n_onprem": 0,
            "n_cloud": 0,
            "n_forced_onprem": 0,
            "wt_onprem_sum": 0.0,
            "wt_onprem_inf": 0,
        }

    def select_action(self, env) -> Tuple[int, float]:
        """
        Returns:
            (action_raw, predicted_onprem_wait_time)
            action_raw: 0=オンプレ, 1=クラウド
        """
        if self.stats is None:
            self.reset_stats()

        # キューが空ならオンプレを返す（env.step内で時間遷移して次のジョブを待つ）
        if getattr(env, "rear_job_queue", 0) <= 0 or np.all(env.job_queue[0] == 0):
            wt_onprem = 0.0
            self.stats["n_decisions"] += 1
            self.stats["n_onprem"] += 1
            return 0, wt_onprem

        job = env.job_queue[0]
        can_use_cloud = int(job[2]) if len(job) > 2 else 1

        # オンプレでの予測待ち時間を取得
        _, wt_onprem = env.find_allocation_position([0, 0])
        wt_onprem = float(wt_onprem)
        if not np.isfinite(wt_onprem):
            self.stats["wt_onprem_inf"] += 1

        self.stats["wt_onprem_sum"] += (wt_onprem if np.isfinite(wt_onprem) else 0.0)

        # 閾値判定
        if self.strict_greater_equal:
            offload = (wt_onprem >= self.threshold)
        else:
            offload = (wt_onprem > self.threshold)

        # まず希望先を決める
        prefer_cloud = offload and (can_use_cloud == 1)

        if prefer_cloud:
            # クラウドに置けるならクラウド、置けないならオンプレ
            pos_cloud, _ = env.find_allocation_position([0, 1])
            action = 1 if pos_cloud is not None else 0
        else:
            # オンプレに置けるならオンプレ、置けないならクラウド（可能なら）
            pos_onprem, _ = env.find_allocation_position([0, 0])
            if pos_onprem is not None:
                action = 0
            else:
                if can_use_cloud == 1:
                    pos_cloud, _ = env.find_allocation_position([0, 1])
                    action = 1 if pos_cloud is not None else 0
                else:
                    action = 0

        # stats更新
        self.stats["n_decisions"] += 1
        if action == 0:
            self.stats["n_onprem"] += 1
            if can_use_cloud == 0:
                self.stats["n_forced_onprem"] += 1
        else:
            self.stats["n_cloud"] += 1

        return action, wt_onprem



