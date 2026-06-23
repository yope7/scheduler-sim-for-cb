"""スケジュールマップ（ノード×時刻行列）のオンデマンド生成フラグ。"""
from __future__ import annotations

import os
from typing import Optional


def schedule_maps_enabled(explicit: Optional[bool] = None) -> bool:
    """True のときのみ ``build_schedule_maps`` / ``finalize(..., build_maps=True)`` で行列を構築。"""
    if explicit is not None:
        return bool(explicit)
    if "SCHEDULER_BUILD_SCHEDULE_MAPS" in os.environ:
        return os.environ.get("SCHEDULER_BUILD_SCHEDULE_MAPS", "0") == "1"
    return os.environ.get("DISTRIBUTED_PCN_BUILD_SCHEDULE_MAPS", "0") == "1"
