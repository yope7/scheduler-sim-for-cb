"""
観測表現別のスケジューリング環境バリアント。

- ``bitmap_c_env.SchedulingEnvCacheOptimized`` … C + リングバッファ・ビットマップ観測（既定の高速環境）
- ``event_c_env.SchedulingEnvEventObs`` … 同上ロジックにイベント列観測

使い分けの目安::

    from src.envs.scheduling_variants import OBSERVATION_ENV_BY_KEY
    EnvCls = OBSERVATION_ENV_BY_KEY["bitmap_c"]   # または "event_c"

クラスを直接 import してもよい::

    from src.envs.scheduling_variants import SchedulingEnvCacheOptimized, SchedulingEnvEventObs
"""

from .bitmap_c_env import SchedulingEnvCacheOptimized
from .event_c_env import SchedulingEnvEventObs
from .event_native_env import SchedulingEnvEventNative

OBSERVATION_ENV_BY_KEY = {
    "bitmap_c": SchedulingEnvCacheOptimized,
    "event_c": SchedulingEnvEventObs,
    "event_native": SchedulingEnvEventNative,
}

__all__ = [
    "SchedulingEnvCacheOptimized",
    "SchedulingEnvEventObs",
    "SchedulingEnvEventNative",
    "OBSERVATION_ENV_BY_KEY",
]
