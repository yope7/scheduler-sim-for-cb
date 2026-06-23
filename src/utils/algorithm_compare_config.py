"""
アルゴリズム横断実験用の設定（config.yml の param_algorithm_compare）を読み取る。

scripts/main.py と scripts/benchmarks/ 配下（micro_benchmark_three_algorithms / mo_benchmark_hv 等）で共有する。
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple, Type

import yaml

# 既定（YAML に param_algorithm_compare が無い場合のフォールバック）
_DEFAULT_PAC: Dict[str, Any] = {
    # アルゴリズム比較で使う環境実装: "c" = SchedulingEnvCacheOptimized（C拡張）, "python" = 純Python SchedulingEnv
    "env_backend": "c",
    "seed": 0,
    "nb_jobs": 24,
    "n_on_premise_node": 256,
    "n_cloud_node": 1024,
    "job_generation_episodes": 2000,
    "nsga2": {
        "pop_size": 100,
        "num_generations": 300,
        "n_workers": -1,
    },
    "dqn_single": {
        "train_episodes": 50000,
    },
    # DQN 重みスイープ + 非支配 + HV（scripts/main.py --mode pareto_distributed）
    "dqn_distributed_pareto": {
        "weight_steps": 10,
        "episodes_per_weight": 1000,
        "num_workers": 4,
    },
    "distributed_pcn": {
        "n_iterations": 100,
        "n_actors": 32,
        "initial_episodes": 100,
        "quick": False,
        "profile": False,
        "use_event_obs": True,
    },
    "micro_benchmark_wall_clock": {
        "nsga_pop": 5,
        "nsga_gen": 2,
        "dqn_episodes": 30,
    },
    "mo_hypervolume_benchmark": {
        "ref_margin": 0.05,
        "tts_fraction": 0.9,
        "hv_threshold": None,
        "out_dir": "mo_hv_out",
        "algorithms": "nsga2,pcn_distributed",
        "dqn_weight_steps": 5,
        "dqn_episodes_per_weight": 500,
        "nsga_n_workers": -1,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_path() -> str:
    return os.path.abspath(os.environ.get("SCHEDULER_CONFIG", "config/config.yml"))


def load_full_config(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or config_path()
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_param_algorithm_compare(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """param_algorithm_compare セクションを既定とマージして返す。"""
    user = cfg.get("param_algorithm_compare")
    if not user or not isinstance(user, dict):
        return copy.deepcopy(_DEFAULT_PAC)
    return _deep_merge(_DEFAULT_PAC, user)


def scheduling_env_class_for_config(cfg: Dict[str, Any]) -> Tuple[Type[Any], str]:
    """
    param_algorithm_compare.env_backend に従い環境クラスを返す。
    比較実験では同一ラン内でこのクラスだけを使うこと（単一要因のため他パラメータは固定）。
    """
    pac = get_param_algorithm_compare(cfg)
    backend = str(pac.get("env_backend", "c")).lower()
    if backend == "python":
        from src.envs.scheduling_env import SchedulingEnv

        return SchedulingEnv, "python"
    from src.envs.scheduling_variants.bitmap_c_env import (
        SchedulingEnvCacheOptimized,
    )

    return SchedulingEnvCacheOptimized, "c"


def job_generation_episodes(cfg: Dict[str, Any]) -> int:
    """JobGenerator 最終引数 nb_episodes: PAC 優先、無ければ param_simulation.nb_episodes。"""
    pac = get_param_algorithm_compare(cfg)
    if "job_generation_episodes" in pac:
        return int(pac["job_generation_episodes"])
    ps = cfg.get("param_simulation") or {}
    return int(ps.get("nb_episodes", 2000))
