"""Learner replay スナップショットの読み込みと eval 用環境構築。"""
from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.agents.pcn_agent import get_non_dominated_inds_minimize
from src.envs.scheduling_variants.event_c_env import SchedulingEnvEventObs
from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator


def load_config(config_path: str | None = None) -> dict:
    path = config_path or os.environ.get("DISTRIBUTED_PCN_CONFIG", "config/config.yml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_n_jobs(config: dict, default: int = 24) -> int:
    dpc = config.get("param_algorithm_compare", {}).get("distributed_pcn", {})
    if "n_jobs" in dpc:
        return int(dpc["n_jobs"])
    nb = config.get("param_algorithm_compare", {}).get("nb_jobs")
    if nb is not None:
        return int(nb)
    return int(config.get("param_env", {}).get("n_jobs", default))


def create_eval_env(config: dict, job_seed: int = 0, n_jobs: Optional[int] = None):
    """JobGenerator 生成前に n_jobs を確定させる。"""
    cfg = yaml.safe_load(yaml.dump(config))
    nj = int(n_jobs if n_jobs is not None else eval_n_jobs(cfg))
    cfg["param_env"]["n_jobs"] = nj
    jg = JobGenerator(
        job_seed,
        1,
        cfg["param_env"]["n_window"],
        cfg["param_env"]["n_on_premise_node"],
        cfg["param_env"]["n_cloud_node"],
        cfg,
        nj,
        0.2,
        0,
    )
    jobs_set = jg.generate_jobs_set()
    env_cls = SchedulingEnvEventNative
    if os.environ.get("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1") != "1":
        env_cls = SchedulingEnvEventObs
    return env_cls(
        np.inf,
        cfg["param_env"]["n_window"],
        cfg["param_env"]["n_on_premise_node"],
        cfg["param_env"]["n_cloud_node"],
        cfg["param_env"]["n_job_queue_obs"],
        cfg["param_env"]["n_job_queue_bck"],
        cfg["param_agent"]["weight_wt"],
        cfg["param_agent"]["weight_cost"],
        cfg["param_env"]["penalty_not_allocate"],
        cfg["param_env"]["penalty_invalid_action"],
        jobs_set,
        None,
        flag=0,
    )


def load_learner_replay_snapshot(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "episodes" in payload:
        return payload
    return {"metadata": {}, "episodes": payload}


def episode_objectives_from_snapshot(snapshot: Dict[str, Any], n_jobs: int) -> np.ndarray:
    """replay 内の全エピソード先頭の (Cost, AvgWait) — 探索・学習で到達した点の集合。"""
    episodes = snapshot.get("episodes", [])
    values = []
    for episode in episodes:
        if not episode:
            continue
        first = episode[0]
        if hasattr(first, "objective_values") and first.objective_values is not None:
            obj = first.objective_values
            values.append([float(obj[0]), float(obj[2])])
        else:
            r = np.asarray(first.reward, dtype=np.float64)
            values.append([-float(r[1]), -float(r[0]) / max(1, n_jobs)])
    if not values:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def archive_pf_from_snapshot(snapshot: Dict[str, Any], n_jobs: int) -> np.ndarray:
    arr = episode_objectives_from_snapshot(snapshot, n_jobs)
    if not arr.size:
        return arr
    nd = get_non_dominated_inds_minimize(arr)
    return arr[nd]


def find_replay_snapshot_for_checkpoint(checkpoint: Path) -> Optional[Path]:
    """iteration_XXX/model.pth と同じ実行ディレクトリの learner_replay_snapshot.pkl.gz を探す。"""
    ckpt = checkpoint.resolve()
    for parent in [ckpt.parent.parent, ckpt.parent.parent.parent]:
        cand = parent / "learner_replay_snapshot.pkl.gz"
        if cand.is_file():
            return cand
    return None
