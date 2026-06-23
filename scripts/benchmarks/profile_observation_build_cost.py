#!/usr/bin/env python3
"""
bitmap vs event: step 内の get_observation 構築時間の内訳検証用。
ヒューリスティックは観測を読まないが、SchedulingEnv.step は成功時に必ず get_observation を返す。
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.heuristic_agent import HeuristicAgent
from src.envs.scheduling_variants.bitmap_c_env import (
    SchedulingEnvCacheOptimized,
)
from src.envs.scheduling_variants.event_c_env import SchedulingEnvEventObs
from src.utils.job_gen.job_generator import JobGenerator


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_nodes(cfg: dict, n_onprem: int, n_cloud: int) -> dict:
    c = deepcopy(cfg)
    c["param_env"]["n_on_premise_node"] = n_onprem
    c["param_env"]["n_cloud_node"] = n_cloud
    return c


def make_jobs(cfg: dict, nb_jobs: int, seed: int) -> dict:
    pe = cfg["param_env"]
    jg = JobGenerator(
        seed,
        1,
        pe["n_window"],
        pe["n_on_premise_node"],
        pe["n_cloud_node"],
        cfg,
        nb_jobs,
        cfg["param_job"].get("lam", 0.2),
        0,
    )
    return jg.generate_jobs_set()


class TimedBitmapEnv(SchedulingEnvCacheOptimized):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._sec_get_obs = 0.0
        self._n_get_obs = 0

    def get_observation(self):
        t0 = time.perf_counter()
        o = super().get_observation()
        self._sec_get_obs += time.perf_counter() - t0
        self._n_get_obs += 1
        return o


class TimedEventEnv(SchedulingEnvEventObs):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._sec_get_obs = 0.0
        self._n_get_obs = 0
        self._sec_do_schedule_extra = 0.0

    def get_observation(self):
        t0 = time.perf_counter()
        o = super().get_observation()
        self._sec_get_obs += time.perf_counter() - t0
        self._n_get_obs += 1
        return o

    def do_schedule(self, action, job, position):
        t0 = time.perf_counter()
        wt = super().do_schedule(action, job, position)
        self._sec_do_schedule_extra += time.perf_counter() - t0
        return wt


def run_episode(EnvClass, cfg: dict, jobs_set: dict, nb_jobs: int):
    pe = cfg["param_env"]
    pa = cfg["param_agent"]
    env = EnvClass(
        float("inf"),
        pe["n_window"],
        pe["n_on_premise_node"],
        pe["n_cloud_node"],
        pe["n_job_queue_obs"],
        pe["n_job_queue_bck"],
        pa["weight_wt"],
        pa["weight_cost"],
        pe["penalty_not_allocate"],
        pe["penalty_invalid_action"],
        jobs_set,
        None,
        flag=0,
    )
    agent = HeuristicAgent(
        base_wait_time_threshold=5,
        width_factor=0.3,
        use_cloud_fallback=True,
    )
    env.reset()

    t_main0 = time.perf_counter()
    steps = 0
    max_steps = min(nb_jobs * 10, 50000)
    while not env.check_is_done() and steps < max_steps:
        action, is_valid = agent.select_action(env)
        if is_valid:
            env.step(action)
        else:
            env.step(0)
        steps += 1
    t_main = time.perf_counter() - t_main0

    out = {
        "steps": steps,
        "main_sec": t_main,
        "get_obs_sec": getattr(env, "_sec_get_obs", 0.0),
        "get_obs_calls": getattr(env, "_n_get_obs", 0),
        "do_schedule_extra_sec": getattr(env, "_sec_do_schedule_extra", 0.0),
        "max_scheduled_events": (
            int(env._event_buf.count)
            if getattr(env, "_event_buf", None) is not None
            else len(getattr(env, "_scheduled_events", []) or [])
        ),
    }
    return out


def main():
    base = load_config(PROJECT_ROOT / "config" / "config.yml")
    nb_jobs = 64
    cfg = apply_nodes(base, 256, 1024)
    jobs = make_jobs(cfg, nb_jobs, seed=42)

    r_b = run_episode(TimedBitmapEnv, cfg, jobs, nb_jobs)
    r_e = run_episode(TimedEventEnv, cfg, jobs, nb_jobs)

    # 同一ジョブで再実行 sanity（イベント環境は super().do_schedule を挟むので extra は全体ではなくオーバーヘッド近似）
    print("=== 同一ワークロード (nb_jobs=64, 256×1024) ===\n")
    print(
        "Bitmap (C ringbuffer get_observation):\n"
        f"  main_sec={r_b['main_sec']:.6f}\n"
        f"  get_observation 累計={r_b['get_obs_sec']:.6f}s ({100*r_b['get_obs_sec']/r_b['main_sec']:.1f}% of main)\n"
        f"  get_observation 呼び出し回数={r_b['get_obs_calls']}\n"
    )
    print(
        "Event (C: get_observation_event + イベント記録は Python do_schedule 内):\n"
        f"  main_sec={r_e['main_sec']:.6f}\n"
        f"  get_observation 累計={r_e['get_obs_sec']:.6f}s ({100*r_e['get_obs_sec']/r_e['main_sec']:.1f}% of main)\n"
        f"  get_observation 呼び出し回数={r_e['get_obs_calls']}\n"
        f"  do_schedule 計測時間(親含む全体)={r_e['do_schedule_extra_sec']:.6f}s\n"
        f"  記録イベント数(終了時)={r_e['max_scheduled_events']}\n"
    )
    ratio_obs = r_e["get_obs_sec"] / max(r_b["get_obs_sec"], 1e-12)
    print(
        f"get_observation 時間比 (event/bitmap) = {ratio_obs:.2f}x\n"
        "\n結論: step() は配置成功のたびに観測を構築する。\n"
        "ビットマップは get_observation_ringbuffer（C）、イベントは get_observation_event（C）。\n"
        "旧 Python 実装ではイベント側のフィルタ・ソートが律速になりやすかった。\n"
    )


if __name__ == "__main__":
    main()
