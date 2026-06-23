#!/usr/bin/env python3
"""
3時代の環境スループット比較（同一ヒューリスティック・同一ジョブ生成シード）

1. python_bitmap  … SchedulingEnv（純Python・ビットマップ）
2. c_bitmap       … SchedulingEnvCacheOptimized（C実装・内部はリングバッファだが観測は従来型）
3. event_c        … SchedulingEnvEventObs（C実装・イベント観測）

ジョブ数・ノード数を変えた格子で wall time（main_sec 等）を記録する。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.heuristic_agent import HeuristicAgent
from src.envs.scheduling_env import SchedulingEnv
from src.envs.scheduling_variants.bitmap_c_env import (
    SchedulingEnvCacheOptimized,
)
from src.envs.scheduling_variants.event_c_env import SchedulingEnvEventObs
from src.utils.job_gen.job_generator import JobGenerator


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def make_jobs_set(
    config: Dict[str, Any],
    nb_jobs: int,
    seed: int,
) -> Dict[int, Any]:
    pe = config["param_env"]
    job_gen = JobGenerator(
        seed,
        1,
        pe["n_window"],
        pe["n_on_premise_node"],
        pe["n_cloud_node"],
        config,
        nb_jobs,
        config["param_job"].get("lam", 0.2),
        0,
    )
    return job_gen.generate_jobs_set()


def run_episode(
    EnvClass,
    config: Dict[str, Any],
    jobs_set: Dict[int, Any],
    nb_jobs: int,
) -> Dict[str, float]:
    pe = config["param_env"]
    pa = config["param_agent"]
    max_step = float("inf")

    t0 = time.perf_counter()
    env = EnvClass(
        max_step,
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
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    env.reset()
    t_reset = time.perf_counter() - t0

    agent = HeuristicAgent(
        base_wait_time_threshold=5,
        width_factor=0.3,
        use_cloud_fallback=True,
    )

    max_steps = min(nb_jobs * 10, 50000)
    step_count = 0
    t0 = time.perf_counter()
    while not env.check_is_done() and step_count < max_steps:
        action, is_valid = agent.select_action(env)
        if is_valid:
            env.step(action)
        else:
            env.step(0)
        step_count += 1
    t_main = time.perf_counter() - t0

    t0 = time.perf_counter()
    env.finalize_window_history()
    t_finalize = time.perf_counter() - t0

    return {
        "init_sec": t_init,
        "reset_sec": t_reset,
        "main_sec": t_main,
        "finalize_sec": t_finalize,
        "total_sec": t_init + t_reset + t_main + t_finalize,
        "steps": step_count,
    }


def apply_node_override(cfg: Dict[str, Any], n_onprem: int, n_cloud: int) -> Dict[str, Any]:
    c = deepcopy(cfg)
    c["param_env"]["n_on_premise_node"] = n_onprem
    c["param_env"]["n_cloud_node"] = n_cloud
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="3時代×ワークロードの環境ベンチマーク")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yml",
        help="ベース設定ファイル",
    )
    parser.add_argument(
        "--nb_jobs",
        type=str,
        default="32,64,128",
        help="カンマ区切りジョブ数",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default="256x1024,512x2048",
        help='ノード組「オンプレxクラウド」をカンマ区切り（例: "256x1024,512x2048"）',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1, help="各セルの繰り返し（平均を取る）")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    nb_list = [int(x.strip()) for x in args.nb_jobs.split(",") if x.strip()]
    node_specs: List[Tuple[str, int, int]] = []
    for part in args.nodes.split(","):
        part = part.strip()
        if not part or "x" not in part:
            continue
        a, b = part.lower().split("x", 1)
        name = part
        node_specs.append((name, int(a), int(b)))

    if not node_specs:
        node_specs = [("default", base_config["param_env"]["n_on_premise_node"], base_config["param_env"]["n_cloud_node"])]

    eras: List[Tuple[str, Any]] = [
        ("python_bitmap", SchedulingEnv),
        ("c_bitmap_ringbuffer", SchedulingEnvCacheOptimized),
        ("event_c", SchedulingEnvEventObs),
    ]

    results: List[Dict[str, Any]] = []

    for node_name, n_onprem, n_cloud in node_specs:
        cfg = apply_node_override(base_config, n_onprem, n_cloud)
        for nb_jobs in nb_list:
            row: Dict[str, Any] = {
                "workload": {
                    "nb_jobs": nb_jobs,
                    "n_on_premise_node": n_onprem,
                    "n_cloud_node": n_cloud,
                    "n_window": cfg["param_env"]["n_window"],
                    "node_preset_label": node_name,
                },
                "eras": {},
            }
            for era_name, EnvClass in eras:
                runs = []
                for run_idx in range(args.n_runs):
                    jobs_set = make_jobs_set(cfg, nb_jobs, args.seed + run_idx)
                    r = run_episode(EnvClass, cfg, jobs_set, nb_jobs)
                    runs.append(r)
                mean_main = float(np.mean([x["main_sec"] for x in runs]))
                mean_total = float(np.mean([x["total_sec"] for x in runs]))
                row["eras"][era_name] = {
                    "runs": runs,
                    "mean_main_sec": mean_main,
                    "mean_total_sec": mean_total,
                }
            py_t = row["eras"]["python_bitmap"]["mean_main_sec"]
            for era_name in row["eras"]:
                m = row["eras"][era_name]["mean_main_sec"]
                row["eras"][era_name]["speedup_vs_python_bitmap"] = (
                    (py_t / m) if m > 0 else None
                )
            results.append(row)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(args.config.resolve()),
        "seed": args.seed,
        "n_runs": args.n_runs,
        "description": (
            "python_bitmap=純Python SchedulingEnv; "
            "c_bitmap_ringbuffer=C環境(リングバッファ); "
            "event_c=C環境+イベント観測"
        ),
        "results": results,
    }

    out = args.output
    if out is None or out == "auto":
        out = str(
            PROJECT_ROOT
            / "0403"
            / "raw"
            / f"benchmark_env_three_eras_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    n = len(payload["results"])
    print(f"完了: {n} ワークロード × 3 時代。保存: {out}")


if __name__ == "__main__":
    main()
