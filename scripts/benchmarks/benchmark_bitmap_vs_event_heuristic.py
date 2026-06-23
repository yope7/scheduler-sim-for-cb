#!/usr/bin/env python3
"""
C ビットマップ観測（SchedulingEnvCacheOptimized） vs イベント観測（SchedulingEnvEventObs）
同一ヒューリスティック・同一ジョブシードでの wall time 比較。

ジョブ数 2 パターン × ノード（オンプレ×クラウド）2 パターン = 4 ワークロード。
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
    parser = argparse.ArgumentParser(
        description="Cビットマップ vs イベント観測（ヒューリスティック）の時間計測"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yml",
        help="ベース設定ファイル",
    )
    parser.add_argument(
        "--nb_jobs",
        type=str,
        default="32,64",
        help="カンマ区切りジョブ数（2パターン想定）",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default="256x1024,512x2048",
        help='ノード組「オンプレxクラウド」をカンマ区切り（例: "256x1024,512x2048"）',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1, help="各セル・各モードの繰り返し（平均）")
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
        node_specs = [
            (
                "default",
                base_config["param_env"]["n_on_premise_node"],
                base_config["param_env"]["n_cloud_node"],
            )
        ]

    modes: List[Tuple[str, Any]] = [
        ("c_bitmap", SchedulingEnvCacheOptimized),
        ("event_obs", SchedulingEnvEventObs),
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
                "modes": {},
            }
            for mode_name, EnvClass in modes:
                runs = []
                for run_idx in range(args.n_runs):
                    jobs_set = make_jobs_set(cfg, nb_jobs, args.seed + run_idx)
                    r = run_episode(EnvClass, cfg, jobs_set, nb_jobs)
                    runs.append(r)
                mean_main = float(np.mean([x["main_sec"] for x in runs]))
                mean_total = float(np.mean([x["total_sec"] for x in runs]))
                row["modes"][mode_name] = {
                    "runs": runs,
                    "mean_main_sec": mean_main,
                    "mean_total_sec": mean_total,
                }
            bm = row["modes"]["c_bitmap"]["mean_main_sec"]
            ev = row["modes"]["event_obs"]["mean_main_sec"]
            row["ratio_event_vs_bitmap_main"] = (ev / bm) if bm > 0 else None
            results.append(row)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(args.config.resolve()),
        "seed": args.seed,
        "n_runs": args.n_runs,
        "description": (
            "c_bitmap=SchedulingEnvCacheOptimized（C・ビットマップ観測）; "
            "event_obs=SchedulingEnvEventObs（イベント観測）; "
            "同一 HeuristicAgent(base_wait_time_threshold=5, width_factor=0.3)"
        ),
        "results": results,
    }

    out = args.output
    if out is None or out == "auto":
        out_dir = PROJECT_ROOT / "0511"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(
            out_dir / f"benchmark_bitmap_vs_event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 人間向けサマリ
    lines = [
        "bitmap vs event (heuristic) — mean_main_sec",
        "",
        f"{'jobs':>6} {'on×cloud':>14} {'c_bitmap':>12} {'event_obs':>12} {'event/bitmap':>12}",
        "-" * 62,
    ]
    for row in results:
        w = row["workload"]
        m = row["modes"]
        label = f"{w['n_on_premise_node']}×{w['n_cloud_node']}"
        ratio = row.get("ratio_event_vs_bitmap_main")
        ratio_s = f"{ratio:.4f}" if ratio is not None else "n/a"
        lines.append(
            f"{w['nb_jobs']:>6} {label:>14} "
            f"{m['c_bitmap']['mean_main_sec']:>12.6f} {m['event_obs']['mean_main_sec']:>12.6f} {ratio_s:>12}"
        )
    summary_path = Path(out).with_suffix(".summary.txt")
    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"JSON: {out}")
    print(f"サマリ: {summary_path}")


if __name__ == "__main__":
    main()
