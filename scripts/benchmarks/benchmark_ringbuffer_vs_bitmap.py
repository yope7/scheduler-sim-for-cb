#!/usr/bin/env python3
"""
リングバッファ版 vs ビットマップ版 の実行時間比較ベンチマーク

複数のジョブ数・ウィンドウサイズで両実装を実行し、実行時間を比較する。
"""

import sys
import os
import json
import numpy as np
import time
import yaml
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.job_gen.job_generator import JobGenerator


def load_config(path: str = "config/config.yml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_jobs_set(nb_jobs: int, config: Dict, seed: int = 42) -> Dict:
    """固定シードでジョブセットを作成（両実装で同じジョブを使用）"""
    lam = config['param_job'].get('lam', 0.2)
    job_gen = JobGenerator(
        seed, 1,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, nb_jobs, lam, 0
    )
    return job_gen.generate_jobs_set()


def load_bitmap_env():
    """バックアップからビットマップ版の環境クラスを読み込み"""
    backup_path = str(
        REPO_ROOT / "src" / "envs" / "backup_bitmap" / "scheduling_env_cache_optimized.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scheduling_env_cache_optimized_bitmap", backup_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scheduling_env_cache_optimized_bitmap"] = mod
    spec.loader.exec_module(mod)
    return mod.SchedulingEnvCacheOptimized


def load_ringbuffer_env():
    """現在のリングバッファ版の環境クラスを読み込み"""
    from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
    return SchedulingEnvCacheOptimized


def run_episode(EnvClass, config: Dict, jobs_set: Dict, nb_jobs: int) -> Dict[str, float]:
    """1エピソードを実行し、各フェーズの時間を返す"""
    max_step = float("inf")
    n_window = config['param_env']['n_window']
    n_on_premise_node = config['param_env']['n_on_premise_node']
    n_cloud_node = config['param_env']['n_cloud_node']
    n_job_queue_obs = config['param_env']['n_job_queue_obs']
    n_job_queue_bck = config['param_env']['n_job_queue_bck']
    weight_wt = config['param_agent']['weight_wt']
    weight_cost = config['param_agent']['weight_cost']
    penalty_not_allocate = config['param_env']['penalty_not_allocate']
    penalty_invalid_action = config['param_env']['penalty_invalid_action']

    t0 = time.perf_counter()
    env = EnvClass(
        max_step, n_window, n_on_premise_node, n_cloud_node,
        n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action,
        jobs_set, None, flag=0
    )
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    env.reset()
    t_reset = time.perf_counter() - t0

    from src.agents.heuristic_agent import HeuristicAgent
    agent = HeuristicAgent(
        base_wait_time_threshold=5,
        width_factor=0.3,
        use_cloud_fallback=True
    )

    max_steps = min(nb_jobs * 10, 50000)
    step_count = 0
    done = False
    t0 = time.perf_counter()
    while not env.check_is_done() and step_count < max_steps:
        action, is_valid = agent.select_action(env)
        if is_valid:
            obs, rewards, scheduled, wt_step, done = env.step(action)
        else:
            obs, rewards, scheduled, wt_step, done = env.step(0)
        step_count += 1
        if done:
            break
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


def run_benchmark(
    nb_jobs_list: List[int],
    n_runs: int = 2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """複数サイズで両実装をベンチマーク"""
    config = load_config()
    BitmapEnv = load_bitmap_env()
    RingbufferEnv = load_ringbuffer_env()

    results_bitmap = []
    results_ringbuffer = []

    for nb_jobs in nb_jobs_list:
        print(f"\n--- nb_jobs={nb_jobs} ---")
        jobs_set = create_jobs_set(nb_jobs, config, seed=seed)

        # ビットマップ版
        times_b = []
        for r in range(n_runs):
            t0 = time.perf_counter()
            res = run_episode(BitmapEnv, config, jobs_set, nb_jobs)
            times_b.append(res)
            print(f"  Bitmap    run{r+1}: {res['total_sec']:.3f}s (main={res['main_sec']:.3f}s)")
        results_bitmap.append({
            "nb_jobs": nb_jobs,
            "runs": times_b,
            "mean_total": np.mean([r["total_sec"] for r in times_b]),
            "mean_main": np.mean([r["main_sec"] for r in times_b]),
        })

        # リングバッファ版（同じジョブセットを再生成）
        jobs_set = create_jobs_set(nb_jobs, config, seed=seed)
        times_r = []
        for r in range(n_runs):
            t0 = time.perf_counter()
            res = run_episode(RingbufferEnv, config, jobs_set, nb_jobs)
            times_r.append(res)
            print(f"  Ringbuf   run{r+1}: {res['total_sec']:.3f}s (main={res['main_sec']:.3f}s)")
        results_ringbuffer.append({
            "nb_jobs": nb_jobs,
            "runs": times_r,
            "mean_total": np.mean([r["total_sec"] for r in times_r]),
            "mean_main": np.mean([r["main_sec"] for r in times_r]),
        })

    return results_bitmap, results_ringbuffer


def print_comparison_table(results_bitmap: List[Dict], results_ringbuffer: List[Dict]):
    """比較表を出力"""
    print("\n" + "=" * 90)
    print("実行時間比較: ビットマップ vs リングバッファ")
    print("=" * 90)
    print(f"{'nb_jobs':>8} | {'Bitmap総(秒)':>12} | {'Ringbuf総(秒)':>12} | {'高速化':>8} | {'Bitmap main':>12} | {'Ringbuf main':>12}")
    print("-" * 90)

    for i, nb in enumerate([r["nb_jobs"] for r in results_bitmap]):
        b = results_bitmap[i]
        r = results_ringbuffer[i]
        speedup = b["mean_total"] / r["mean_total"] if r["mean_total"] > 0 else 0
        print(f"{nb:>8} | {b['mean_total']:>12.3f} | {r['mean_total']:>12.3f} | {speedup:>7.2f}x | {b['mean_main']:>12.3f} | {r['mean_main']:>12.3f}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="リングバッファ vs ビットマップ 実行時間比較")
    parser.add_argument("--nb_jobs", type=str, default="20,50,100,200,500",
                        help="カンマ区のジョブ数リスト (default: 20,50,100,200,500)")
    parser.add_argument("--n_runs", type=int, default=2, help="各サイズの実行回数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="結果をJSONで保存するパス (例: results/benchmark_YYYYMMDD_HHMMSS.json)")
    args = parser.parse_args()

    nb_jobs_list = [int(x.strip()) for x in args.nb_jobs.split(",")]
    print(f"ベンチマーク: nb_jobs={nb_jobs_list}, n_runs={args.n_runs}, seed={args.seed}")

    results_bitmap, results_ringbuffer = run_benchmark(
        nb_jobs_list, n_runs=args.n_runs, seed=args.seed
    )

    print_comparison_table(results_bitmap, results_ringbuffer)

    if args.output:
        out_path = args.output
        if out_path == "auto":
            out_path = f"benchmark_ringbuffer_vs_bitmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "nb_jobs_list": nb_jobs_list,
            "n_runs": args.n_runs,
            "seed": args.seed,
            "bitmap": [{"nb_jobs": r["nb_jobs"], "mean_total": r["mean_total"], "mean_main": r["mean_main"]} for r in results_bitmap],
            "ringbuffer": [{"nb_jobs": r["nb_jobs"], "mean_total": r["mean_total"], "mean_main": r["mean_main"]} for r in results_ringbuffer],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
