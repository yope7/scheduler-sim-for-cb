#!/usr/bin/env python3
"""
オンプレ予測待ち時間が閾値以上ならクラウドに流す（AIなし）ルールをsweepして、
待ち時間とコストのトレードオフを実験・可視化するスクリプト。

要件:
- Cで書かれたEnv（scheduling_env_core）を呼ぶ（SchedulingEnvCacheOptimizedを使用）
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

# repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.font_setup import setup_japanese_font
from src.utils.job_gen.job_generator import JobGenerator
from src.agents.waittime_threshold_agent import WaitTimeThresholdPolicy

try:
    from src.envs.scheduling_variants.bitmap_c_env import SchedulingEnvCacheOptimized
    C_AVAILABLE = True
except Exception:
    SchedulingEnvCacheOptimized = None
    C_AVAILABLE = False


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env(config: Dict, jobs_set: Dict) -> "SchedulingEnvCacheOptimized":
    if not C_AVAILABLE or SchedulingEnvCacheOptimized is None:
        raise ImportError(
            "C言語実装版の環境（SchedulingEnvCacheOptimized）が利用できません。"
            "先に `uv sync` でプロジェクトをセットアップしてください。"
        )
    return SchedulingEnvCacheOptimized(
        max_step=np.inf,
        n_window=config["param_env"]["n_window"],
        n_on_premise_node=config["param_env"]["n_on_premise_node"],
        n_cloud_node=config["param_env"]["n_cloud_node"],
        n_job_queue_obs=config["param_env"]["n_job_queue_obs"],
        n_job_queue_bck=config["param_env"]["n_job_queue_bck"],
        weight_wt=config["param_agent"]["weight_wt"],
        weight_cost=config["param_agent"]["weight_cost"],
        penalty_not_allocate=config["param_env"]["penalty_not_allocate"],
        penalty_invalid_action=config["param_env"]["penalty_invalid_action"],
        jobs_set=jobs_set,
        flag=0,
    )


def generate_jobs_set(config: Dict, seed: int, nb_jobs: int, nb_episodes: int, lam: float) -> Dict:
    job_gen = JobGenerator(
        seed,
        config["param_simulation"]["nb_steps"],
        config["param_env"]["n_window"],
        config["param_env"]["n_on_premise_node"],
        config["param_env"]["n_cloud_node"],
        config,
        nb_jobs,
        lam,
        nb_episodes,
    )
    return job_gen.generate_jobs_set()


def thresholds_int(min_th: int, max_th: int) -> List[int]:
    if max_th < min_th:
        raise ValueError("max_th must be >= min_th")
    return list(range(int(min_th), int(max_th) + 1))


def run_one_episode(env, policy: WaitTimeThresholdPolicy, episode_idx: int) -> Dict:
    env.episode = episode_idx
    policy.reset_stats()
    _ = env.reset()

    total_wait = 0.0
    total_cost = 0.0
    n_scheduled = 0
    wt_onprem_pred_sum = 0.0
    wt_onprem_pred_n = 0

    done = False
    while not done:
        action_raw, wt_onprem_pred = policy.select_action(env)
        if np.isfinite(wt_onprem_pred):
            wt_onprem_pred_sum += float(wt_onprem_pred)
            wt_onprem_pred_n += 1

        _, rewards, scheduled, wt_step, done = env.step(action_raw)
        wait = float(-rewards[0])
        cost = float(-rewards[1])
        total_wait += wait
        total_cost += cost
        if scheduled:
            n_scheduled += 1

    env.finalize_window_history()

    avg_wait = total_wait / n_scheduled if n_scheduled > 0 else 0.0
    avg_pred_onprem_wait = wt_onprem_pred_sum / wt_onprem_pred_n if wt_onprem_pred_n > 0 else 0.0

    return {
        "episode": episode_idx,
        "n_scheduled": n_scheduled,
        "total_wait": total_wait,
        "avg_wait": avg_wait,
        "total_cost": total_cost,
        "avg_pred_onprem_wait": avg_pred_onprem_wait,
        "n_onprem": policy.stats["n_onprem"],
        "n_cloud": policy.stats["n_cloud"],
        "n_forced_onprem": policy.stats["n_forced_onprem"],
    }


def plot_all_points(df_per_episode: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    setup_japanese_font(required=True)

    x = df_per_episode["total_cost"].to_numpy(dtype=float)
    y = df_per_episode["avg_wait"].to_numpy(dtype=float)
    c = df_per_episode["wt_threshold"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=60, alpha=0.75, edgecolors="none")
    cb = plt.colorbar(sc)
    cb.set_label("待ち時間しきい値")
    plt.xlabel("コスト（クラウド利用面積の合計）")
    plt.ylabel("待ち時間（開始時刻までの待ち：エピソード平均）")
    plt.title("待ち時間しきい値 sweep：全点（全エピソード）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_summary(df_summary: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    setup_japanese_font(required=True)

    x = df_summary["mean_cost"].values
    y = df_summary["mean_avg_wait"].values
    th = df_summary["wt_threshold"].values

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, s=80)
    for xi, yi, thi in zip(x, y, th):
        plt.annotate(f"{int(thi)}", (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xlabel("平均コスト（クラウド利用面積の合計）")
    plt.ylabel("平均待ち時間（開始時刻までの待ち）")
    plt.title("待ち時間しきい値 sweep：待ち時間 vs コスト（平均）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yml")
    p.add_argument("--nb_jobs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--min_wt_th", type=int, default=0, help="待ち時間しきい値の最小（整数）")
    p.add_argument("--max_wt_th", type=int, default=10, help="待ち時間しきい値の最大（整数）")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    config = load_config(args.config)
    lam = float(args.lam) if args.lam is not None else float(config.get("param_job", {}).get("lam", 0.2))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"experiments/wt_threshold_sweep/{ts}_jobs{args.nb_jobs}_ep{args.episodes}_seed{args.seed}_lam{lam}"
    os.makedirs(out_dir, exist_ok=True)

    wt_thresholds = thresholds_int(args.min_wt_th, args.max_wt_th)

    jobs_set = generate_jobs_set(config, seed=args.seed, nb_jobs=args.nb_jobs, nb_episodes=args.episodes, lam=lam)
    env = build_env(config, jobs_set)

    rows = []
    for wt_th in wt_thresholds:
        policy = WaitTimeThresholdPolicy(threshold=float(wt_th), strict_greater_equal=True)
        for ep in range(args.episodes):
            r = run_one_episode(env, policy, episode_idx=ep)
            r.update(
                {
                    "wt_threshold": int(wt_th),
                    "seed": int(args.seed),
                    "nb_jobs": int(args.nb_jobs),
                    "lam": float(lam),
                }
            )
            rows.append(r)

    df = pd.DataFrame(rows)
    per_episode_csv = os.path.join(out_dir, "per_episode.csv")
    df.to_csv(per_episode_csv, index=False)

    g = df.groupby("wt_threshold", as_index=False)
    df_summary = g.agg(
        mean_avg_wait=("avg_wait", "mean"),
        std_avg_wait=("avg_wait", "std"),
        mean_cost=("total_cost", "mean"),
        std_cost=("total_cost", "std"),
        mean_cloud_jobs=("n_cloud", "mean"),
        mean_onprem_jobs=("n_onprem", "mean"),
        mean_pred_onprem_wait=("avg_pred_onprem_wait", "mean"),
    ).sort_values("wt_threshold")

    summary_csv = os.path.join(out_dir, "summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    plot_all_png = os.path.join(out_dir, "tradeoff_all.png")
    plot_summary_png = os.path.join(out_dir, "tradeoff_summary.png")
    plot_all_points(df, plot_all_png)
    plot_summary(df_summary, plot_summary_png)

    print("=== 完了 ===")
    print(f"- out_dir    : {out_dir}")
    print(f"- per-episode: {per_episode_csv}")
    print(f"- summary    : {summary_csv}")
    print(f"- plot(all)  : {plot_all_png}")
    print(f"- plot(mean) : {plot_summary_png}")


if __name__ == "__main__":
    main()



