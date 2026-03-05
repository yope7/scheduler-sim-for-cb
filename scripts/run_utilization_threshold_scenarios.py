#!/usr/bin/env python3
"""
利用率しきい値sweepを、複数の実験シナリオ（ジョブ数×資源サイズ）で一括実行するドライバ。

シナリオ（ユーザ指定）:
- ジョブ数: {32, 128}
- リソースサイズ（オンプレ,クラウド）: {(512,1024), (1024,2048)}
の掛け合わせ（計4シナリオ）

各シナリオで、以下を保存:
- per_episode.csv（全点）
- summary.csv（しきい値ごとの集計）
- tradeoff_all.png（全点プロット）
- tradeoff_summary.png（集計点プロット）
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 既存実験ロジックを再利用
import scripts.utilization_threshold_sweep as sweep


def scenario_grid() -> List[Dict]:
    jobs_list = [32, 128]
    resource_pairs = [(512, 1024), (1024, 2048)]
    scenarios = []
    for nb_jobs in jobs_list:
        for on_nodes, cloud_nodes in resource_pairs:
            scenarios.append(
                {
                    "nb_jobs": nb_jobs,
                    "n_on_premise_node": on_nodes,
                    "n_cloud_node": cloud_nodes,
                }
            )
    return scenarios


def run_one_scenario(
    base_config: Dict,
    scenario: Dict,
    out_dir: str,
    *,
    episodes: int,
    seed: int,
    lam: float,
    thresholds: List[float],
    util_mode: str,
    util_k: int,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    # configをコピーしてシナリオを反映
    config = {**base_config}
    config["param_env"] = {**base_config["param_env"]}
    config["param_simulation"] = {**base_config["param_simulation"]}
    config["param_job"] = {**base_config.get("param_job", {})}
    config["param_agent"] = {**base_config.get("param_agent", {})}

    config["param_env"]["n_on_premise_node"] = int(scenario["n_on_premise_node"])
    config["param_env"]["n_cloud_node"] = int(scenario["n_cloud_node"])

    nb_jobs = int(scenario["nb_jobs"])

    # ジョブセット生成（このシナリオの資源サイズに合わせる）
    jobs_set = sweep.generate_jobs_set(config, seed=seed, nb_jobs=nb_jobs, nb_episodes=episodes, lam=lam)
    env = sweep.build_env(config, jobs_set)

    thresholds = [float(t) for t in thresholds]

    rows = []
    for th in thresholds:
        policy = sweep.UtilizationThresholdPolicy(
            threshold=float(th),
            utilization_mode=util_mode,
            utilization_k=util_k,
        )
        for ep in range(episodes):
            r = sweep.run_one_episode(env, policy, episode_idx=ep)
            r.update(
                {
                    "threshold": float(th),
                    "seed": int(seed),
                    "nb_jobs": nb_jobs,
                    "lam": float(lam),
                    "n_on_premise_node": int(scenario["n_on_premise_node"]),
                    "n_cloud_node": int(scenario["n_cloud_node"]),
                    "util_mode": util_mode,
                    "util_k": int(util_k),
                }
            )
            rows.append(r)

    df = pd.DataFrame(rows)
    per_episode_csv = os.path.join(out_dir, "per_episode.csv")
    df.to_csv(per_episode_csv, index=False)

    g = df.groupby("threshold", as_index=False)
    df_summary = g.agg(
        mean_avg_wait=("avg_wait", "mean"),
        std_avg_wait=("avg_wait", "std"),
        mean_cost=("total_cost", "mean"),
        std_cost=("total_cost", "std"),
        mean_cloud_jobs=("n_cloud", "mean"),
        mean_onprem_jobs=("n_onprem", "mean"),
        mean_onprem_util=("avg_onprem_util", "mean"),
    ).sort_values("threshold")

    summary_csv = os.path.join(out_dir, "summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    plot_all_png = os.path.join(out_dir, "tradeoff_all.png")
    plot_summary_png = os.path.join(out_dir, "tradeoff_summary.png")
    sweep.plot_all_points(df, plot_all_png)
    sweep.plot_summary(df_summary, plot_summary_png)

    return out_dir, df, df_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yml")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=None, help="未指定なら0.2")
    # 閾値はユーザ要望に合わせて 0,0.1,...,1 をデフォルトにする
    p.add_argument("--points", type=int, default=11)
    p.add_argument("--min_th", type=float, default=0.0)
    p.add_argument("--max_th", type=float, default=1.0)
    p.add_argument("--threshold_list", type=str, default=None,
                   help="しきい値をCSVで明示指定（例: '0.99,0.98,...,0.8'）。指定時はpoints/min/maxより優先。")
    p.add_argument("--threshold_start", type=float, default=None)
    p.add_argument("--threshold_end", type=float, default=None)
    p.add_argument("--threshold_step", type=float, default=None)
    p.add_argument("--util_mode", type=str, default="first_k", choices=["col", "first_k", "window"])
    p.add_argument("--util_k", type=int, default=20)
    p.add_argument("--out_root", type=str, default=None, help="未指定なら experiments/scenario_sweeps/<timestamp>/")
    args = p.parse_args()

    base_config = sweep.load_config(args.config)
    lam = float(args.lam) if args.lam is not None else float(base_config.get("param_job", {}).get("lam", 0.2))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or f"experiments/scenario_sweeps/{ts}_ep{args.episodes}_seed{args.seed}_lam{lam}"
    os.makedirs(out_root, exist_ok=True)

    all_rows = []
    index_rows = []

    for sc in scenario_grid():
        sc_dir = os.path.join(
            out_root,
            f"jobs{sc['nb_jobs']}_on{sc['n_on_premise_node']}_cl{sc['n_cloud_node']}",
        )
        os.makedirs(sc_dir, exist_ok=True)

        # thresholds（明示指定があればそれを使う）
        if args.threshold_list:
            thresholds = sweep.explicit_thresholds(args.threshold_list)
        elif (args.threshold_start is not None) and (args.threshold_end is not None) and (args.threshold_step is not None):
            thresholds = sweep.range_thresholds(args.threshold_start, args.threshold_end, args.threshold_step, inclusive=True)
        else:
            thresholds = sweep.linspace_thresholds(int(args.points), float(args.min_th), float(args.max_th))

        _, df, _ = run_one_scenario(
            base_config,
            sc,
            sc_dir,
            episodes=int(args.episodes),
            seed=int(args.seed),
            lam=lam,
            thresholds=thresholds,
            util_mode=str(args.util_mode),
            util_k=int(args.util_k),
        )

        all_rows.append(df)
        index_rows.append(
            {
                "scenario_dir": sc_dir,
                "nb_jobs": sc["nb_jobs"],
                "n_on_premise_node": sc["n_on_premise_node"],
                "n_cloud_node": sc["n_cloud_node"],
            }
        )

    # 全シナリオ結合（比較しやすいように）
    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_csv = os.path.join(out_root, "per_episode_all_scenarios.csv")
    df_all.to_csv(all_csv, index=False)

    index_csv = os.path.join(out_root, "scenarios_index.csv")
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)

    print("=== 完了（全シナリオ） ===")
    print(f"- out_root : {out_root}")
    print(f"- all points(csv): {all_csv}")
    print(f"- index(csv)     : {index_csv}")


if __name__ == "__main__":
    main()


