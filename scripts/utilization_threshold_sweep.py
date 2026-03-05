#!/usr/bin/env python3
"""
オンプレミス利用率しきい値でクラウドへオフロードする単純ルールをsweepして、
待ち時間とコストのトレードオフを実験・可視化するスクリプト。

要件:
- AI（学習/推論）は使わない
- Cで書かれたEnv（scheduling_env_core）を呼ぶ（SchedulingEnvCacheOptimizedを使用）
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# プロジェクトルートをsys.pathへ追加（`python scripts/...` 実行でも `import src...` できるように）
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.job_gen.job_generator import JobGenerator
from src.agents.utilization_threshold_agent import UtilizationThresholdPolicy
from src.utils.font_setup import setup_japanese_font

# C拡張版Env
try:
    from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
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


def linspace_thresholds(n_points: int, min_th: float, max_th: float) -> List[float]:
    if n_points <= 1:
        return [float(min_th)]
    return [float(x) for x in np.linspace(min_th, max_th, n_points)]

def explicit_thresholds(values_csv: str) -> List[float]:
    """
    "0.99,0.98,0.8" のようなCSV文字列をしきい値リストに変換。
    """
    parts = [p.strip() for p in values_csv.split(",") if p.strip() != ""]
    ths = [float(p) for p in parts]
    for t in ths:
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"threshold must be in [0,1], got {t}")
    return ths

def range_thresholds(start: float, end: float, step: float, inclusive: bool = True) -> List[float]:
    """
    startからendまでstep刻みのしきい値を生成（降順も可）。
    例: start=0.99, end=0.80, step=0.01 -> 0.99,0.98,...,0.80
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    start = float(start)
    end = float(end)
    step = float(step)

    if start == end:
        return [start]

    desc = start > end
    if desc:
        # 0.99 -> 0.80
        n = int(np.floor((start - end) / step))
        vals = [start - i * step for i in range(n + 1)]
        if inclusive and (vals[-1] - end) > 1e-12:
            # 端がズレた場合にendを追加
            if vals[-1] > end:
                vals.append(end)
        # 端の丸め
        vals = [float(np.clip(v, 0.0, 1.0)) for v in vals if v >= end - 1e-12]
    else:
        # 0.80 -> 0.99
        n = int(np.floor((end - start) / step))
        vals = [start + i * step for i in range(n + 1)]
        if inclusive and (end - vals[-1]) > 1e-12:
            if vals[-1] < end:
                vals.append(end)
        vals = [float(np.clip(v, 0.0, 1.0)) for v in vals if v <= end + 1e-12]
    # 重複除去
    uniq = []
    for v in vals:
        if not uniq or abs(v - uniq[-1]) > 1e-12:
            uniq.append(v)
    return uniq

def quantile_thresholds(util_samples: np.ndarray, n_points: int) -> List[float]:
    """
    利用率サンプルの分位点からしきい値を作る。
    端（0%/100%）は極端解になりやすいので、少し内側を取る。
    """
    util_samples = np.asarray(util_samples, dtype=float)
    util_samples = util_samples[np.isfinite(util_samples)]
    if util_samples.size == 0:
        return linspace_thresholds(n_points, 0.0, 1.0)

    if n_points <= 1:
        return [float(np.median(util_samples))]

    # 0.05〜0.95 の範囲で等間隔
    qs = np.linspace(0.05, 0.95, n_points)
    ths = np.quantile(util_samples, qs).astype(float).tolist()

    # 重複を避ける（量子化されたutilで同値が出ることがある）
    uniq = []
    for t in ths:
        if not uniq or abs(t - uniq[-1]) > 1e-6:
            uniq.append(float(t))
    return uniq

def collect_util_samples(env, util_mode: str, util_k: int, episodes: int) -> np.ndarray:
    """
    しきい値選びのために利用率の分布をざっくり集める（pilot）。
    両端の挙動（オンプレ固定/クラウド固定）で収集し、混ぜる。
    """
    samples: List[float] = []

    # on-prem重視（threshold=1.0でほぼオンプレ）
    pol_a = UtilizationThresholdPolicy(
        threshold=1.0,
        utilization_mode=util_mode,
        utilization_k=util_k,
    )
    # cloud重視（threshold=0.0でほぼクラウド）
    pol_b = UtilizationThresholdPolicy(
        threshold=0.0,
        utilization_mode=util_mode,
        utilization_k=util_k,
    )

    for pol in (pol_a, pol_b):
        for ep in range(episodes):
            env.episode = ep
            pol.reset_stats()
            _ = env.reset()
            done = False
            while not done:
                action_raw, util = pol.select_action(env)
                samples.append(float(util))
                _, _, _, _, done = env.step(action_raw)

    return np.asarray(samples, dtype=float)


def run_one_episode(env, policy: UtilizationThresholdPolicy, episode_idx: int) -> Dict:
    env.episode = episode_idx
    policy.reset_stats()
    _ = env.reset()

    total_wait = 0.0
    total_cost = 0.0
    n_scheduled = 0
    util_sum = 0.0

    done = False
    while not done:
        action_raw, util = policy.select_action(env)
        util_sum += util

        _, rewards, scheduled, wt_step, done = env.step(action_raw)
        # rewards: [-time_reward_new, -cost]
        # time_reward_new は find_allocation_position の waiting_time（開始時刻まで）に近い値なので、ここではそれを待ち時間として扱う
        wait = float(-rewards[0])
        cost = float(-rewards[1])
        total_wait += wait
        total_cost += cost
        if scheduled:
            n_scheduled += 1

    # 履歴の最終化（calc_objective_values内で参照するため）
    env.finalize_window_history()

    avg_wait = total_wait / n_scheduled if n_scheduled > 0 else 0.0
    avg_util = util_sum / max(1, policy.stats["n_decisions"])

    return {
        "episode": episode_idx,
        "n_scheduled": n_scheduled,
        "total_wait": total_wait,
        "avg_wait": avg_wait,
        "total_cost": total_cost,
        "avg_onprem_util": avg_util,
        "n_onprem": policy.stats["n_onprem"],
        "n_cloud": policy.stats["n_cloud"],
        "n_forced_onprem": policy.stats["n_forced_onprem"],
    }


def plot_summary(df_summary: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    # 日本語のみのラベルを保証するため、日本語フォントが無ければエラーにする
    setup_japanese_font(required=True)

    plt.figure(figsize=(10, 7))
    x = df_summary["mean_cost"].values
    y = df_summary["mean_avg_wait"].values
    th = df_summary["threshold"].values

    plt.scatter(x, y, s=80)
    for xi, yi, thi in zip(x, y, th):
        plt.annotate(f"{thi:.2f}", (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)

    plt.xlabel("平均コスト（クラウド利用面積の合計）")
    plt.ylabel("平均待ち時間（開始時刻までの待ち）")
    plt.title("利用率しきい値 sweep：待ち時間 vs コスト")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_all_points(df_per_episode: pd.DataFrame, out_png: str) -> None:
    """
    しきい値×エピソードで出た全点（全試行）をプロットする。
    """
    import matplotlib.pyplot as plt

    setup_japanese_font(required=True)

    # 1行=1エピソードの点
    x = df_per_episode["total_cost"].to_numpy(dtype=float)
    y = df_per_episode["avg_wait"].to_numpy(dtype=float)
    c = df_per_episode["threshold"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=60, alpha=0.75, edgecolors="none")
    cb = plt.colorbar(sc)
    cb.set_label("しきい値")

    plt.xlabel("コスト（クラウド利用面積の合計）")
    plt.ylabel("待ち時間（開始時刻までの待ち：エピソード平均）")
    plt.title("利用率しきい値 sweep：全点（全エピソード）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yml")
    parser.add_argument("--nb_jobs", type=int, default=32, help="ジョブ数（JobGeneratorへ渡す）")
    parser.add_argument("--episodes", type=int, default=50, help="各しきい値で回すエピソード数")
    parser.add_argument("--seed", type=int, default=0, help="乱数seed（ジョブ生成用）")
    parser.add_argument("--lam", type=float, default=None, help="ポアソン到着率（未指定なら0.2 or config値）")
    parser.add_argument("--points", type=int, default=10, help="しきい値のサンプル点数（等間隔）")
    parser.add_argument("--min_th", type=float, default=0.0)
    parser.add_argument("--max_th", type=float, default=1.0)
    parser.add_argument("--threshold_sampling", type=str, default="linspace", choices=["linspace", "quantile"],
                        help="しきい値の取り方: linspace（等間隔） / quantile（利用率分布の分位点）")
    parser.add_argument("--threshold_list", type=str, default=None,
                        help="しきい値をCSVで明示指定（例: '0.99,0.98,...,0.8'）。指定時は他のsampling設定より優先。")
    parser.add_argument("--threshold_start", type=float, default=None,
                        help="しきい値レンジ指定（開始）。例: 0.99")
    parser.add_argument("--threshold_end", type=float, default=None,
                        help="しきい値レンジ指定（終了）。例: 0.80")
    parser.add_argument("--threshold_step", type=float, default=None,
                        help="しきい値レンジ指定（刻み）。例: 0.01（開始>終了なら降順生成）")
    parser.add_argument("--auto_range", action="store_true",
                        help="（linspace時）利用率の分布から[下位5%,上位95%]を推定し、その範囲を等間隔に10点サンプリングする（両端だけになるのを避けやすい）")
    parser.add_argument("--util_mode", type=str, default="first_k", choices=["col", "first_k", "window"],
                        help="利用率の定義: col=特定列 / first_k=左端K列平均 / window=全ウィンドウ平均")
    parser.add_argument("--util_k", type=int, default=10, help="util_mode=first_k のときのK列")
    parser.add_argument("--out_dir", type=str, default=None, help="出力先ディレクトリ（未指定なら experiments/...）")
    args = parser.parse_args()

    config = load_config(args.config)
    lam = args.lam
    if lam is None:
        lam = float(config.get("param_job", {}).get("lam", 0.2))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"experiments/util_threshold_sweep/{ts}_jobs{args.nb_jobs}_ep{args.episodes}_seed{args.seed}"
    out_dir = str(Path(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # ジョブセットは全しきい値で共通にする（公平比較）
    jobs_set = generate_jobs_set(config, seed=args.seed, nb_jobs=args.nb_jobs, nb_episodes=args.episodes, lam=lam)
    env = build_env(config, jobs_set)

    # しきい値の明示指定があれば最優先
    if args.threshold_list:
        thresholds = explicit_thresholds(args.threshold_list)
        print(f"[thresholds] sampling=explicit_list, points={len(thresholds)}")
        print(f"[thresholds] values={thresholds}")
    elif (args.threshold_start is not None) and (args.threshold_end is not None) and (args.threshold_step is not None):
        thresholds = range_thresholds(args.threshold_start, args.threshold_end, args.threshold_step, inclusive=True)
        print(f"[thresholds] sampling=range, points={len(thresholds)}, start={args.threshold_start}, end={args.threshold_end}, step={args.threshold_step}")
        print(f"[thresholds] values={thresholds}")
    elif args.threshold_sampling == "quantile":
        util_samples = collect_util_samples(env, util_mode=args.util_mode, util_k=args.util_k, episodes=min(3, args.episodes))
        thresholds = quantile_thresholds(util_samples, args.points)
        print(f"[thresholds] sampling=quantile, points={len(thresholds)}")
        print(f"[thresholds] values={thresholds}")
    else:
        if args.auto_range:
            util_samples = collect_util_samples(env, util_mode=args.util_mode, util_k=args.util_k, episodes=min(3, args.episodes))
            util_samples = util_samples[np.isfinite(util_samples)]
            if util_samples.size > 0:
                lo = float(np.quantile(util_samples, 0.05))
                hi = float(np.quantile(util_samples, 0.95))
                # 少しだけマージン
                margin = 0.02
                min_th = max(0.0, lo - margin)
                max_th = min(1.0, hi + margin)
            else:
                min_th, max_th = args.min_th, args.max_th
        else:
            min_th, max_th = args.min_th, args.max_th
        thresholds = linspace_thresholds(args.points, min_th, max_th)
        print(f"[thresholds] sampling=linspace, points={len(thresholds)}, range=[{min_th:.3f},{max_th:.3f}] auto_range={args.auto_range}")

    rows = []
    for th in thresholds:
        policy = UtilizationThresholdPolicy(
            threshold=float(th),
            utilization_mode=args.util_mode,
            utilization_k=args.util_k,
        )
        for ep in range(args.episodes):
            r = run_one_episode(env, policy, episode_idx=ep)
            r.update(
                {
                    "threshold": float(th),
                    "seed": int(args.seed),
                    "nb_jobs": int(args.nb_jobs),
                    "lam": float(lam),
                }
            )
            rows.append(r)

    df = pd.DataFrame(rows)
    per_episode_csv = os.path.join(out_dir, "per_episode.csv")
    df.to_csv(per_episode_csv, index=False)

    # しきい値ごとに集計
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
    plot_all_points(df, plot_all_png)
    plot_summary(df_summary, plot_summary_png)

    print("=== 完了 ===")
    print(f"- per-episode: {per_episode_csv}")
    print(f"- summary    : {summary_csv}")
    print(f"- plot(all)  : {plot_all_png}")
    print(f"- plot(mean) : {plot_summary_png}")


if __name__ == "__main__":
    main()


