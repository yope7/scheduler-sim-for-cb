#!/usr/bin/env python3
"""
学習済みPCNモデルで推論実験を行うスクリプト

- 既知ジョブ（学習時と同じseed）と未知ジョブ（異なるseed）の両方で評価
- 多様な(desired_return, desired_horizon)でPFを張る
- 学習は行わず、与えたモデルで推論のみ
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

# プロジェクトルートをパスに追加
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.agents.nsga2_agent import NSGA2Agent
from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
from src.utils.job_gen.job_generator import JobGenerator

# distributed_pcnの設定を参照
USE_ENHANCED_MODEL = False
N_JOBS = 32
DESIRED_HORIZON = 32  # 残りステップ数 = ジョブ数


def load_config():
    with open("config/config.yml", "r") as f:
        return yaml.safe_load(f)


def create_env(config, job_seed: int, nb_jobs: int = N_JOBS):
    """ジョブセットを生成し環境を作成"""
    job_generator = JobGenerator(
        job_seed, 1,
        config["param_env"]["n_window"],
        config["param_env"]["n_on_premise_node"],
        config["param_env"]["n_cloud_node"],
        config, nb_jobs, 0.2, 0
    )
    jobs_set = job_generator.generate_jobs_set()

    env = SchedulingEnvCacheOptimized(
        np.inf,
        config["param_env"]["n_window"],
        config["param_env"]["n_on_premise_node"],
        config["param_env"]["n_cloud_node"],
        config["param_env"]["n_job_queue_obs"],
        config["param_env"]["n_job_queue_bck"],
        config["param_agent"]["weight_wt"],
        config["param_agent"]["weight_cost"],
        config["param_env"]["penalty_not_allocate"],
        config["param_env"]["penalty_invalid_action"],
        jobs_set,
        None, flag=0
    )
    return env


def load_model(checkpoint_path: str, env, device: str = "cpu"):
    """モデルチェックポイントを読み込みPCNエージェントを構築"""
    state = th.load(checkpoint_path, map_location=device)

    model_type = state.get("model_type", "DiscreteActionsDefaultModel")
    config = state.get("config", load_config())

    agent = PCN(
        env,
        device=device,
        state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1, 1, 1]),
        learning_rate=1e-3,
        batch_size=512,
        hidden_dim=512,
        project_name="temp",
        experiment_name="PCN",
        log=False,
        debug_mode=False,
        use_enhanced_model=(model_type == "EnhancedPCNModel"),
    )

    if "model_state_dict" in state:
        if agent.use_enhanced_model and hasattr(agent, "network"):
            agent.network.load_state_dict(state["model_state_dict"])
        else:
            agent.model.load_state_dict(state["model_state_dict"])
        print(f"モデルを読み込みました: {checkpoint_path}")
    else:
        raise ValueError(f"モデル状態が含まれていません: {checkpoint_path}")

    agent.model.eval()
    if hasattr(agent, "network"):
        agent.network.eval()
    return agent


# 広範囲グリッド用のデフォルト（cost, wt を幅広くカバー）
WIDE_COST_RANGE = (0, 300000)
WIDE_WT_RANGE = (10, 500)


def create_diverse_commands(
    n_points: int,
    reference_pf_path: str = None,
    cost_range: tuple = None,
    wt_range: tuple = None,
    include_endpoints: bool = True,
    use_wide_grid: bool = False,
) -> tuple:
    """
    多様な(desired_return, desired_horizon)ペアを生成

    PCNへの入力:
      - desired_return: 報酬空間 [reward_dim0, reward_dim1] = [-wt, -cost]
      - desired_horizon: 残りステップ数（= ジョブ数）

    実数値 [cost, wt] -> desired_return = (-wt, -cost)
    """
    cost_range = cost_range or WIDE_COST_RANGE
    wt_range = wt_range or WIDE_WT_RANGE

    # 広範囲グリッド: 参照を無視し、広い範囲で多数の点を生成
    # コストが小さい領域を重点的に（累乗分布で低cost側に点を密集）
    if use_wide_grid:
        cost_min, cost_max = cost_range
        wt_min, wt_max = wt_range
        n_c = max(20, int(np.sqrt(n_points * 1.5)))  # cost軸を多めに
        n_w = max(10, (n_points + n_c - 1) // n_c)
        # cost: 低cost側を重点的に（0-100kに50%、100k-300kに50%の点を配置）
        n_c_low = max(1, n_c // 2)   # 低cost側
        n_c_high = n_c - n_c_low
        cost_mid = cost_min + (cost_max - cost_min) * 0.33  # 100k付近で分割
        costs_low = np.linspace(cost_min, cost_mid, n_c_low)
        costs_high = np.linspace(cost_mid, cost_max, n_c_high + 1)[1:]  # 重複回避
        costs = np.concatenate([costs_low, costs_high])
        wts = np.linspace(wt_min, wt_max, n_w)
        cc, ww = np.meshgrid(costs, wts)
        values = np.column_stack([cc.ravel(), ww.ravel()])[:n_points]
        desired_returns = np.column_stack([-values[:, 1], -values[:, 0]]).astype(np.float32)
        horizons = np.full(len(desired_returns), DESIRED_HORIZON, dtype=np.float32)
        return desired_returns, horizons

    values = None
    if reference_pf_path and Path(reference_pf_path).exists():
        raw = []
        with open(reference_pf_path) as f:
            for line in f:
                if line.strip().startswith("解") and ":" in line:
                    parts = line.split(":")[1].strip()
                    try:
                        nums = [float(x.strip()) for x in parts.strip("[]").split(",")]
                        if len(nums) >= 2:
                            raw.append([nums[0], nums[1]])
                    except ValueError:
                        pass
        if raw:
            values = np.unique(np.array(raw), axis=0)

    if values is not None and len(values) > 0:
        if include_endpoints:
            cmin, cmax = values[:, 0].min(), values[:, 0].max()
            wmin, wmax = values[:, 1].min(), values[:, 1].max()
            endpoints = np.array([[cmin, wmax], [cmax, wmin]])
            values = np.unique(np.vstack([endpoints, values]), axis=0)

        if len(values) >= n_points:
            indices = np.linspace(0, len(values) - 1, n_points, dtype=int)
            values = values[indices]
        else:
            n_ref = len(values)
            order = np.lexsort((values[:, 1], values[:, 0]))
            sorted_v = values[order]
            t_old = np.linspace(0, 1, n_ref)
            t_new = np.linspace(0, 1, n_points)
            cost_interp = np.interp(t_new, t_old, sorted_v[:, 0])
            wt_interp = np.interp(t_new, t_old, sorted_v[:, 1])
            values = np.column_stack([cost_interp, wt_interp])
        desired_returns = np.column_stack([-values[:, 1], -values[:, 0]]).astype(np.float32)
    else:
        cost_min, cost_max = cost_range
        wt_min, wt_max = wt_range
        n_c = max(2, int(np.sqrt(n_points)))
        n_w = max(2, (n_points + n_c - 1) // n_c)
        costs = np.linspace(cost_min, cost_max, n_c)
        wts = np.linspace(wt_min, wt_max, n_w)
        cc, ww = np.meshgrid(costs, wts)
        values = np.column_stack([cc.ravel(), ww.ravel()])[:n_points]
        desired_returns = np.column_stack([-values[:, 1], -values[:, 0]]).astype(np.float32)

    horizons = np.full(len(desired_returns), DESIRED_HORIZON, dtype=np.float32)
    return desired_returns, horizons


def run_inference(agent, env, desired_returns, desired_horizons, max_return=None):
    """与えられた(desired_return, desired_horizon)で推論を実行"""
    if max_return is None:
        max_return = np.full(2, 1000.0, dtype=np.float32)

    gamma = getattr(agent, "gamma", 0.99)
    e_returns = []
    e_values = []

    for i, (dr, dh) in enumerate(zip(desired_returns, desired_horizons)):
        transitions, _, _, _, map_fin, value = agent._run_episode(
            env, dr, dh, max_return, eval_mode=True
        )
        # 累積報酬を計算（evaluateと同様の後方累積）
        if len(transitions) > 0:
            rewards = [np.array(t.reward, dtype=np.float64, copy=True) for t in transitions]
            for j in reversed(range(len(rewards) - 1)):
                rewards[j] = rewards[j] + gamma * rewards[j + 1]
            cum_return = rewards[0]
        else:
            cum_return = np.zeros(2)
        e_returns.append(cum_return)
        e_values.append(value)  # [cost, avg_wt]

    return np.array(e_returns), np.array(e_values)


def extract_pareto_front(values):
    """実数値空間（cost, wt 最小化）でパレートフロントを抽出"""
    if len(values) == 0:
        return values
    inds = get_non_dominated_inds_minimize(np.array(values, dtype=np.float64))
    return values[inds]


def run_nsga2(config, job_seed: int, nb_jobs: int, num_generations: int = 100, pop_size: int = 100, verbose: bool = False):
    """NSGA-IIを実行してパレートフロントを取得（目的関数: cost, avg_waiting_time）"""
    env = create_env(config, job_seed, nb_jobs)
    agent = NSGA2Agent(
        pop_size=pop_size,
        num_generations=num_generations,
        crossover_prob=0.7,
        mutation_prob=0.1,
        eliminate_duplicates=True,
    )
    result = agent.run(env, nb_jobs, verbose=verbose, n_jobs=-1)
    objectives = result["objectives"]
    if len(objectives) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    pf = extract_pareto_front(objectives)
    return objectives, pf


def main():
    parser = argparse.ArgumentParser(description="PCN推論実験: 既知/未知ジョブでPF評価")
    parser.add_argument(
        "--model",
        type=str,
        default="execution_20260211_182837/final/final_model.pth",
        help="モデルチェックポイントのパス",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=200,
        help="多様な条件点の数（広範囲グリッド時は大量に使用）",
    )
    parser.add_argument(
        "--no_wide_grid",
        action="store_true",
        help="広範囲グリッドを無効化し参照PFを使用（デフォルトは広範囲グリッド cost 0~300k, wt 10~500）",
    )
    parser.add_argument(
        "--ref_pf",
        type=str,
        default="execution_20260212_003852/iteration_100/pareto_front_details_current_20260212_010353.txt",
        help="参照PFファイル（多様な点の生成に使用）",
    )
    parser.add_argument(
        "--unknown_seed",
        type=int,
        default=42,
        help="未知ジョブ用の乱数シード",
    )
    parser.add_argument(
        "--known_seed",
        type=int,
        default=0,
        help="既知ジョブ用の乱数シード（学習時と同じ）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="出力ディレクトリ（未指定時は実行日時で自動作成）",
    )
    parser.add_argument(
        "--nsga2",
        action="store_true",
        help="NSGA-IIを実行する（seed42で未知ジョブ最適化）",
    )
    parser.add_argument(
        "--nsga2_generations",
        type=int,
        default=100,
        help="NSGA-IIの世代数（--nsga2時のみ）",
    )
    args = parser.parse_args()

    config = load_config()
    device = "cuda" if th.cuda.is_available() else "cpu"

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"pcn_inference_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"出力ディレクトリ: {out_dir}")

    # 既知ジョブ用環境
    env_known = create_env(config, args.known_seed)
    agent = load_model(args.model, env_known, device)

    # 多様な条件点を生成（デフォルト: 広範囲グリッド、--no_wide_gridで参照PF使用）
    use_wide_grid = not args.no_wide_grid
    desired_returns, desired_horizons = create_diverse_commands(
        args.n_points,
        reference_pf_path=args.ref_pf,
        use_wide_grid=use_wide_grid,
    )
    print(f"条件点数: {len(desired_returns)}, desired_horizon={DESIRED_HORIZON}")
    # PCN入力の範囲を表示（desired_return = [-wt, -cost]、報酬空間）
    print(f"PCN入力 desired_return (報酬空間[-wt,-cost]): "
          f"dim0(wt)={desired_returns[:, 0].min():.0f}~{desired_returns[:, 0].max():.0f}, "
          f"dim1(cost)={desired_returns[:, 1].min():.0f}~{desired_returns[:, 1].max():.0f}")

    max_return = np.full(2, 1000.0, dtype=np.float32)

    # === 既知ジョブで推論 ===
    print("\n=== 既知ジョブ（seed={}）で推論 ===".format(args.known_seed))
    e_returns_known, e_values_known = run_inference(
        agent, env_known, desired_returns, desired_horizons, max_return
    )
    pf_known = extract_pareto_front(e_values_known)
    print(f"既知ジョブ: 全解数={len(e_values_known)}, PF解数={len(pf_known)}")

    # === 未知ジョブで推論 ===
    print("\n=== 未知ジョブ（seed={}）で推論 ===".format(args.unknown_seed))
    env_unknown = create_env(config, args.unknown_seed)
    e_returns_unknown, e_values_unknown = run_inference(
        agent, env_unknown, desired_returns, desired_horizons, max_return
    )
    pf_unknown = extract_pareto_front(e_values_unknown)
    print(f"未知ジョブ: 全解数={len(e_values_unknown)}, PF解数={len(pf_unknown)}")

    # === NSGA-II（オプション指定時のみ）===
    pf_nsga2 = np.array([]).reshape(0, 2)
    all_nsga2 = np.array([]).reshape(0, 2)
    if args.nsga2:
        print(f"\n=== NSGA-II（seed={args.unknown_seed}, {args.nsga2_generations}世代）===")
        all_nsga2, pf_nsga2 = run_nsga2(
            config, args.unknown_seed, N_JOBS,
            num_generations=args.nsga2_generations, verbose=False
        )
        print(f"NSGA-II: 全解数={len(all_nsga2)}, PF解数={len(pf_nsga2)}")

    # === 結果保存 ===
    np.savez(
        os.path.join(out_dir, "inference_results.npz"),
        known_all=e_values_known,
        known_pf=pf_known,
        unknown_all=e_values_unknown,
        unknown_pf=pf_unknown,
        nsga2_all=all_nsga2,
        nsga2_pf=pf_nsga2,
        desired_returns=desired_returns,
        desired_horizons=desired_horizons,
    )

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("PCN推論実験結果\n")
        f.write(f"モデル: {args.model}\n")
        f.write(f"既知ジョブ seed={args.known_seed}: PF解数={len(pf_known)}\n")
        f.write(f"未知ジョブ seed={args.unknown_seed}: PF解数={len(pf_unknown)}\n")
        if len(pf_nsga2) > 0:
            f.write(f"NSGA-II seed={args.unknown_seed} ({args.nsga2_generations}世代): PF解数={len(pf_nsga2)}\n")
        f.write("\n=== 既知ジョブ PF ===\n")
        for i, v in enumerate(pf_known):
            f.write(f"解{i+1}: [cost={v[0]:.2f}, wt={v[1]:.2f}]\n")
        f.write("\n=== 未知ジョブ PF ===\n")
        for i, v in enumerate(pf_unknown):
            f.write(f"解{i+1}: [cost={v[0]:.2f}, wt={v[1]:.2f}]\n")
        if len(pf_nsga2) > 0:
            f.write("\n=== NSGA-II PF (seed42) ===\n")
            for i, v in enumerate(pf_nsga2):
                f.write(f"解{i+1}: [cost={v[0]:.2f}, wt={v[1]:.2f}]\n")

    # === 可視化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 既知ジョブ
    ax = axes[0]
    ax.scatter(e_values_known[:, 0], e_values_known[:, 1], c="steelblue", alpha=0.6, s=40, label="全解")
    ax.scatter(pf_known[:, 0], pf_known[:, 1], c="red", s=80, marker="*", label="PF", zorder=5)
    if len(pf_known) > 1:
        order = np.lexsort((pf_known[:, 1], pf_known[:, 0]))
        sorted_pf = pf_known[order]
        ax.plot(sorted_pf[:, 0], sorted_pf[:, 1], "r-", alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Avg Waiting Time")
    ax.set_title(f"既知ジョブ (seed={args.known_seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 未知ジョブ
    ax = axes[1]
    ax.scatter(e_values_unknown[:, 0], e_values_unknown[:, 1], c="steelblue", alpha=0.6, s=40, label="全解")
    ax.scatter(pf_unknown[:, 0], pf_unknown[:, 1], c="red", s=80, marker="*", label="PF", zorder=5)
    if len(pf_unknown) > 1:
        order = np.lexsort((pf_unknown[:, 1], pf_unknown[:, 0]))
        sorted_pf = pf_unknown[order]
        ax.plot(sorted_pf[:, 0], sorted_pf[:, 1], "r-", alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Avg Waiting Time")
    ax.set_title(f"未知ジョブ (seed={args.unknown_seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_inference.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可視化を保存: {out_dir}/pareto_inference.png")

    # 比較プロット（両方のPFを重ねて表示）
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pf_known[:, 0], pf_known[:, 1], c="blue", s=100, marker="o", label=f"既知 (seed={args.known_seed})", alpha=0.8)
    ax.scatter(pf_unknown[:, 0], pf_unknown[:, 1], c="orange", s=100, marker="s", label=f"未知 (seed={args.unknown_seed})", alpha=0.8)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Avg Waiting Time")
    ax.set_title("PF比較: 既知 vs 未知ジョブ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"比較図を保存: {out_dir}/pareto_comparison.png")

    # PCN vs NSGA-II 比較（seed42の同一問題で最適化度合いを比較）
    if len(pf_nsga2) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pf_unknown[:, 0], pf_unknown[:, 1], c="red", s=80, marker="o", label="PCN (推論)", alpha=0.8)
        ax.scatter(pf_nsga2[:, 0], pf_nsga2[:, 1], c="green", s=80, marker="s", label=f"NSGA-II ({args.nsga2_generations}世代)", alpha=0.8)
        ax.set_xlabel("Cost")
        ax.set_ylabel("Avg Waiting Time")
        ax.set_title(f"最適化比較: PCN vs NSGA-II (seed={args.unknown_seed})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pareto_pcn_vs_nsga2.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"PCN vs NSGA-II 比較図を保存: {out_dir}/pareto_pcn_vs_nsga2.png")

    print("\n実験完了.")


if __name__ == "__main__":
    main()
