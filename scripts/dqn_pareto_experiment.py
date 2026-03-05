#!/usr/bin/env python3
"""
DQNパレート実験: seed0で32モデルを学習し、seed42で推論してPFを可視化

使い方:
  python scripts/dqn_pareto_experiment.py --mode dqn
  python scripts/dqn_pareto_experiment.py --mode dqn --n_models 32 --train_episodes 500
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch as th
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from src.agents.dqn_agent import DQNAgent
from src.agents.pcn_agent import get_non_dominated_inds_minimize
from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
from src.utils.job_gen.job_generator import JobGenerator

N_JOBS = 32
TRAIN_SEED = 0
INFER_SEED = 42


def load_config():
    with open("config/config.yml", "r") as f:
        return yaml.safe_load(f)


def create_env(config, job_seed: int, nb_jobs: int = N_JOBS):
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


def train_dqn(config, job_seed: int, w_idx: int, w_wt: float, w_cost: float,
              n_episodes: int, model_dir: str) -> str:
    """1つの重みでDQNを学習し保存"""
    env = create_env(config, job_seed)
    state_dim = env.observation_space.shape[0]

    agent = DQNAgent(
        env,
        device="auto",
        state_dim=state_dim,
        learning_rate=1e-2,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        buffer_size=2000,
        batch_size=512,
        hidden_dim=512,
        target_update=10,
        weight_cost=w_cost,
        weight_id=f"weight_{w_idx}",
    )
    agent.train(n_episodes)

    model_path = os.path.join(model_dir, f"dqn_weight_{w_idx:02d}.pt")
    os.makedirs(model_dir, exist_ok=True)
    th.save({
        "policy_state_dict": agent.policy_net.state_dict(),
        "state_dim": state_dim,
        "action_dim": agent.action_dim,
        "hidden_dim": agent.hidden_dim,
        "w_wt": w_wt,
        "w_cost": w_cost,
    }, model_path)
    return model_path


def run_dqn_inference(env, agent, num_episodes: int = 1) -> tuple:
    """DQNで推論実行し(cost, avg_wt)を返す"""
    agent.epsilon = 0.0  # 推論時はgreedy
    values = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, scheduled, wt_step, done = env.step(action)
            if done:
                env.finalize_window_history()
        cost, _, avg_wt = env.calc_objective_values()
        values.append([cost, avg_wt])
    return np.mean(values, axis=0)


def extract_pareto(values):
    if len(values) == 0:
        return values
    inds = get_non_dominated_inds_minimize(np.array(values, dtype=np.float64))
    return values[inds]


def main():
    parser = argparse.ArgumentParser(description="DQNパレート実験: seed0学習→seed42推論")
    parser.add_argument("--mode", type=str, default="dqn", help="実行モード")
    parser.add_argument("--n_models", type=int, default=32, help="学習するDQNモデル数（重み数）")
    parser.add_argument("--n_workers", type=int, default=None, help="並列ワーカー数（未指定時はn_modelsと同数）")
    parser.add_argument("--train_episodes", type=int, default=200, help="学習エピソード数")
    parser.add_argument("--model_dir", type=str, default="dqn_models_32", help="モデル保存先")
    parser.add_argument("--out_dir", type=str, default=None, help="出力ディレクトリ")
    parser.add_argument("--skip_train", action="store_true", help="学習をスキップ（既存モデルで推論のみ）")
    args = parser.parse_args()

    config = load_config()
    out_dir = args.out_dir or f"dqn_pareto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"出力ディレクトリ: {out_dir}")

    # === 学習: seed0で32モデル（並列）===
    if not args.skip_train:
        n_workers = args.n_workers or min(args.n_models, os.cpu_count() or 32)
        print(f"\n=== DQN学習 (seed={TRAIN_SEED}, {args.n_models}モデル, {args.train_episodes}ep, {n_workers}並列) ===")
        weights = np.linspace(0, 1, args.n_models)
        os.makedirs(args.model_dir, exist_ok=True)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for w_idx, w_wt in enumerate(weights):
                w_cost = 1 - w_wt
                future = executor.submit(
                    train_dqn, config, TRAIN_SEED, w_idx, w_wt, w_cost,
                    args.train_episodes, args.model_dir
                )
                futures[future] = (w_idx, w_wt, w_cost)

            completed = 0
            for future in as_completed(futures):
                w_idx, w_wt, w_cost = futures[future]
                try:
                    path = future.result()
                    completed += 1
                    print(f"完了 {completed}/{args.n_models}: w_idx={w_idx}, w_wt={w_wt:.3f} -> {path}")
                except Exception as e:
                    completed += 1
                    print(f"エラー w_idx={w_idx}: {e}")

    # === 推論: seed42 ===
    print(f"\n=== DQN推論 (seed={INFER_SEED}) ===")
    env_infer = create_env(config, INFER_SEED)
    state_dim = env_infer.observation_space.shape[0]
    action_dim = env_infer.action_space.n

    all_values = []
    for w_idx in range(args.n_models):
        model_path = os.path.join(args.model_dir, f"dqn_weight_{w_idx:02d}.pt")
        if not os.path.exists(model_path):
            print(f"スキップ: {model_path} が見つかりません")
            continue

        state = th.load(model_path, map_location="cpu")
        agent = DQNAgent(
            env_infer,
            device="cpu",
            state_dim=state["state_dim"],
            hidden_dim=state.get("hidden_dim", 512),
        )
        agent.policy_net.load_state_dict(state["policy_state_dict"])
        agent.epsilon = 0.0

        cost, avg_wt = run_dqn_inference(env_infer, agent, num_episodes=1)
        all_values.append([cost, avg_wt])
        if (w_idx + 1) % 8 == 0:
            print(f"  推論完了 {w_idx+1}/{args.n_models}")

    all_values = np.array(all_values)
    pf = extract_pareto(all_values)

    # 保存
    np.savez(os.path.join(out_dir, "dqn_results.npz"), all=all_values, pf=pf)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"DQN Pareto実験\n")
        f.write(f"学習: seed={TRAIN_SEED}, {args.n_models}モデル, {args.train_episodes}ep\n")
        f.write(f"推論: seed={INFER_SEED}\n")
        f.write(f"全解数: {len(all_values)}, PF解数: {len(pf)}\n")
        for i, v in enumerate(pf):
            f.write(f"解{i+1}: [cost={v[0]:.2f}, wt={v[1]:.2f}]\n")

    # === PF可視化 ===
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_values[:, 0], all_values[:, 1], c="steelblue", alpha=0.6, s=60, label="All")
    ax.scatter(pf[:, 0], pf[:, 1], c="red", s=100, marker="*", label="Pareto Front", zorder=5)
    if len(pf) > 1:
        order = np.lexsort((pf[:, 1], pf[:, 0]))
        sorted_pf = pf[order]
        ax.plot(sorted_pf[:, 0], sorted_pf[:, 1], "r-", alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Avg Waiting Time")
    ax.set_title(f"DQN Pareto Front (seed{TRAIN_SEED} train, seed{INFER_SEED} inference)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dqn_pareto.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nDQN PF可視化保存: {out_dir}/dqn_pareto.png")
    print(f"PF解数: {len(pf)}")


if __name__ == "__main__":
    main()
