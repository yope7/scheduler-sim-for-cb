#!/usr/bin/env python3
"""固定観測で desired_return / desired_horizon をスイープし、方策の感度を可視化する。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn.functional as F
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from src.agents.pcn_agent import PCN
from src.envs.scheduling_variants.event_c_env import SchedulingEnvEventObs
from src.utils.job_gen.job_generator import JobGenerator

USE_ENHANCED_MODEL = False


def load_config():
    with open("config/config.yml", "r") as f:
        return yaml.safe_load(f)


def create_event_env(config, job_seed: int = 0):
    n_jobs = config["param_env"].get("n_jobs", 32)
    jg = JobGenerator(
        job_seed,
        1,
        config["param_env"]["n_window"],
        config["param_env"]["n_on_premise_node"],
        config["param_env"]["n_cloud_node"],
        config,
        n_jobs,
        0.2,
        0,
    )
    jobs_set = jg.generate_jobs_set()
    return SchedulingEnvEventObs(
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
        None,
        flag=0,
    )


def load_agent(checkpoint_path: str, env, device: str = "cpu"):
    state = th.load(checkpoint_path, map_location=device, weights_only=False)
    config = state.get("config", load_config())
    model_type = state.get("model_type", "DiscreteActionsDefaultModel")
    agent = PCN(
        env,
        device=device,
        state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1, 1, 1]),
        learning_rate=1e-3,
        batch_size=512,
        hidden_dim=512,
        project_name="sweep",
        experiment_name="desired_return_sweep",
        log=False,
        debug_mode=False,
        use_enhanced_model=(model_type == "EnhancedPCNModel"),
    )
    sd = state["model_state_dict"]
    if agent.use_enhanced_model and hasattr(agent, "network"):
        agent.network.load_state_dict(sd, strict=False)
    else:
        agent.model.load_state_dict(sd, strict=False)
    agent.model.eval()
    if hasattr(agent, "network"):
        agent.network.eval()
    return agent, config


def policy_step0(agent: PCN, obs: np.ndarray, desired_return, desired_horizon: float):
    logits = agent._policy_logits_1d(obs, desired_return, np.float32(desired_horizon))
    logits = np.asarray(logits, dtype=np.float64).ravel()
    probs = F.softmax(th.tensor(logits), dim=-1).numpy()
    return logits, int(np.argmax(logits)), float(probs[1] if probs.size > 1 else np.nan)


def run_return_grid(agent, obs, out_dir: Path, horizons=(1024.0,)):
    # reward[0]=待ち時間累積, reward[1]=コスト累積（負の目標）
    wt_vals = np.linspace(0.0, -3.0e6, 13)
    cost_vals = np.linspace(0.0, -2.0e7, 13)
    rows = []
    for hz in horizons:
        for r0 in wt_vals:
            for r1 in cost_vals:
                dr = np.array([r0, r1], dtype=np.float32)
                logits, action, p1 = policy_step0(agent, obs, dr, hz)
                rows.append(
                    {
                        "desired_horizon": float(hz),
                        "return_wt": float(r0),
                        "return_cost": float(r1),
                        "logit0": float(logits[0]),
                        "logit1": float(logits[1]) if logits.size > 1 else None,
                        "prob_action1": p1,
                        "action": action,
                    }
                )
    import csv

    csv_path = out_dir / "return_grid_step0.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for hz in horizons:
        sub = [r for r in rows if r["desired_horizon"] == hz]
        Z = np.full((len(wt_vals), len(cost_vals)), np.nan)
        for r in sub:
            i = int(np.argmin(np.abs(wt_vals - r["return_wt"])))
            j = int(np.argmin(np.abs(cost_vals - r["return_cost"])))
            Z[i, j] = r["prob_action1"]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[cost_vals[0], cost_vals[-1], wt_vals[0], wt_vals[-1]],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("desired_return[1] (cost cumulative)")
        ax.set_ylabel("desired_return[0] (wait cumulative)")
        ax.set_title(f"P(action=1) at step0, horizon={int(hz)}")
        plt.colorbar(im, ax=ax, label="prob(action=1)")
        fig.tight_layout()
        fig.savefig(out_dir / f"prob_action1_heatmap_h{int(hz)}.png", dpi=150)
        plt.close(fig)

    uniq_logits = len({(r["logit0"], r["logit1"]) for r in rows if r["desired_horizon"] == horizons[0]})
    summary = {
        "grid_wt": [float(x) for x in wt_vals],
        "grid_cost": [float(x) for x in cost_vals],
        "horizons": [float(h) for h in horizons],
        "n_grid_points": len(rows),
        "n_unique_logits_step0_primary_horizon": uniq_logits,
        "prob_action1_min": float(min(r["prob_action1"] for r in rows)),
        "prob_action1_max": float(max(r["prob_action1"] for r in rows)),
        "prob_action1_std_primary": float(
            np.std([r["prob_action1"] for r in rows if r["desired_horizon"] == horizons[0]])
        ),
    }
    (out_dir / "return_grid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows, summary


def run_horizon_sweep(agent, obs, out_dir: Path, fixed_return):
    horizons = np.unique(
        np.concatenate(
            [
                np.array([1, 8, 16, 32, 64, 128, 256, 512, 768, 1024], dtype=np.float32),
                np.linspace(1, 1024, 25),
            ]
        )
    )
    rows = []
    dr = np.asarray(fixed_return, dtype=np.float32)
    for hz in horizons:
        logits, action, p1 = policy_step0(agent, obs, dr, float(hz))
        rows.append(
            {
                "desired_horizon": float(hz),
                "return_wt": float(dr[0]),
                "return_cost": float(dr[1]),
                "logit0": float(logits[0]),
                "logit1": float(logits[1]) if logits.size > 1 else None,
                "prob_action1": p1,
                "action": action,
            }
        )
    import csv

    csv_path = out_dir / "horizon_sweep_step0.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hz_arr = np.array([r["desired_horizon"] for r in rows])
    p1_arr = np.array([r["prob_action1"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hz_arr, p1_arr, "o-", ms=3)
    ax.set_xlabel("desired_horizon")
    ax.set_ylabel("P(action=1) at step0")
    ax.set_title(f"horizon sweep return=({dr[0]:.0f}, {dr[1]:.0f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "horizon_sweep_prob_action1.png", dpi=150)
    plt.close(fig)

    summary = {
        "fixed_return": [float(dr[0]), float(dr[1])],
        "n_unique_logits": len({(r["logit0"], r["logit1"]) for r in rows}),
        "prob_action1_std": float(np.std(p1_arr)),
    }
    (out_dir / "horizon_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="desired_return スイープ（固定 obs・step0）")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--job-seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
    os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

    ckpt = Path(args.checkpoint)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ckpt.parent / f"desired_return_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    env = create_event_env(config, job_seed=args.job_seed)
    agent, _ = load_agent(str(ckpt), env, device=args.device)
    obs = env.reset()

    meta = {
        "checkpoint": str(ckpt.resolve()),
        "obs_dim": int(obs.shape[0]),
        "job_seed": args.job_seed,
        "device": args.device,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _, grid_summary = run_return_grid(agent, obs, out_dir, horizons=(1024.0,))
    run_return_grid(agent, obs, out_dir, horizons=(512.0, 128.0, 32.0))

    # 評価 diag と同様の代表ターゲットで horizon スイープ
    for label, ret in [
        ("low_cost_only", [-1.8e7, 0.0]),
        ("mixed", [-1.1e7, -1.8e6]),
        ("mid", [-1.6e7, -5.9e5]),
    ]:
        sub = out_dir / f"horizon_sweep_{label}"
        sub.mkdir(exist_ok=True)
        run_horizon_sweep(agent, obs, sub, ret)

    print(json.dumps({"output_dir": str(out_dir), "grid_summary": grid_summary}, indent=2))


if __name__ == "__main__":
    main()
