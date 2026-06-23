#!/usr/bin/env python3
"""学習済みモデルの conditioning 指標（スイープ・c_emb 分散・バッチ内感度）を JSON 出力。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "desired_return_sweep",
    repo_root / "scripts" / "desired_return_sweep.py",
)
_drs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_drs)
create_event_env = _drs.create_event_env
load_agent = _drs.load_agent
policy_step0 = _drs.policy_step0


def measure_c_emb(agent, obs: np.ndarray, dr_grid, hz: float, device: str) -> dict:
    """固定 obs で desired_return を振ったときの c_emb 出力の分散。"""
    if agent.use_enhanced_model:
        return {"c_emb_std_mean": None, "note": "enhanced model skipped"}

    model = agent.model
    obs_t = th.tensor([obs], device=device, dtype=th.float32)
    c_list = []
    for r0 in dr_grid[0]:
        for r1 in dr_grid[1]:
            dr = np.array([r0, r1], dtype=np.float32)
            ret_t = th.tensor([dr], device=device, dtype=th.float32)
            hz_t = th.tensor([[hz]], device=device, dtype=th.float32)
            with th.no_grad():
                dr_in = th.clamp(ret_t, -1e9, 1e9)
                if hasattr(model, "desired_return_center"):
                    dr_in = (dr_in - model.desired_return_center) / model.desired_return_scale
                elif os.environ.get("PCN_ADAPTIVE_RETURN_NORMALIZATION", "1") != "1":
                    scale = float(os.environ.get("PCN_DESIRED_RETURN_SCALE", "10000"))
                    if scale > 0:
                        dr_in = dr_in / scale
                c_in = th.cat((dr_in, th.clamp(hz_t, 0, 1e6)), dim=-1) * model.scaling_factor
                c_emb = model.c_emb(c_in)
            c_list.append(c_emb.detach().cpu().numpy().ravel())
    c_arr = np.stack(c_list, axis=0)
    return {
        "c_emb_std_mean": float(np.mean(np.std(c_arr, axis=0))),
        "c_emb_std_max": float(np.max(np.std(c_arr, axis=0))),
        "c_emb_unique_rows": int(len(np.unique(np.round(c_arr, 6), axis=0))),
    }


def run_return_sweep(agent, obs, out_dir: Path, horizons=(1024.0,)):
    wt_vals = np.linspace(0.0, -3.0e6, 9)
    cost_vals = np.linspace(0.0, -2.0e7, 9)
    logits_set = set()
    probs = []
    for hz in horizons:
        for r0 in wt_vals:
            for r1 in cost_vals:
                dr = np.array([r0, r1], dtype=np.float32)
                logits, action, p1 = policy_step0(agent, obs, dr, float(hz))
                logits_set.add(tuple(np.round(logits, 6)))
                probs.append(p1)
    return {
        "grid_size": len(wt_vals) * len(cost_vals) * len(horizons),
        "n_unique_logits": len(logits_set),
        "prob_action1_std": float(np.std(probs)),
        "prob_action1_min": float(np.min(probs)),
        "prob_action1_max": float(np.max(probs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--job-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
    os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

    with open("config/config.yml") as f:
        config = yaml.safe_load(f)
    env = create_event_env(config, job_seed=args.job_seed)
    agent, _ = load_agent(args.checkpoint, env, device=args.device)
    obs = env.reset()

    dr_grid = (np.linspace(0, -3e6, 5), np.linspace(0, -2e7, 5))
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sweep": run_return_sweep(agent, obs, Path(".")),
        "c_emb": measure_c_emb(agent, obs, dr_grid, 1024.0, args.device),
    }

    out = Path(args.output) if args.output else Path(args.checkpoint).parent / "conditioning_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
