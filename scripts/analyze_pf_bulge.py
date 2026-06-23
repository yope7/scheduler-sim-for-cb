#!/usr/bin/env python3
"""cost 中域（~0.5e6–2.5e6）の PF 膨らみ / 品質低下を定量化する。

指標:
- archive_gap: Eval PF 上の点について、同 cost 帯の Archive PF 最小 wait との差
- knee_cost: Eval PF で wait が最大→急落する cost（膝点）
- command_sensitivity: 固定 obs で command を振ったときの到達 cost/wait の分散
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th

from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from scripts.pcn_replay_snapshot import (
    archive_pf_from_snapshot,
    create_eval_env,
    episode_objectives_from_snapshot,
    eval_n_jobs,
    find_replay_snapshot_for_checkpoint,
    load_config,
    load_learner_replay_snapshot,
)


def eval_uniform_points(agent, env, n_jobs, cost_max, wait_max, grid=10, extend=1.1):
    r1 = np.linspace(0.0, -cost_max * extend, grid)
    r0 = np.linspace(0.0, -wait_max * n_jobs * extend, grid)
    max_return = np.full(2, np.inf, dtype=np.float32)
    pts = []
    for r0v in r0:
        for r1v in r1:
            dr = np.array([float(r0v), float(r1v)], dtype=np.float32)
            _, _, _, _, _, val = agent._run_episode(
                env, dr.copy(), np.float32(n_jobs), max_return, eval_mode=True
            )
            pts.append([float(val[0]), float(val[1])])
    return np.asarray(pts, dtype=np.float64)


def pf_gap_in_band(eval_pf: np.ndarray, archive_pf: np.ndarray,
                   cost_lo: float, cost_hi: float) -> dict:
    """eval_pf 各点について archive の同帯域 best wait との差（正=eval が悪い）。"""
    if not eval_pf.size or not archive_pf.size:
        return {"n": 0, "mean_gap": np.nan, "max_gap": np.nan, "frac_dominated": np.nan}
    gaps = []
    for c, w in eval_pf:
        if not (cost_lo <= c <= cost_hi):
            continue
        band = archive_pf[(archive_pf[:, 0] >= cost_lo) & (archive_pf[:, 0] <= cost_hi)]
        if not band.size:
            band = archive_pf
        best_w = float(band[:, 1].min())
        gaps.append(w - best_w)
    if not gaps:
        return {"n": 0, "mean_gap": np.nan, "max_gap": np.nan, "frac_dominated": np.nan}
    gaps = np.asarray(gaps)
    return {
        "n": int(len(gaps)),
        "mean_gap": float(gaps.mean()),
        "max_gap": float(gaps.max()),
        "frac_dominated": float((gaps > 50).mean()),  # wait 50 以上悪い
    }


def detect_knee(eval_pf: np.ndarray) -> dict:
    """PF 上で最大の wait 落差（縦落ち）を検出。"""
    if len(eval_pf) < 3:
        return {}
    o = np.argsort(eval_pf[:, 0])
    pf = eval_pf[o]
    drops = np.diff(pf[:, 1])
    idx = int(np.argmin(drops))  # 最も急な下落
    return {
        "knee_cost": float(pf[idx, 0]),
        "wait_before": float(pf[idx, 1]),
        "wait_after": float(pf[idx + 1, 1]),
        "drop": float(pf[idx, 1] - pf[idx + 1, 1]),
        "cost_step": float(pf[idx + 1, 0] - pf[idx, 0]),
    }


def analyze(checkpoint: Path, snapshot: Path | None, grid: int = 10) -> dict:
    config = load_config()
    snap_path = snapshot or find_replay_snapshot_for_checkpoint(checkpoint)
    snap = load_learner_replay_snapshot(str(snap_path)) if snap_path else None
    n_jobs = int(snap.get("metadata", {}).get("n_jobs", 1024)) if snap else 1024
    env = create_eval_env(config, 0, n_jobs)

    state = th.load(str(checkpoint), map_location="cpu", weights_only=False)
    h_scale = 1.0 / max(1, n_jobs)
    agent = PCN(
        env, device="cpu", state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1.0, 1.0, h_scale]), learning_rate=1e-3, batch_size=512,
        hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
        use_enhanced_model=(state.get("model_type") == "EnhancedPCNModel"),
    )
    target = agent.network if agent.use_enhanced_model else agent.model
    target.load_state_dict(state.get("model_state_dict", state), strict=False)
    target.eval()

    exploration = episode_objectives_from_snapshot(snap, n_jobs) if snap else np.zeros((0, 2))
    archive_pf = archive_pf_from_snapshot(snap, n_jobs) if snap else np.zeros((0, 2))
    ref = exploration if exploration.size else archive_pf
    cost_max = float(ref[:, 0].max()) if ref.size else 6e6
    wait_max = float(ref[:, 1].max()) if ref.size else 18000.0

    pts = eval_uniform_points(agent, env, n_jobs, cost_max, wait_max, grid=grid)
    nd = get_non_dominated_inds_minimize(pts)
    eval_pf = pts[nd]

    bands = [
        ("low", 0, 5e5),
        ("low_slope", 0, 1.2e6),
        ("knee_05e6", 3.5e5, 6.5e5),
        ("mid_1e6", 5e5, 1.5e6),
        ("knee", 1.5e6, 2.5e6),
        ("high", 2.5e6, cost_max * 1.1),
    ]
    band_stats = {name: pf_gap_in_band(eval_pf, archive_pf, lo, hi) for name, lo, hi in bands}

    # archive 自体の mid 帯密度（探索は十分か）
    mid_arch = archive_pf[(archive_pf[:, 0] >= 5e5) & (archive_pf[:, 0] <= 2.5e6)]
    knee = detect_knee(eval_pf)

    return {
        "checkpoint": str(checkpoint),
        "snapshot": str(snap_path) if snap_path else "",
        "n_jobs": n_jobs,
        "n_exploration": int(len(exploration)),
        "n_archive_pf": int(len(archive_pf)),
        "n_eval_pf": int(len(eval_pf)),
        "n_mid_archive_pf": int(len(mid_arch)),
        "knee": knee,
        "bands": band_stats,
        "eval_cost_range": [float(eval_pf[:, 0].min()), float(eval_pf[:, 0].max())] if eval_pf.size else [],
        "eval_wait_range": [float(eval_pf[:, 1].min()), float(eval_pf[:, 1].max())] if eval_pf.size else [],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--label", action="append", default=[])
    p.add_argument("--snapshot", default="")
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--output", default="experiments/distributed_pcn/pf_best_current/pf_bulge_analysis.json")
    args = p.parse_args()
    snap = Path(args.snapshot) if args.snapshot else None
    results = []
    labels = args.label if args.label else [Path(c).stem for c in args.checkpoint]
    for i, ck in enumerate(args.checkpoint):
        lab = labels[i] if i < len(labels) else Path(ck).parent.name
        r = analyze(Path(ck), snap, grid=args.grid)
        r["label"] = lab
        results.append(r)
        print(f"\n=== {lab} ===")
        print(f"  exploration={r['n_exploration']} archive_pf={r['n_archive_pf']} eval_pf={r['n_eval_pf']}")
        if r.get("knee"):
            k = r["knee"]
            print(f"  knee: cost={k['knee_cost']:.0f} drop={k['drop']:.0f} "
                  f"({k['wait_before']:.0f} -> {k['wait_after']:.0f})")
        for name, bs in r["bands"].items():
            if bs["n"]:
                print(f"  band {name}: mean_gap={bs['mean_gap']:.0f} max_gap={bs['max_gap']:.0f} "
                      f"dominated_frac={bs['frac_dominated']:.2f} (n={bs['n']})")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
