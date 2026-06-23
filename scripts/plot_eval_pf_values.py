#!/usr/bin/env python3
"""Eval PF (赤) と Archive PF (シアン) を learner replay スナップショットで比較描画。"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("PCN_EVAL_PF_GRID", "64")
os.environ.setdefault("PCN_EVAL_STOCHASTIC", "0")

from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from scripts.pcn_replay_snapshot import (
    archive_pf_from_snapshot,
    create_eval_env,
    eval_n_jobs,
    load_config,
    load_learner_replay_snapshot,
)


def _apply_pf_axis_limits(ax, *point_arrays: np.ndarray) -> None:
    chunks: List[np.ndarray] = []
    for arr in point_arrays:
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0 or a.ndim != 2 or a.shape[1] < 2:
            continue
        chunks.append(a[:, :2])
    if not chunks:
        return
    zoom_x = os.environ.get("PCN_PF_ZOOM_COST_MAX")
    y_pad = float(os.environ.get("PCN_PF_Y_PAD_RATIO", "0.12"))
    if zoom_x:
        x_hi = float(zoom_x)
        ys_in_x = []
        for a in chunks:
            mask = a[:, 0] <= x_hi + 1e-6
            if np.any(mask):
                ys_in_x.append(a[mask, 1])
        if ys_in_x:
            y_max = float(np.max(np.concatenate(ys_in_x)))
        else:
            y_max = float(np.max(np.vstack(chunks)[:, 1]))
        y_margin = max(y_max * y_pad, 1.0)
        ax.set_xlim(0.0, x_hi)
        ax.set_ylim(0.0, y_max + y_margin)
        return
    data = np.vstack(chunks)
    xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
    ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    xm = max((xmax - xmin) * 0.12, 1.0)
    ym = max((ymax - ymin) * 0.12, 1.0)
    ax.set_xlim(max(0.0, xmin - xm), xmax + xm)
    ax.set_ylim(max(0.0, ymin - ym), ymax + ym)


def run_one(
    checkpoint: Path,
    replay_snapshot: Path,
    out_dir: Path,
    label: str,
    n_eval: int = 200,
    device: str = "cpu",
) -> Tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    snap = load_learner_replay_snapshot(str(replay_snapshot))
    n_jobs = int(snap.get("metadata", {}).get("n_jobs", eval_n_jobs(config)))
    env = create_eval_env(config, job_seed=0, n_jobs=n_jobs)

    state = th.load(str(checkpoint), map_location=device, weights_only=False)
    model_type = state.get("model_type", "DiscreteActionsDefaultModel")
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
        use_enhanced_model=(model_type == "EnhancedPCNModel"),
    )
    sd = state.get("model_state_dict", state)
    target = agent.network if agent.use_enhanced_model else agent.model
    target.load_state_dict(sd, strict=False)
    target.eval()

    import heapq

    for i, ep in enumerate(snap.get("episodes", [])):
        if ep:
            heapq.heappush(agent.experience_replay, (1.0, (i, i), ep))

    max_return = np.full(2, 100.0, dtype=np.float32)
    _, e_values, _, _ = agent.evaluate(env, max_return, n=n_eval, save_history=False)
    e_values_np = np.asarray(e_values, dtype=np.float64)
    nd_i = get_non_dominated_inds_minimize(e_values_np)
    eval_pf = e_values_np[nd_i]
    dedupe_keys = {tuple(np.round(row, 4)) for row in e_values_np}
    deduped = np.array(list(dedupe_keys), dtype=np.float64) if dedupe_keys else np.zeros((0, 2))
    nd_deduped = get_non_dominated_inds_minimize(deduped) if len(deduped) else np.array([], dtype=int)
    eval_pf_deduped = deduped[nd_deduped] if len(nd_deduped) else np.zeros((0, 2))
    archive_pf = archive_pf_from_snapshot(snap, n_jobs)

    fig, ax = plt.subplots(figsize=(8, 6))
    if archive_pf.size:
        ax.scatter(
            archive_pf[:, 0],
            archive_pf[:, 1],
            c="cyan",
            s=42,
            marker="D",
            alpha=0.85,
            label=f"Archive PF ({len(archive_pf)})",
            zorder=6,
        )
        if len(archive_pf) > 1:
            o = np.lexsort((archive_pf[:, 1], archive_pf[:, 0]))
            ax.plot(archive_pf[o, 0], archive_pf[o, 1], color="cyan", ls=":", lw=1.2, alpha=0.65)
    if e_values_np.size:
        ax.scatter(
            e_values_np[:, 0],
            e_values_np[:, 1],
            c="blue",
            s=40,
            alpha=0.45,
            label=f"Eval all ({len(e_values_np)})",
            zorder=3,
        )
    if eval_pf.size:
        ax.scatter(
            eval_pf[:, 0],
            eval_pf[:, 1],
            c="red",
            s=55,
            label=f"Eval PF ({len(eval_pf)})",
            zorder=5,
        )
        if len(eval_pf) > 1:
            o = np.lexsort((eval_pf[:, 1], eval_pf[:, 0]))
            ax.plot(eval_pf[o, 0], eval_pf[o, 1], "r-", lw=1.5, alpha=0.85)

    _apply_pf_axis_limits(ax, e_values_np, archive_pf)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Average Waiting Time")
    ax.set_title(f"PF values — {label}\nNon-dominated: {len(nd_i)}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pareto_front_values_{label}_{ts}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_eval_points": int(len(e_values_np)),
        "n_eval_pf": int(len(nd_i)),
        "n_eval_pf_deduped": int(len(nd_deduped)),
        "n_unique_values": int(len(dedupe_keys)),
        "eval_pf_cost_range": [float(eval_pf[:, 0].min()), float(eval_pf[:, 0].max())]
        if len(eval_pf)
        else None,
        "checkpoint": str(checkpoint.resolve()),
        "replay_snapshot": str(replay_snapshot.resolve()),
        "n_jobs": n_jobs,
        "replay_episodes_loaded": len(snap.get("episodes", [])),
        "n_archive_pf": int(len(archive_pf)),
        "plot": str(out_path.resolve()),
        "label": label,
    }
    (out_dir / f"eval_stats_{label}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(
        f"[{label}] saved {out_path}  eval_PF={stats['n_eval_pf']}  "
        f"pf_deduped={stats['n_eval_pf_deduped']}  unique={stats['n_unique_values']}  "
        f"all={stats['n_eval_points']}"
    )
    return out_path, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--replay-snapshot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="eval")
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run_one(
        Path(args.checkpoint),
        Path(args.replay_snapshot),
        Path(args.output),
        args.label,
        n_eval=args.n_eval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
