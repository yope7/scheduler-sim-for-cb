#!/usr/bin/env python3
"""均等サンプリングした command(desired_return) を方策へ与え、到達 PF の全域カバーを検証する。

PCN 原理どおり「決定者が desired_return を選ぶ」評価を、目的空間の一様グリッドで行う。
- command = desired_return = [r0=-total_wait, r1=-cost]
- cost≈0 端 (r1=0) と wait≈最小 端 (r1=-cost_max) を必ず含む。
出力: 到達点の散布図 + 非支配集合、cost/avgwait のカバレッジ統計。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

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


from src.utils.pf_uniform_plot import build_r1_grid
from src.utils.pf_command_eval import (
    dedupe_pf,
    load_ref_pf,
    pf_points_to_commands,
    stratified_sample_pf,
)


def run(checkpoint: Path, replay_snapshot: Path | None, out_dir: Path, label: str,
        grid: int = 16, extend: float = 1.10, device: str = "cpu",
        n_jobs_override: int = 0,
        config_path: str | None = None,
        focus_cost_frac: float = 0.0, focus_r1_extra: int = 0,
        focus_half_width_frac: float = 0.04,
        low_tail_frac: float = 0.0, low_tail_extra: int = 0,
        command_mode: str = "uniform",
        ref_pf_npz: Path | None = None,
        pf_command_max: int = 0,
        pf_command_count: int = 0,
        pf_command_jitter: float = 0.0,
        pf_low_wait_frac: float = 0.0,
        pf_low_wait_quota: int = 0,
        pf_low_wait_max: float = 0.0,
        pf_include_extremes: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        os.environ["DISTRIBUTED_PCN_CONFIG"] = str(config_path)
    config = load_config(config_path)
    job_type = int(config.get("param_job", {}).get("job_type", 1))
    print(f"[eval] config={config_path or os.environ.get('DISTRIBUTED_PCN_CONFIG', 'config/config.yml')} "
          f"job_type={job_type} trace_n={config.get('param_job', {}).get('job_trace_n_jobs', '?')}")
    # snapshot があれば archive レンジ・n_jobs をそこから、無ければ CLI の n_jobs と
    # チェックポイント内の学習済み desired_return_scale（到達レンジの頑健最大）から推定する。
    snap = None
    snap_path: Path | None = None
    if replay_snapshot is not None and str(replay_snapshot):
        snap_path = Path(replay_snapshot)
    else:
        snap_path = find_replay_snapshot_for_checkpoint(checkpoint)
        if snap_path is not None:
            print(f"[auto snapshot] {snap_path}")
    if snap_path is not None and snap_path.is_file():
        snap = load_learner_replay_snapshot(str(snap_path))
        n_jobs = int(snap.get("metadata", {}).get("n_jobs", eval_n_jobs(config)))
    else:
        n_jobs = int(n_jobs_override) if n_jobs_override else int(eval_n_jobs(config))
    env = create_eval_env(config, job_seed=0, n_jobs=n_jobs)

    state = th.load(str(checkpoint), map_location=device, weights_only=False)
    model_type = state.get("model_type", "DiscreteActionsDefaultModel")
    h_scale = 1.0 / max(1, n_jobs)
    agent = PCN(env, device=device, state_dim=env.observation_space.shape[0],
                scaling_factor=np.array([1.0, 1.0, h_scale], dtype=np.float32), learning_rate=1e-3, batch_size=512,
                hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
                use_enhanced_model=(model_type == "EnhancedPCNModel"))
    target = agent.network if agent.use_enhanced_model else agent.model
    target.load_state_dict(state.get("model_state_dict", state), strict=False)
    target.eval()

    exploration_pts = np.empty((0, 2), dtype=np.float64)
    archive_pf = np.empty((0, 2), dtype=np.float64)
    if snap is not None:
        exploration_pts = episode_objectives_from_snapshot(snap, n_jobs)
        archive_pf = archive_pf_from_snapshot(snap, n_jobs)
        ref = exploration_pts if exploration_pts.size else archive_pf
        cost_max = float(ref[:, 0].max()) if ref.size else 90000.0
        wait_max = float(ref[:, 1].max()) if ref.size else 220.0
    else:
        scale = target.desired_return_scale.detach().cpu().numpy().astype(np.float64)
        cost_max = float(scale[1])
        wait_max = float(scale[0]) / max(1, n_jobs)
        print(f"[range from checkpoint scale] cost_max={cost_max:.0f} avgwait_max={wait_max:.1f} "
              f"(total_wait_scale={scale[0]:.0f}) n_jobs={n_jobs}")

    # command 生成: uniform 格子 or PF 点（参照 / archive）
    command_mode = (command_mode or "uniform").strip().lower()
    ref_pf_targets = np.empty((0, 2), dtype=np.float64)
    cmd_triples: list[tuple[float, float, float]] = []
    if command_mode == "pf_ref":
        if ref_pf_npz is None:
            raise ValueError("command_mode=pf_ref requires ref_pf_npz")
        ref_pf_targets = load_ref_pf(Path(ref_pf_npz))
        if pf_command_count > 0:
            ref_pf_targets = stratified_sample_pf(
                ref_pf_targets,
                pf_command_count,
                low_wait_frac=pf_low_wait_frac,
                low_wait_quota=pf_low_wait_quota,
                low_wait_max=pf_low_wait_max,
                include_extremes=pf_include_extremes,
            )
        elif pf_command_max > 0:
            ref_pf_targets = stratified_sample_pf(
                ref_pf_targets,
                pf_command_max,
                low_wait_frac=pf_low_wait_frac,
                low_wait_quota=pf_low_wait_quota,
                low_wait_max=pf_low_wait_max,
                include_extremes=pf_include_extremes,
            )
        cmd_pairs = pf_points_to_commands(
            ref_pf_targets, n_jobs, jitter_frac=pf_command_jitter, dedupe=(pf_command_count <= 0)
        )
        cmd_triples = [(float(dr[0]), float(dr[1]), float(hz)) for dr, hz in cmd_pairs]
    elif command_mode == "pf_archive":
        ref_pf_targets = dedupe_pf(archive_pf) if archive_pf.size else dedupe_pf(exploration_pts)
        if pf_command_count > 0 and ref_pf_targets.size:
            ref_pf_targets = stratified_sample_pf(
                ref_pf_targets,
                pf_command_count,
                low_wait_frac=pf_low_wait_frac,
                low_wait_quota=pf_low_wait_quota,
                low_wait_max=pf_low_wait_max,
                include_extremes=pf_include_extremes,
            )
        elif pf_command_max > 0 and ref_pf_targets.size:
            ref_pf_targets = stratified_sample_pf(
                ref_pf_targets,
                pf_command_max,
                low_wait_frac=pf_low_wait_frac,
                low_wait_quota=pf_low_wait_quota,
                low_wait_max=pf_low_wait_max,
                include_extremes=pf_include_extremes,
            )
        cmd_pairs = pf_points_to_commands(ref_pf_targets, n_jobs, jitter_frac=pf_command_jitter)
        cmd_triples = [(float(dr[0]), float(dr[1]), float(hz)) for dr, hz in cmd_pairs]
    else:
        r1_grid = build_r1_grid(
            cost_max,
            extend,
            grid,
            focus_frac=focus_cost_frac,
            focus_extra=focus_r1_extra,
            focus_half_width_frac=focus_half_width_frac,
            low_tail_frac=low_tail_frac,
            low_tail_extra=low_tail_extra,
        )
        r0_grid = np.linspace(0.0, -wait_max * n_jobs * extend, grid)
        cmd_triples = [(float(r0), float(r1), float(n_jobs)) for r0 in r0_grid for r1 in r1_grid]

    max_return = np.full(2, np.inf, dtype=np.float32)
    command_targets = [
        [float(-r1), float(-r0) / max(1, n_jobs)]
        for r0, r1, _hz in cmd_triples
    ]
    achieved = []
    for r0, r1, hz in cmd_triples:
        dr = np.array([r0, r1], dtype=np.float32)
        _, _, _, _, _, val = agent._run_episode(env, dr.copy(), np.float32(hz),
                                                 max_return, eval_mode=True)
        achieved.append([float(val[0]), float(val[1])])  # cost, avgwait
    pts = np.asarray(achieved, dtype=np.float64)

    nd = get_non_dominated_inds_minimize(pts)
    pf = pts[nd]
    uniq = {tuple(np.round(r, 3)) for r in pts}
    uniq_arr = np.array(list(uniq), dtype=np.float64)
    nd_u = get_non_dominated_inds_minimize(uniq_arr) if len(uniq_arr) else np.array([], int)
    pf_plot = uniq_arr[nd_u] if len(nd_u) else uniq_arr
    if command_mode in ("pf_ref", "pf_archive"):
        pf = pf_plot
        nd = nd_u

    stats = {
        "label": label,
        "command_mode": command_mode,
        "n_commands": len(cmd_triples),
        "n_pf_command_targets": int(len(ref_pf_targets)),
        "grid": grid if command_mode == "uniform" else 0,
        "n_pf": int(len(nd)),
        "n_pf_deduped": int(len(nd_u)),
        "n_unique_achieved": int(len(uniq)),
        "cost_min": float(pts[:, 0].min()), "cost_max": float(pts[:, 0].max()),
        "wait_min": float(pts[:, 1].min()), "wait_max": float(pts[:, 1].max()),
        "archive_cost_max": cost_max, "archive_wait_max": wait_max,
        "n_exploration_pts": int(len(exploration_pts)),
        "n_archive_pf": int(len(archive_pf)),
        "n_jobs": n_jobs,
        "checkpoint": str(checkpoint.resolve()),
        "replay_snapshot": str(snap_path.resolve()) if snap_path else "",
        "command_targets": command_targets,
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    pf_command_eval = command_mode in ("pf_ref", "pf_archive")
    if exploration_pts.size:
        ax.scatter(
            exploration_pts[:, 0], exploration_pts[:, 1],
            c="cyan", s=14, alpha=0.35, marker="o",
            label=f"Exploration / archive ({len(exploration_pts)})", zorder=2,
        )
    if archive_pf.size:
        ax.scatter(archive_pf[:, 0], archive_pf[:, 1], c="cyan", s=40, marker="D",
                   edgecolors="teal", linewidths=0.4, alpha=0.9,
                   label=f"Archive PF ({len(archive_pf)})", zorder=4)
    if ref_pf_targets.size and command_mode == "pf_ref":
        ax.scatter(
            ref_pf_targets[:, 0], ref_pf_targets[:, 1],
            c="gold", s=12, alpha=0.55, marker=".",
            label=f"Reference PF ({len(ref_pf_targets)})", zorder=1,
        )
    elif ref_pf_targets.size and command_mode == "pf_archive":
        ax.scatter(
            ref_pf_targets[:, 0], ref_pf_targets[:, 1],
            c="lightgray", s=18, alpha=0.5, marker=".",
            label=f"Command targets ({len(ref_pf_targets)})", zorder=1,
        )
    if not pf_command_eval:
        ax.scatter(pts[:, 0], pts[:, 1], c="blue", s=24, alpha=0.4,
                   label=f"Achieved (uniform cmd, {len(pts)})", zorder=3)
    if pf.size:
        o = np.lexsort((pf[:, 1], pf[:, 0]))
        ax.plot(pf[o, 0], pf[o, 1], "r-o", lw=1.6, ms=5,
                label=f"Eval PF ({len(nd)})", zorder=5)
    mode_title = {"uniform": "Uniform-command", "pf_ref": "PF-ref-command", "pf_archive": "PF-archive-command"}.get(
        command_mode, command_mode
    )
    ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time")
    ax.set_title(f"{mode_title} PF — {label}\n"
                 f"PF={len(nd)} dedup={len(nd_u)} cost[{stats['cost_min']:.0f},{stats['cost_max']:.0f}] "
                 f"wait[{stats['wait_min']:.0f},{stats['wait_max']:.0f}]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"uniform_cmd_pf_{label}_{ts}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    stats["plot"] = str(out_path.resolve())
    stats["achieved_points"] = pts.tolist()
    stats["pareto_front"] = pf.tolist()
    stats["score_points"] = uniq_arr.tolist() if command_mode in ("pf_ref", "pf_archive") else pts.tolist()
    (out_dir / f"uniform_cmd_stats_{label}.json").write_text(json.dumps(stats, indent=2))
    print(f"[{label}] {out_path}\n  PF={stats['n_pf']} dedup={stats['n_pf_deduped']} "
          f"unique={stats['n_unique_achieved']}/{stats['n_commands']}  "
          f"cost[{stats['cost_min']:.0f},{stats['cost_max']:.0f}] "
          f"wait[{stats['wait_min']:.1f},{stats['wait_max']:.1f}]")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.environ.get("DISTRIBUTED_PCN_CONFIG", ""),
                   help="Eval 環境の config（未指定時 config/config.yml。trace24 は job_trace_24_scratch_pass.yml）")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--replay-snapshot", default="")  # 省略可: 無ければ checkpoint の scale を使用
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="uniform")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-jobs", type=int, default=0)  # snapshot 無し時に使用
    p.add_argument(
        "--focus-cost-frac",
        type=float,
        default=0.0,
        help="cost≈frac*cmax 付近で r1 を追加密化（例: 0.077≈0.5e6/6.5e6）",
    )
    p.add_argument("--focus-r1-extra", type=int, default=0, help="追加 r1 本数")
    p.add_argument("--focus-half-width-frac", type=float, default=0.04)
    p.add_argument(
        "--low-tail-frac",
        type=float,
        default=0.0,
        help="cost=0〜frac*cmax の r1 を追加密化（左上先端の診断用）",
    )
    p.add_argument("--low-tail-extra", type=int, default=0, help="低 cost 側の追加 r1 本数")
    p.add_argument(
        "--command-mode",
        choices=("uniform", "pf_ref", "pf_archive"),
        default=os.environ.get("PCN_EVAL_COMMAND_MODE", "uniform"),
        help="uniform=一様格子, pf_ref=参照PF点, pf_archive=snapshot archive PF",
    )
    p.add_argument("--ref-pf-npz", type=Path, default=None, help="command-mode=pf_ref 時の参照 npz")
    p.add_argument("--pf-command-max", type=int, default=int(os.environ.get("PCN_PF_COMMAND_MAX", "0")),
                   help="PF command 数上限（0=全点、pf-command-count 未指定時）")
    p.add_argument("--pf-command-count", type=int, default=int(os.environ.get("PCN_PF_COMMAND_COUNT", "0")),
                   help="参照 PF から cost 層化で N 点 command（0=pf-command-max/全点）")
    p.add_argument("--pf-command-jitter", type=float, default=float(os.environ.get("PCN_PF_COMMAND_JITTER", "0")),
                   help="PF command への相対ジッター")
    p.add_argument(
        "--pf-low-wait-frac",
        type=float,
        default=float(os.environ.get("PCN_PF_COMMAND_LOW_WAIT_FRAC", "0")),
        help="PF command サンプルで低 wait 側へ予約する割合（0=無効）",
    )
    p.add_argument(
        "--pf-low-wait-quota",
        type=int,
        default=int(os.environ.get("PCN_PF_COMMAND_LOW_WAIT_QUOTA", "0")),
        help="PF command サンプルで低 wait 側へ予約する本数（0=割合のみ）",
    )
    p.add_argument(
        "--pf-low-wait-max",
        type=float,
        default=float(os.environ.get("PCN_PF_COMMAND_LOW_WAIT_MAX", "0")),
        help="低 wait 予約枠の閾値（avg wait、0=quantile tail）",
    )
    p.add_argument(
        "--no-pf-command-extremes",
        action="store_true",
        help="PF command サンプルに cost 最小端・wait 最小端を固定投入しない",
    )
    args = p.parse_args()
    snap_arg = Path(args.replay_snapshot) if args.replay_snapshot else None
    run(
        Path(args.checkpoint),
        snap_arg,
        Path(args.output),
        args.label,
        grid=args.grid,
        device=args.device,
        n_jobs_override=args.n_jobs,
        config_path=args.config or None,
        focus_cost_frac=args.focus_cost_frac,
        focus_r1_extra=args.focus_r1_extra,
        focus_half_width_frac=args.focus_half_width_frac,
        low_tail_frac=args.low_tail_frac,
        low_tail_extra=args.low_tail_extra,
        command_mode=args.command_mode,
        ref_pf_npz=args.ref_pf_npz,
        pf_command_max=args.pf_command_max,
        pf_command_count=args.pf_command_count,
        pf_command_jitter=args.pf_command_jitter,
        pf_low_wait_frac=args.pf_low_wait_frac,
        pf_low_wait_quota=args.pf_low_wait_quota,
        pf_low_wait_max=args.pf_low_wait_max,
        pf_include_extremes=not args.no_pf_command_extremes,
    )


if __name__ == "__main__":
    main()
