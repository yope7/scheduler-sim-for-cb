#!/usr/bin/env python3
"""SchedulingEnvEventObs のスケジュール結果を Gantt 形式で可視化する。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator


def load_config(root: Path) -> dict:
    with open(root / "config" / "config.yml") as f:
        return yaml.safe_load(f)


def run_episode(
    env: SchedulingEnvEventNative,
    seed: int,
    *,
    fifo: bool = False,
    schedule_action: int = 1,
) -> None:
    """1 step ごとに次ジョブを割当。fifo=True なら到着順（jobs を submit_time 順）+ 固定 action。"""
    if fifo and hasattr(env, "jobs_set") and env.jobs_set:
        ep = int(getattr(env, "episode", 0))
        jobs = sorted(env.jobs_set[ep], key=lambda j: j[0])
        env.jobs_set[ep] = jobs
    rng = np.random.default_rng(seed)
    env.reset()
    done = False
    while not done:
        action = int(schedule_action) if fifo else int(rng.integers(0, 2))
        _, _, _, _, done = env.step(action)
    env.finalize_window_history(build_maps=False)


def job_color_map(job_ids: list[int]) -> dict[int, tuple]:
    palette = []
    for cm in (plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")):
        for i in range(cm.N):
            palette.append(cm(i))
    return {jid: palette[i % len(palette)] for i, jid in enumerate(job_ids)}


def crop_time(mat: np.ndarray, col_pad: int = 1) -> tuple[np.ndarray, int]:
    cols = np.where(np.any(mat >= 0, axis=0))[0]
    if cols.size == 0:
        return mat[:, :1], 0
    c0 = max(0, int(cols.min()) - col_pad)
    c1 = min(mat.shape[1], int(cols.max()) + 1 + col_pad)
    return mat[:, c0:c1], c0


def build_matrix_from_records(
    records: list[dict],
    n_nodes: int,
    use_cloud: bool,
) -> np.ndarray:
    """絶対時刻軸で占有セルを job_id で埋める（a=0 割当ログ用）。"""
    subset = [r for r in records if r["use_cloud"] == use_cloud]
    if not subset:
        return np.full((n_nodes, 1), -1, dtype=np.int32)
    t_end = max(int(r["end"]) for r in subset)
    mat = np.full((n_nodes, t_end), -1, dtype=np.int32)
    for r in subset:
        jid = int(r["job_id"])
        t0 = int(r["start"])
        t1 = int(r["end"])
        h = int(r["height"])
        node_cols = r.get("node_cols")
        if node_cols is not None:
            for col_off, nodes in enumerate(node_cols):
                t = t0 + col_off
                if t >= t1:
                    break
                for node in nodes:
                    ni = int(node)
                    if 0 <= ni < n_nodes:
                        mat[ni, t] = jid
        else:
            i0 = int(r["start_node"])
            i1 = min(i0 + h, n_nodes)
            if t0 < t1 and i0 < n_nodes:
                mat[i0:i1, t0:t1] = jid
    return mat


def _contiguous_runs(nodes: list[int]) -> list[tuple[int, int]]:
    if not nodes:
        return []
    sorted_nodes = sorted(int(n) for n in nodes)
    runs = []
    start = prev = sorted_nodes[0]
    for node in sorted_nodes[1:]:
        if node == prev + 1:
            prev = node
            continue
        runs.append((start, prev + 1))
        start = prev = node
    runs.append((start, prev + 1))
    return runs


def draw_records(
    ax,
    records: list[dict],
    n_nodes: int,
    use_cloud: bool,
    title: str,
    colors: dict[int, tuple],
) -> None:
    """イベント矩形を直接描画する。小さいジョブも枠線とラベルで見えるようにする。"""
    subset = [r for r in records if r["use_cloud"] == use_cloud]
    ax.set_facecolor("#f7f7f7")
    if not subset:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        return

    max_t = max(int(r["end"]) for r in subset)
    max_node = 0
    for r in subset:
        color = colors[int(r["job_id"])]
        t0 = int(r["start"])
        width = int(r["end"]) - t0
        if r.get("node_cols") is not None:
            # 分散割当は列ごとの実ノードを小矩形で描く。
            for col_off, nodes in enumerate(r["node_cols"]):
                for y0, y1 in _contiguous_runs(list(nodes)):
                    max_node = max(max_node, y1)
                    ax.add_patch(
                        Rectangle(
                            (t0 + col_off, y0),
                            1,
                            y1 - y0,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.25,
                        )
                    )
        else:
            y0 = int(r["start_node"])
            height = int(r["height"])
            max_node = max(max_node, y0 + height)
            ax.add_patch(
                Rectangle(
                    (t0, y0),
                    width,
                    height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.45,
                )
            )
            if width >= 2 and height >= 4:
                ax.text(
                    t0 + width / 2,
                    y0 + height / 2,
                    str(int(r["job_id"])),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                    clip_on=True,
                )

    ax.set_xlim(0, max_t + 1)
    ax.set_ylim(min(n_nodes, max_node + 8), 0)
    ax.set_title(title)
    ax.set_xlabel("time (absolute)")
    ax.set_ylabel("node (absolute)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maps/event_env_32jobs_schedule.png"),
    )
    parser.add_argument(
        "--build-maps",
        action="store_true",
        help="ノード×時刻の job_id 行列も構築（重い。既定はイベント records の Gantt のみ）",
    )
    parser.add_argument(
        "--job-trace",
        type=Path,
        default=None,
        help="job_type=2 用 CSV（未指定時は config.yml の param_job）",
    )
    parser.add_argument("--job-trace-n-jobs", type=int, default=24)
    parser.add_argument(
        "--fifo",
        action="store_true",
        help="到着順（submit_time）に並べ、全ジョブを同一 action で割当（既定 action=1 クラウド）",
    )
    parser.add_argument(
        "--schedule-action",
        type=int,
        choices=[0, 1],
        default=1,
        help="FIFO 時の action: 0=オンプレ, 1=クラウド",
    )
    args = parser.parse_args()

    config = load_config(ROOT)
    if args.job_trace is not None:
        config["param_job"] = {**config.get("param_job", {})}
        config["param_job"]["job_type"] = 2
        config["param_job"]["job_trace_path"] = str(args.job_trace)
        config["param_job"]["job_trace_n_jobs"] = int(args.job_trace_n_jobs)
    pe = config["param_env"]
    pa = config["param_agent"]

    job_gen = JobGenerator(
        0,
        config["param_simulation"]["nb_steps"],
        pe["n_window"],
        pe["n_on_premise_node"],
        pe["n_cloud_node"],
        config,
        args.n_jobs,
        0.2,
        1,
    )
    env = SchedulingEnvEventNative(
        np.inf,
        pe["n_window"],
        pe["n_on_premise_node"],
        pe["n_cloud_node"],
        pe["n_job_queue_obs"],
        pe["n_job_queue_bck"],
        pa["weight_wt"],
        pa["weight_cost"],
        pe["penalty_not_allocate"],
        pe["penalty_invalid_action"],
        job_gen.generate_jobs_set(),
        None,
        flag=0,
    )
    run_episode(
        env,
        args.seed,
        fifo=args.fifo,
        schedule_action=args.schedule_action,
    )

    if args.build_maps:
        env.build_schedule_maps()
        print(
            "schedule maps:",
            env.on_premise_window_history_full.shape,
            env.cloud_window_history_full.shape,
        )

    cost, makespan, avg_wt = env.calc_objective_values()
    records = getattr(env, "_absolute_schedule_records", [])

    subtitle = "absolute time from event records"
    job_ids = sorted({int(r["job_id"]) for r in records})
    colors = job_color_map(job_ids)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True, sharex=True)
    draw_records(
        axes[0],
        records,
        env.n_on_premise_node,
        False,
        f"On-Premise ({subtitle})",
        colors,
    )
    draw_records(
        axes[1],
        records,
        env.n_cloud_node,
        True,
        f"Cloud ({subtitle})",
        colors,
    )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=8, color=colors[j])
        for j in job_ids
    ]
    axes[1].legend(
        handles,
        [f"job_id={j}" for j in job_ids],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=7,
    )
    policy = (
        f"FIFO action={args.schedule_action} ({'cloud' if args.schedule_action else 'on-prem'})"
        if args.fifo
        else f"random seed={args.seed}"
    )
    trace_label = args.job_trace.name if args.job_trace else "config job"
    fig.suptitle(
        f"EventNative / {args.n_jobs} jobs / {trace_label} / {policy}\n"
        f"cost={cost}, makespan={makespan}, avg_wait={avg_wt:.2f}",
        fontsize=11,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
