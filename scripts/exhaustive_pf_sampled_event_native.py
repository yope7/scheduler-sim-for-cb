#!/usr/bin/env python3
"""EventNative 環境で OP/CL 割当ベクトルをサンプリングし、近似パレートフロントを描画する。"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.pcn_agent import get_non_dominated_inds_minimize
from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator


def load_config(root: Path) -> dict:
    with open(root / "config" / "config.yml") as f:
        return yaml.safe_load(f)


def index_to_action(index: int, nb_jobs: int) -> tuple[int, ...]:
    return tuple((int(index) >> j) & 1 for j in range(nb_jobs))


def _dedupe_action_sets(action_sets: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    unique: list[tuple[int, ...]] = []
    for a in action_sets:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


def generate_stratified_by_cloud_count(
    nb_jobs: int, per_k: int, seed: int
) -> list[tuple[int, ...]]:
    """クラウド台数 k 固定でランダムにジョブを選ぶ（中間コスト帯の穴を埋める）。"""
    if per_k <= 0:
        return []
    rng = np.random.default_rng(seed + 1)
    out: list[tuple[int, ...]] = []
    for k in range(nb_jobs + 1):
        if k == 0:
            out.append(tuple([0] * nb_jobs))
            continue
        if k == nb_jobs:
            out.append(tuple([1] * nb_jobs))
            continue
        # k 台クラウドの組合せをサンプル（重複あり得るが後で dedupe）
        n_try = per_k
        for _ in range(n_try):
            cloud_idx = rng.choice(nb_jobs, size=k, replace=False)
            action = [0] * nb_jobs
            for i in cloud_idx:
                action[int(i)] = 1
            out.append(tuple(action))
    return out


def generate_action_sets(
    nb_jobs: int,
    num_samples: int,
    seed: int,
    *,
    stratified_per_k: int = 0,
) -> list[tuple[int, ...]]:
    total = 2**nb_jobs
    if nb_jobs < 20 and num_samples >= total and stratified_per_k <= 0:
        return list(itertools.product([0, 1], repeat=nb_jobs))

    action_sets: list[tuple[int, ...]] = [
        tuple([0] * nb_jobs),
        tuple([1] * nb_jobs),
    ]
    rng = np.random.default_rng(seed)

    # 2^n 空間を均等に覆うインデックス
    n_lin = max(0, num_samples - 2)
    if n_lin > 0:
        lin_idx = np.linspace(0, total - 1, n_lin, dtype=np.uint64)
        for idx in lin_idx:
            action_sets.append(index_to_action(int(idx), nb_jobs))

    # 線形サンプルに無い分をランダム補完
    n_rand = max(0, num_samples - len(action_sets))
    for _ in range(n_rand):
        action_sets.append(tuple(int(x) for x in rng.integers(0, 2, nb_jobs)))

    if stratified_per_k > 0:
        action_sets.extend(
            generate_stratified_by_cloud_count(nb_jobs, stratified_per_k, seed)
        )

    return _dedupe_action_sets(action_sets)


def load_trace_jobs(
    trace_path: Path,
    n_jobs: int,
    *,
    exclude_largest_outlier: bool = False,
) -> list[list]:
    """トレース先頭から n_jobs 件。外れ値除外時は n_jobs+1 件読み pt×nodes 最大を落とす。"""
    repo_root = ROOT
    path = Path(trace_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    read_n = int(n_jobs) + 1 if exclude_largest_outlier else int(n_jobs)
    jobs = pd.read_csv(path, nrows=read_n).values.tolist()
    if exclude_largest_outlier:
        if len(jobs) <= n_jobs:
            raise ValueError(f"need at least {n_jobs + 1} trace rows to exclude one outlier")
        scores = [float(row[1]) * float(row[2]) for row in jobs]
        drop_i = int(np.argmax(scores))
        dropped = jobs.pop(drop_i)
        for i, row in enumerate(jobs):
            row[5] = i
        print(
            f"[trace] excluded largest outlier: trace job_id={dropped[5]}, "
            f"pt={dropped[1]}, nodes={dropped[2]}, pt*nodes={scores[drop_i]:.0f}",
        )
    if len(jobs) != n_jobs:
        raise ValueError(f"expected {n_jobs} jobs after filter, got {len(jobs)}")
    return jobs


def build_env_params(
    config: dict,
    n_jobs: int,
    job_trace: Path | None,
    job_trace_n_jobs: int,
    *,
    exclude_largest_outlier: bool = False,
    jobs_override: list[list] | None = None,
) -> dict:
    pe = config["param_env"]
    pa = config["param_agent"]
    if jobs_override is not None:
        jobs_set = [jobs_override]
    else:
        if job_trace is not None:
            config = {**config, "param_job": {**config.get("param_job", {})}}
            config["param_job"]["job_type"] = 2
            config["param_job"]["job_trace_path"] = str(job_trace)
            config["param_job"]["job_trace_n_jobs"] = int(job_trace_n_jobs)
        job_gen = JobGenerator(
            0,
            config["param_simulation"]["nb_steps"],
            pe["n_window"],
            pe["n_on_premise_node"],
            pe["n_cloud_node"],
            config,
            n_jobs,
            0.2,
            1,
        )
        jobs_set = job_gen.generate_jobs_set()
    return {
        "args": (
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
            jobs_set,
            None,
            0,
        ),
    }


def evaluate_batch(batch: list[tuple[int, ...]], env_params: dict) -> list[tuple[float, float]]:
    env = SchedulingEnvEventNative(*env_params["args"])
    out: list[tuple[float, float]] = []
    for action_set in batch:
        env.reset()
        done = False
        step = 0
        while not done:
            _, _, _, _, done = env.step(action_set[step])
            step += 1
        env.finalize_window_history(build_maps=False)
        cost, _, avg_wt = env.calc_objective_values()
        out.append((float(cost), float(avg_wt)))
    return out


_ENV_PARAMS: dict | None = None


def _init_worker(env_params: dict) -> None:
    global _ENV_PARAMS
    _ENV_PARAMS = env_params


def _worker(batch: list[tuple[int, ...]]) -> list[tuple[float, float]]:
    assert _ENV_PARAMS is not None
    return evaluate_batch(batch, _ENV_PARAMS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument(
        "--job-trace",
        type=Path,
        default=Path("job_trace/FY2024/scacctreq_202412_dec1_1000_jobs.csv"),
    )
    parser.add_argument("--job-trace-n-jobs", type=int, default=24)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2**20,
        help="2^n からの均等サンプル数（既定 2^20）",
    )
    parser.add_argument(
        "--stratified-per-k",
        type=int,
        default=12000,
        help="クラウド台数 k=0..n ごとのランダム割当数（0 で無効）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="0=CPU数")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/distributed_pcn/trace24_sampled_exhaustive_pf.png"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--exclude-largest-outlier",
        action="store_true",
        help="先頭 n+1 件から pt×nodes 最大の1件を除き n 件にする",
    )
    args = parser.parse_args()

    config = load_config(ROOT)
    jobs_override = None
    if args.exclude_largest_outlier and args.job_trace is not None:
        jobs_override = load_trace_jobs(
            args.job_trace,
            args.n_jobs,
            exclude_largest_outlier=True,
        )
        print("[JobGenerator] 先頭5件サンプル (outlier除外後):", jobs_override[:5])
    env_params = build_env_params(
        config,
        args.n_jobs,
        args.job_trace,
        args.job_trace_n_jobs,
        exclude_largest_outlier=args.exclude_largest_outlier,
        jobs_override=jobs_override,
    )
    action_sets = generate_action_sets(
        args.n_jobs,
        args.num_samples,
        args.seed,
        stratified_per_k=args.stratified_per_k,
    )
    total = len(action_sets)
    full = 2**args.n_jobs
    print(
        f"n_jobs={args.n_jobs}, evaluate={total} unique assignments "
        f"({100*total/full:.4f}% of 2^{args.n_jobs}); "
        f"lin={args.num_samples}, stratified_per_k={args.stratified_per_k}",
    )

    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    batches = [
        action_sets[i : i + args.batch_size]
        for i in range(0, total, args.batch_size)
    ]
    points: list[tuple[float, float]] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(env_params,)
    ) as ex:
        futs = [ex.submit(_worker, b) for b in batches]
        done_n = 0
        for fut in as_completed(futs):
            points.extend(fut.result())
            done_n += 1
            if done_n % max(1, len(futs) // 10) == 0 or done_n == len(futs):
                print(
                    f"  progress {done_n}/{len(futs)} batches, "
                    f"{len(points)}/{total} points, {time.time()-t0:.1f}s",
                    flush=True,
                )
    elapsed = time.time() - t0
    print(f"evaluated {len(points)} points in {elapsed:.1f}s ({len(points)/max(elapsed,1e-6):.1f} eps/s)")

    arr = np.asarray(points, dtype=np.float64)
    nd = get_non_dominated_inds_minimize(arr)
    pf = arr[nd]
    pf = pf[np.argsort(pf[:, 0])]

    npz_path = args.output.with_suffix(".npz")
    np.savez(
        npz_path,
        points=arr,
        pareto_front=pf,
        n_jobs=args.n_jobs,
        num_samples=args.num_samples,
        stratified_per_k=args.stratified_per_k,
    )
    print(f"saved data {npz_path}")

    # 参考点
    ref_all_onpre = evaluate_batch([tuple([0] * args.n_jobs)], env_params)[0]
    ref_all_cloud = evaluate_batch([tuple([1] * args.n_jobs)], env_params)[0]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        arr[:, 0],
        arr[:, 1],
        c="#9ecae1",
        s=8,
        alpha=0.35,
        label=f"sampled ({len(arr)})",
        rasterized=True,
    )
    ax.plot(
        pf[:, 0],
        pf[:, 1],
        "o-",
        color="#08519c",
        markersize=7,
        linewidth=1.5,
        label=f"Pareto front ({len(pf)} nd)",
    )
    ax.scatter(
        [ref_all_onpre[0]],
        [ref_all_onpre[1]],
        marker="s",
        s=120,
        c="#2ca02c",
        edgecolors="black",
        zorder=5,
        label=f"all on-prem (cost={ref_all_onpre[0]:.0f})",
    )
    ax.scatter(
        [ref_all_cloud[0]],
        [ref_all_cloud[1]],
        marker="^",
        s=120,
        c="#d62728",
        edgecolors="black",
        zorder=5,
        label=f"all cloud (cost={ref_all_cloud[0]:.0f})",
    )
    ax.set_xlabel("total cost (minimize)")
    ax.set_ylabel("avg waiting time (minimize)")
    outlier_note = " (no largest outlier)" if args.exclude_largest_outlier else ""
    ax.set_title(
        f"Sampled exhaustive PF — EventNative trace24{outlier_note}\n"
        f"{args.job_trace.name}, n={args.n_jobs}, "
        f"{total}/{full} assignments, {elapsed:.0f}s",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output}")
    print(f"Pareto front ({len(pf)} points), cost [{pf[:,0].min():.0f}, {pf[:,0].max():.0f}], "
          f"avg_wait [{pf[:,1].min():.2f}, {pf[:,1].max():.2f}]")


if __name__ == "__main__":
    main()
