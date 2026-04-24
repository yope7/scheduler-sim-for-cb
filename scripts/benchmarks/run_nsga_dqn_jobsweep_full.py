#!/usr/bin/env python3
"""
NSGA-II と分散 DQN（pareto_distributed）を同一 config でジョブ数スイープし、
壁時計時間とログを 0403/raw 配下にまとめる。

前提:
  - uv run またはプロジェクト環境で gym / ray が利用可能
  - DQN は SchedulingEnvEventObs + ラーナー側ビットマップ復元（SCHEDULER_LEARNER_BITMAP=1）
  - 並列ワーカー数 32（--num_workers）

例:
  uv run python scripts/benchmarks/run_nsga_dqn_jobsweep_full.py \\
    --out-root 0403/raw/nsga_dqn_full_100x100_20260411 \\
    --jobs 32 64 128 256
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_subprocess(
    repo_root: Path,
    cwd: Path,
    cmd: list[str],
    log_path: Path,
    extra_env: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    # scripts/main.py はパッケージ未インストール時も動くようリポジトリルートを PYTHONPATH に含める
    env["PYTHONPATH"] = str(repo_root)
    env.update(extra_env)
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"# cmd: {' '.join(cmd)}\n")
        log_f.write(f"# cwd: {cwd}\n")
        log_f.flush()
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    return p.returncode


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="NSGA-II + DQN job sweep with wall-clock logging")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="出力ルート（既定: 0403/raw/nsga_dqn_full_100x100_<日付>）",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="ジョブ数リスト",
    )
    parser.add_argument("--pop-size", type=int, default=100, help="NSGA-II 集団サイズ")
    parser.add_argument("--num-generations", type=int, default=100, help="NSGA-II 世代数")
    parser.add_argument("--how-many-episodes", type=int, default=100, help="DQN 各重みのエピソード数")
    parser.add_argument("--weight-steps", type=int, default=10, help="DQN 重み分割数")
    parser.add_argument("--num-workers", type=int, default=32, help="DQN Ray ワーカー数")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "config.yml",
        help="SCHEDULER_CONFIG に渡す YAML",
    )
    parser.add_argument("--skip-nsga", action="store_true")
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="コマンドのみ表示")
    args = parser.parse_args()

    day = datetime.now().strftime("%Y%m%d")
    out_root = args.out_root
    if out_root is None:
        out_root = root / "0403" / "raw" / f"nsga_dqn_full_100x100_{day}"
    out_root = out_root.resolve()
    cfg_abs = args.config.resolve()

    manifest = {
        "description": (
            "NSGA-II (pop/gen=100/100) + DQN pareto_distributed "
            f"(episodes/weight={args.how_many_episodes}, weight_steps={args.weight_steps}, "
            f"num_workers={args.num_workers}); event_obs + learner bitmap for DQN"
        ),
        "config_path": str(cfg_abs),
        "python": sys.executable,
        "runs": [],
    }

    uv_or_python = ["uv", "run", "python"]
    # uv が無い環境ではそのまま python を使う
    if subprocess.run(["which", "uv"], capture_output=True).returncode != 0:
        uv_or_python = [sys.executable]

    for nj in args.jobs:
        run_entry: dict = {"n_jobs": nj, "nsga2": None, "dqn": None}

        base = out_root / f"njobs_{nj}"
        if not args.skip_nsga:
            nsga_dir = base / "nsga2"
            t0 = time.perf_counter()
            cmd = uv_or_python + [
                str(root / "scripts" / "main.py"),
                "--mode",
                "nsga2",
                "--nb_jobs",
                str(nj),
                "--pop_size",
                str(args.pop_size),
                "--num_generations",
                str(args.num_generations),
            ]
            log_path = nsga_dir / "console.log"
            meta = {
                "wall_time_sec": None,
                "returncode": None,
                "log": str(log_path),
            }
            print(f"[NSGA] n_jobs={nj} -> {nsga_dir}")
            if args.dry_run:
                print(" ", " ".join(cmd))
            else:
                rc = _run_subprocess(
                    root,
                    root,
                    cmd,
                    log_path,
                    {"SCHEDULER_CONFIG": str(cfg_abs)},
                )
                meta["returncode"] = rc
                meta["wall_time_sec"] = time.perf_counter() - t0
                with open(nsga_dir / "wall_time.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            run_entry["nsga2"] = meta

        if not args.skip_dqn:
            dqn_dir = base / "dqn"
            t0 = time.perf_counter()
            cmd = uv_or_python + [
                str(root / "scripts" / "main.py"),
                "--mode",
                "pareto_distributed",
                "--nb_jobs",
                str(nj),
                "--how_many_episodes",
                str(args.how_many_episodes),
                "--weight_steps",
                str(args.weight_steps),
                "--num_workers",
                str(args.num_workers),
            ]
            log_path = dqn_dir / "console.log"
            meta = {
                "wall_time_sec": None,
                "returncode": None,
                "log": str(log_path),
            }
            print(f"[DQN] n_jobs={nj} -> {dqn_dir}")
            if args.dry_run:
                print(" ", " ".join(cmd))
            else:
                rc = _run_subprocess(
                    root,
                    dqn_dir,
                    cmd,
                    log_path,
                    {
                        "SCHEDULER_CONFIG": str(cfg_abs),
                        "SCHEDULER_LEARNER_BITMAP": "1",
                        "DISTRIBUTED_PCN_USE_EVENT_OBS": "1",
                    },
                )
                meta["returncode"] = rc
                meta["wall_time_sec"] = time.perf_counter() - t0
                with open(dqn_dir / "wall_time.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            run_entry["dqn"] = meta

        manifest["runs"].append(run_entry)

    out_root.mkdir(parents=True, exist_ok=True)
    man_path = out_root / "full_experiment_manifest.json"
    if not args.dry_run:
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Wrote {man_path}")


if __name__ == "__main__":
    main()
