#!/usr/bin/env python3
"""
3アルゴリズム（分散 PCN / NSGA-II / DQN）のベンチマーク。

- NSGA-II: プロセス内で run_nsga2_trace（世代ごとのパレート → HV 系列）
- PCN: サブプロセス + DISTRIBUTED_PCN_MO_HV_EXPORT（評価ごとのパレート）
- DQN: Ray 分散重みスイープ（pareto_distributed）→ pareto JSON

3者とも同一参照点（目的空間の和集合 + margin）で final HV・HV 系列・TTS を
build_quality_report_from_traces（mo_benchmark_hv）で算出する。
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.utils.algorithm_compare_config import get_param_algorithm_compare, load_full_config


def _load_mo_benchmark_hv():
    p = PROJECT_ROOT / "scripts" / "benchmarks" / "mo_benchmark_hv.py"
    spec = importlib.util.spec_from_file_location("_mo_benchmark_hv_loaded", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_subprocess(cmd: list, env: Dict[str, str]):
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    return proc, wall


def _pareto_json_paths() -> List[str]:
    return sorted(
        glob.glob(str(PROJECT_ROOT / "distributed_pareto_results" / "pareto_data_distributed_*.json"))
    )


def _summarize_trace_for_json(tr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """JSON 保存用に巨大配列を削る（quality_metrics に詳細指標あり）。"""
    if not tr:
        return None
    name = tr.get("name")
    if name == "nsga2":
        pfs = tr.get("pareto_fronts_per_generation") or []
        return {
            "name": name,
            "n_generations": len(pfs),
            "final_objectives": tr.get("final_objectives"),
            "time_axis_note": tr.get("time_axis_note"),
        }
    if name == "pcn_distributed":
        ev = tr.get("pareto_fronts_per_eval") or []
        return {
            "name": name,
            "returncode": tr.get("returncode"),
            "n_evaluations": len(ev),
            "n_points_last_eval": len(ev[-1]) if ev else 0,
        }
    if name == "dqn_pareto":
        return {
            "name": name,
            "n_weights": len(tr.get("cumulative_points_per_weight") or []),
            "pareto_json_path": tr.get("pareto_json_path"),
            "time_axis_note": tr.get("time_axis_note"),
        }
    return {"name": name, "keys": list(tr.keys())}


def _dqn_trace_from_new_pareto_json(before: set, wall_sec: float) -> Optional[Dict[str, Any]]:
    """今回の実行で増えた pareto_data_distributed JSON から dqn_pareto 形式のトレースを作る。"""
    after = set(_pareto_json_paths())
    new_paths = sorted(after - before)
    if not new_paths:
        latest = sorted(after)
        path = latest[-1] if latest else None
    else:
        path = new_paths[-1]
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sols = sorted(data.get("all_solutions") or [], key=lambda s: s["solution_id"])
    cumulative = [[float(s["cost"]), float(s["waiting_time"])] for s in sols]
    n = len(cumulative)
    if n == 0:
        elapsed: List[float] = []
    else:
        elapsed = [wall_sec * (i + 1) / n for i in range(n)]
    return {
        "name": "dqn_pareto",
        "wall_total_s": wall_sec,
        "cumulative_points_per_weight": cumulative,
        "elapsed_seconds": elapsed,
        "pareto_json_path": path,
        "time_axis_note": "elapsed_seconds は各重みの終了時刻が無いため、wall 上で均等分割した近似",
    }


def _points_for_pf_plot(
    trace_nsga: Optional[Dict[str, Any]],
    trace_pcn: Optional[Dict[str, Any]],
    trace_dqn: Optional[Dict[str, Any]],
    key: str,
) -> Optional[np.ndarray]:
    """解空間の点 (n,2) = [cost, waiting_time]（最小化）。"""
    if key == "nsga2":
        tr = trace_nsga
        if not tr:
            return None
        fo = tr.get("final_objectives")
        if not fo:
            return None
        return np.asarray(fo, dtype=np.float64)
    if key == "pcn_distributed":
        tr = trace_pcn
        if not tr:
            return None
        ev = tr.get("pareto_fronts_per_eval") or []
        if not ev:
            return None
        return np.asarray(ev[-1], dtype=np.float64)
    if key == "dqn_pareto":
        tr = trace_dqn
        if not tr:
            return None
        pj = tr.get("pareto_json_path")
        if not pj or not Path(pj).is_file():
            return None
        with open(pj, encoding="utf-8") as f:
            data = json.load(f)
        pf = data.get("pareto_front") or []
        if not pf:
            return None
        return np.array([[float(p["cost"]), float(p["waiting_time"])] for p in pf], dtype=np.float64)
    return None


def save_pareto_front_figures(
    trace_nsga: Optional[Dict[str, Any]],
    trace_pcn: Optional[Dict[str, Any]],
    trace_dqn: Optional[Dict[str, Any]],
    base_path: Path,
) -> Dict[str, str]:
    """横軸: 待ち時間、縦軸: コスト（いずれも最小化が良い方向）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("nsga2", "NSGA-II", "#4477aa"),
        ("pcn_distributed", "PCN (distributed)", "#cc6677"),
        ("dqn_pareto", "DQN (weight sweep)", "#228833"),
    ]
    out: Dict[str, str] = {}
    overlay = base_path.with_name(base_path.stem + "_pareto_overlay.png")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key, label, color in specs:
        pts = _points_for_pf_plot(trace_nsga, trace_pcn, trace_dqn, key)
        if pts is None or pts.size == 0:
            continue
        ax.scatter(
            pts[:, 1],
            pts[:, 0],
            c=color,
            label=label,
            alpha=0.88,
            s=36,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_xlabel("Waiting time (minimize)")
    ax.set_ylabel("Cost (minimize)")
    ax.set_title("Pareto fronts (solution space)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(overlay, dpi=150)
    plt.close(fig)
    out["pareto_overlay_png"] = str(overlay.resolve())

    grid_path = base_path.with_name(base_path.stem + "_pareto_grid.png")
    fig2, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, label, color) in zip(axes, specs):
        pts = _points_for_pf_plot(trace_nsga, trace_pcn, trace_dqn, key)
        if pts is None or pts.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        else:
            ax.scatter(pts[:, 1], pts[:, 0], c=color, alpha=0.88, s=28)
        ax.set_title(label)
        ax.set_xlabel("Waiting time")
        ax.set_ylabel("Cost")
        ax.grid(True, alpha=0.35)
    fig2.suptitle("Pareto fronts per algorithm", y=1.02)
    fig2.tight_layout()
    fig2.savefig(grid_path, dpi=150)
    plt.close(fig2)
    out["pareto_grid_png"] = str(grid_path.resolve())
    return out


def main() -> None:
    mo_mod = _load_mo_benchmark_hv()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=os.environ.get("SCHEDULER_CONFIG", "config/config.yml"))
    pre_args, _ = pre.parse_known_args()
    pre_cfg_path = Path(pre_args.config)
    if not pre_cfg_path.is_absolute():
        pre_cfg_path = PROJECT_ROOT / pre_cfg_path

    parser = argparse.ArgumentParser()
    _base_cfg = load_full_config(str(pre_cfg_path))
    _pac = get_param_algorithm_compare(_base_cfg)
    _mb = _pac.get("micro_benchmark_wall_clock") or {}
    _ddp = _pac.get("dqn_distributed_pareto") or {}
    parser.add_argument("--nb_jobs", type=int, default=int(_pac.get("nb_jobs", 24)))
    parser.add_argument("--n_onprem", type=int, default=int(_pac.get("n_on_premise_node", 256)))
    parser.add_argument("--n_cloud", type=int, default=int(_pac.get("n_cloud_node", 1024)))
    parser.add_argument("--nsga_pop", type=int, default=int(_mb.get("nsga_pop", 5)))
    parser.add_argument("--nsga_gen", type=int, default=int(_mb.get("nsga_gen", 2)))
    parser.add_argument(
        "--nsga-n-workers",
        type=int,
        default=None,
        help="NSGA-II 並列評価（既定: mo_hypervolume_benchmark.nsga_n_workers または nsga2.n_workers）",
    )
    parser.add_argument("--dqn-weight-steps", type=int, default=None)
    parser.add_argument("--dqn-num-workers", type=int, default=None)
    parser.add_argument("--dqn-episodes-per-weight", type=int, default=None)
    parser.add_argument("--dqn_episodes", type=int, default=None)
    parser.add_argument("--pcn_event_obs", action="store_true")
    parser.add_argument("--config", type=str, default=str(pre_args.config))
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument(
        "--full-trace",
        action="store_true",
        help="results にパレート全配列を含める（既定は要約のみ）",
    )
    parser.add_argument(
        "--no-plot-pf",
        action="store_true",
        help="パレートフロント図（オーバーレイ・3分割）を保存しない",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_full_config(str(cfg_path))
    pac = get_param_algorithm_compare(cfg)
    pac["nb_jobs"] = args.nb_jobs
    pac["n_on_premise_node"] = args.n_onprem
    pac["n_cloud_node"] = args.n_cloud
    mb = pac.get("micro_benchmark_wall_clock") or {}
    mb["nsga_pop"] = args.nsga_pop
    mb["nsga_gen"] = args.nsga_gen
    pac["micro_benchmark_wall_clock"] = mb

    ddp = dict(pac.get("dqn_distributed_pareto") or {})
    weight_steps = args.dqn_weight_steps if args.dqn_weight_steps is not None else int(ddp.get("weight_steps", 10))
    num_workers = args.dqn_num_workers if args.dqn_num_workers is not None else int(ddp.get("num_workers", 4))
    epw = args.dqn_episodes_per_weight
    if epw is None:
        epw = args.dqn_episodes
    if epw is None:
        epw = int(ddp.get("episodes_per_weight", mb.get("dqn_episodes", 1000)))
    ddp["weight_steps"] = weight_steps
    ddp["num_workers"] = num_workers
    ddp["episodes_per_weight"] = epw
    pac["dqn_distributed_pareto"] = ddp

    cfg["param_algorithm_compare"] = pac
    if str(pac.get("env_backend", "c")).lower() == "python":
        print(
            "micro_benchmark_three_algorithms は分散PCN（C環境）を含むため "
            "env_backend=python では環境を揃えられません。",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg.setdefault("param_env", {})["n_on_premise_node"] = args.n_onprem
    cfg["param_env"]["n_cloud_node"] = args.n_cloud
    cfg["param_env"]["n_jobs"] = args.nb_jobs

    nsga = pac.get("nsga2") or {}
    mo = pac.get("mo_hypervolume_benchmark") or {}
    nsga_n_workers = args.nsga_n_workers
    if nsga_n_workers is None:
        nsga_n_workers = int(mo.get("nsga_n_workers", nsga.get("n_workers", -1)))

    fd, tmp_path = tempfile.mkstemp(suffix="_micro_bench.yml", text=True)
    os.close(fd)
    tmp_cfg = Path(tmp_path)
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    trace_nsga: Optional[Dict[str, Any]] = None
    trace_pcn: Optional[Dict[str, Any]] = None
    trace_dqn: Optional[Dict[str, Any]] = None
    nsga_err: Optional[str] = None
    dqn_proc: Any = None
    pcn_wall = nsga_wall = dqn_wall = 0.0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["SCHEDULER_CONFIG"] = str(tmp_cfg.resolve())
    env["DISTRIBUTED_PCN_CONFIG"] = str(tmp_cfg.resolve())
    env["DISTRIBUTED_PCN_JOBS"] = str(args.nb_jobs)
    env["DISTRIBUTED_PCN_ONPREM"] = str(args.n_onprem)
    env["DISTRIBUTED_PCN_CLOUD"] = str(args.n_cloud)
    dpc = pac.get("distributed_pcn") or {}
    if dpc.get("quick", True):
        env["DISTRIBUTED_PCN_QUICK"] = "1"
    else:
        env["DISTRIBUTED_PCN_QUICK"] = "0"
    if dpc.get("profile", True):
        env["DISTRIBUTED_PCN_PROFILE"] = "1"
    if args.pcn_event_obs or dpc.get("use_event_obs"):
        env["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1"
    else:
        env["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "0"

    use_event = args.pcn_event_obs or dpc.get("use_event_obs")
    pcn_module = "src.distributed.distributed_pcn_event" if use_event else "src.distributed.distributed_pcn"
    pcn_cmd = [sys.executable, "-m", pcn_module]

    dqn_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "main.py"),
        "--mode",
        "pareto_distributed",
        "--nb_jobs",
        str(args.nb_jobs),
        "--how_many_episodes",
        str(epw),
        "--weight_steps",
        str(weight_steps),
        "--num_workers",
        str(num_workers),
    ]

    quality_report: Dict[str, Any] = {}

    try:
        # 1) NSGA-II（プロセス内・HV 用トレース）
        try:
            trace_nsga = mo_mod.run_nsga2_trace(
                cfg,
                args.nb_jobs,
                args.nsga_pop,
                args.nsga_gen,
                nsga_n_workers,
            )
            nsga_wall = float(trace_nsga.get("wall_total_s", 0.0))
        except Exception as e:
            trace_nsga = None
            nsga_wall = 0.0
            nsga_err = str(e)

        # 2) PCN + MO-HV 書き出し（run_pcn_distributed_subprocess が1回で完結）
        fd_pcn, pcn_export_path = tempfile.mkstemp(suffix="_pcn_mo_hv.json", text=True)
        os.close(fd_pcn)
        pcn_export = Path(pcn_export_path)
        env_pcn: Dict[str, str] = {
            "DISTRIBUTED_PCN_JOBS": str(args.nb_jobs),
            "DISTRIBUTED_PCN_ONPREM": str(args.n_onprem),
            "DISTRIBUTED_PCN_CLOUD": str(args.n_cloud),
            "DISTRIBUTED_PCN_QUICK": env.get("DISTRIBUTED_PCN_QUICK", "0"),
            "DISTRIBUTED_PCN_PROFILE": "1" if dpc.get("profile", True) else "0",
            "DISTRIBUTED_PCN_USE_EVENT_OBS": "1" if use_event else "0",
        }
        trace_pcn = mo_mod.run_pcn_distributed_subprocess(str(tmp_cfg.resolve()), pcn_export, env_pcn)
        pcn_wall = float(trace_pcn.get("wall_total_s", 0.0))
        try:
            pcn_export.unlink(missing_ok=True)
        except OSError:
            pass

        # 3) DQN 分散パレート（重みスイープ）
        pareto_before = set(_pareto_json_paths())
        dqn_proc, dqn_wall = run_subprocess(dqn_cmd, env)
        if dqn_proc.returncode == 0:
            trace_dqn = _dqn_trace_from_new_pareto_json(pareto_before, dqn_wall)

        # 4) 共通参照点での HV / TTS
        traces_list: List[Dict[str, Any]] = [t for t in (trace_nsga, trace_pcn, trace_dqn) if t is not None]
        quality_report = mo_mod.build_quality_report_from_traces(traces_list, pac, str(cfg_path.resolve()))

    finally:
        try:
            tmp_cfg.unlink(missing_ok=True)
        except OSError:
            pass

    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT
        / "0403"
        / "raw"
        / f"micro_benchmark_three_algorithms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pareto_plots: Dict[str, str] = {}
    if not args.no_plot_pf:
        try:
            pareto_plots = save_pareto_front_figures(trace_nsga, trace_pcn, trace_dqn, out_path)
            print(f"Pareto plots: {pareto_plots}", file=sys.stderr)
        except Exception as e:
            pareto_plots = {"error": str(e)}
            print(f"[plot] {e}", file=sys.stderr)

    out = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(cfg_path),
        "param_algorithm_compare_used": pac,
        "program_state": {
            "entrypoint": "run_nsga2_trace + run_pcn_distributed_subprocess + main pareto_distributed",
            "python": sys.version.split()[0],
            "cwd": str(PROJECT_ROOT),
        },
        "workload": {
            "nb_jobs": args.nb_jobs,
            "n_on_premise_node": args.n_onprem,
            "n_cloud_node": args.n_cloud,
        },
        "quality_metrics": quality_report,
        "pareto_plots": pareto_plots,
        "results": {
            "nsga2": {
                "wall_sec": nsga_wall,
                "trace": trace_nsga if args.full_trace else _summarize_trace_for_json(trace_nsga),
                "error": nsga_err,
            },
            "pcn_distributed": {
                "wall_sec": pcn_wall,
                "returncode": trace_pcn.get("returncode") if trace_pcn else None,
                "argv": pcn_cmd,
                "event_obs": bool(use_event),
                "trace": trace_pcn if args.full_trace else _summarize_trace_for_json(trace_pcn),
            },
            "dqn_distributed_pareto": {
                "wall_sec": dqn_wall,
                "returncode": dqn_proc.returncode if dqn_proc else None,
                "argv": dqn_cmd,
                "weight_steps": weight_steps,
                "episodes_per_weight": epw,
                "num_workers": num_workers,
                "trace": trace_dqn if args.full_trace else _summarize_trace_for_json(trace_dqn),
                "stderr_tail": (dqn_proc.stderr or "")[-2000:] if dqn_proc else "",
            },
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n保存: {out_path}")

    ok = (
        trace_nsga is not None
        and trace_pcn is not None
        and trace_pcn.get("returncode") == 0
        and dqn_proc is not None
        and dqn_proc.returncode == 0
    )
    if not ok:
        print("--- 一部失敗（quality_metrics を確認） ---", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
