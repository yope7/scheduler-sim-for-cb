#!/usr/bin/env python3
"""Eval PF を参照（全探索サンプル）PF と比較し、劣解率・平均 gap をスコア化する。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.pcn_agent import get_non_dominated_inds_minimize


def _nd(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    nd = get_non_dominated_inds_minimize(points)
    return points[nd] if len(nd) else points


def _dedupe_points(points: np.ndarray, *, decimals: int = 3) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.unique(np.round(points, decimals=decimals), axis=0)


def mean_archive_gap(eval_pf: np.ndarray, ref_pf: np.ndarray) -> dict:
    """eval 各点について ref PF 上の best wait との差（正=劣解）。"""
    if eval_pf.size == 0 or ref_pf.size == 0:
        return {"n": 0, "mean_gap": float("nan"), "max_gap": float("nan"), "frac_bad": float("nan")}
    gaps = []
    for c, w in eval_pf:
        better = ref_pf[ref_pf[:, 0] <= c + 1e-6]
        if not better.size:
            best_w = float(ref_pf[:, 1].min())
        else:
            best_w = float(better[:, 1].min())
        gaps.append(float(w - best_w))
    gaps_arr = np.asarray(gaps, dtype=np.float64)
    thr = max(20.0, float(np.percentile(ref_pf[:, 1], 10)) * 0.05)
    return {
        "n": int(len(gaps_arr)),
        "mean_gap": float(gaps_arr.mean()),
        "max_gap": float(gaps_arr.max()),
        "frac_bad": float((gaps_arr > thr).mean()),
        "gap_threshold": thr,
    }


def command_achievability_stats(
    achieved_pts: np.ndarray,
    command_targets: np.ndarray,
    *,
    low_wait_max: float = 0.0,
    wait_gap_tolerance: float = 120.0,
) -> dict:
    """PF command の target 点に対する到達 wait gap を集計する。"""
    achieved = np.asarray(achieved_pts, dtype=np.float64)
    targets = np.asarray(command_targets, dtype=np.float64)
    if achieved.size == 0 or targets.size == 0:
        return {
            "n_command_pairs": 0,
            "command_mean_wait_gap": float("nan"),
            "command_frac_wait_bad": float("nan"),
        }
    n = min(len(achieved), len(targets))
    achieved = achieved[:n]
    targets = targets[:n]
    wait_gap = achieved[:, 1] - targets[:, 1]
    cost_gap = achieved[:, 0] - targets[:, 0]
    bad = wait_gap > float(wait_gap_tolerance)
    stats = {
        "n_command_pairs": int(n),
        "command_mean_wait_gap": float(wait_gap.mean()),
        "command_mean_positive_wait_gap": float(np.maximum(wait_gap, 0.0).mean()),
        "command_max_wait_gap": float(wait_gap.max()),
        "command_mean_cost_gap": float(cost_gap.mean()),
        "command_frac_wait_bad": float(bad.mean()),
        "command_wait_gap_tolerance": float(wait_gap_tolerance),
    }
    if low_wait_max > 0:
        mask = targets[:, 1] < float(low_wait_max)
        low_n = int(mask.sum())
        if low_n > 0:
            low_achieved = achieved[mask]
            low_targets = targets[mask]
            low_gap = low_achieved[:, 1] - low_targets[:, 1]
            stats.update(
                {
                    "low_wait_max": float(low_wait_max),
                    "low_wait_target_n": low_n,
                    "low_wait_achieved_n": int((low_achieved[:, 1] < float(low_wait_max)).sum()),
                    "low_wait_covered_frac": float(
                        (low_achieved[:, 1] < float(low_wait_max)).mean()
                    ),
                    "low_wait_mean_gap": float(low_gap.mean()),
                    "low_wait_mean_positive_gap": float(np.maximum(low_gap, 0.0).mean()),
                    "low_wait_min_achieved": float(low_achieved[:, 1].min()),
                    "low_wait_target_min": float(low_targets[:, 1].min()),
                }
            )
        else:
            stats.update(
                {
                    "low_wait_max": float(low_wait_max),
                    "low_wait_target_n": 0,
                    "low_wait_achieved_n": 0,
                    "low_wait_covered_frac": float("nan"),
                    "low_wait_mean_gap": float("nan"),
                }
            )
    return stats


def score_eval_vs_reference(
    eval_pts: np.ndarray,
    ref_pts: np.ndarray,
    *,
    mean_gap_max: float = 120.0,
    frac_bad_max: float = 0.35,
    min_unique_pf: int = 8,
    min_cost_span_frac: float = 0.05,
    eval_is_pf_commands: bool = False,
    command_targets: Optional[np.ndarray] = None,
    low_wait_max: float = 0.0,
    min_low_wait_covered_frac: float = 0.0,
    command_wait_gap_tolerance: float = 120.0,
) -> dict:
    eval_arr = np.asarray(eval_pts, dtype=np.float64)
    if eval_is_pf_commands:
        eval_pf_unique = _dedupe_points(eval_arr)
        eval_pf = eval_pf_unique
    else:
        eval_pf = _nd(eval_arr)
        eval_pf_unique = _dedupe_points(eval_pf)
    ref_pf = _nd(np.asarray(ref_pts, dtype=np.float64))
    stats = mean_archive_gap(eval_pf_unique, ref_pf)
    ref_cmax = float(np.max(ref_pf[:, 0])) if len(ref_pf) else 0.0
    cost_span = (
        float(eval_pf_unique[:, 0].max() - eval_pf_unique[:, 0].min())
        if len(eval_pf_unique)
        else 0.0
    )
    cost_span_ok = ref_cmax <= 0.0 or cost_span >= min_cost_span_frac * ref_cmax
    achievability = (
        command_achievability_stats(
            eval_arr,
            np.asarray(command_targets, dtype=np.float64),
            low_wait_max=low_wait_max,
            wait_gap_tolerance=command_wait_gap_tolerance,
        )
        if command_targets is not None
        else {}
    )
    low_wait_ok = True
    if low_wait_max > 0 and min_low_wait_covered_frac > 0 and achievability:
        low_n = int(achievability.get("low_wait_target_n", 0))
        covered = float(achievability.get("low_wait_covered_frac", float("nan")))
        low_wait_ok = low_n <= 0 or (np.isfinite(covered) and covered >= min_low_wait_covered_frac)
    passed = (
        stats["n"] > 0
        and np.isfinite(stats["mean_gap"])
        and stats["mean_gap"] <= mean_gap_max
        and stats["frac_bad"] <= frac_bad_max
        and len(eval_pf_unique) >= min_unique_pf
        and cost_span_ok
        and low_wait_ok
    )
    return {
        "passed": bool(passed),
        "eval_pf_n": int(len(eval_pf)),
        "eval_pf_unique_n": int(len(eval_pf_unique)),
        "ref_pf_n": int(len(ref_pf)),
        "cost_span": cost_span,
        **stats,
        **achievability,
        "criteria": {
            "mean_gap_max": mean_gap_max,
            "frac_bad_max": frac_bad_max,
            "min_unique_pf": min_unique_pf,
            "min_cost_span_frac": min_cost_span_frac,
            "low_wait_max": low_wait_max,
            "min_low_wait_covered_frac": min_low_wait_covered_frac,
            "command_wait_gap_tolerance": command_wait_gap_tolerance,
        },
    }


def load_npz_pf(path: Path, key: str = "pareto_front") -> np.ndarray:
    data = np.load(path)
    if key in data:
        return np.asarray(data[key], dtype=np.float64)
    if "points" in data:
        pts = np.asarray(data["points"], dtype=np.float64)
        return _nd(pts)
    raise KeyError(f"{path} has no {key} or points")


def load_npz_optional_array(path: Path, key: str) -> Optional[np.ndarray]:
    data = np.load(path)
    if key not in data:
        return None
    return np.asarray(data[key], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-npz", type=Path, required=True, help="Eval 点群 npz または --eval-png-dir から latest")
    p.add_argument("--ref-npz", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--mean-gap-max", type=float, default=120.0)
    p.add_argument("--frac-bad-max", type=float, default=0.35)
    p.add_argument(
        "--low-wait-max",
        type=float,
        default=float(os.environ.get("PCN_SCORE_LOW_WAIT_MAX", "0")),
        help="PF command target の低 wait 閾値（0=achievability 合否なし）",
    )
    p.add_argument(
        "--min-low-wait-covered-frac",
        type=float,
        default=float(os.environ.get("PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC", "0")),
        help="target wait < low-wait-max のうち achieved も閾値未満になる最低割合",
    )
    p.add_argument(
        "--command-wait-gap-tolerance",
        type=float,
        default=float(os.environ.get("PCN_SCORE_COMMAND_WAIT_GAP_TOL", "120")),
    )
    args = p.parse_args()
    eval_pts = load_npz_pf(args.eval_npz)
    ref_pts = load_npz_pf(args.ref_npz)
    command_targets = load_npz_optional_array(args.eval_npz, "command_targets")
    eval_is_pf_commands = False
    if command_targets is not None:
        achieved_points = load_npz_optional_array(args.eval_npz, "points")
        if achieved_points is not None and len(achieved_points) == len(command_targets):
            eval_pts = achieved_points
            eval_is_pf_commands = True
    result = score_eval_vs_reference(
        eval_pts,
        ref_pts,
        mean_gap_max=args.mean_gap_max,
        frac_bad_max=args.frac_bad_max,
        eval_is_pf_commands=eval_is_pf_commands,
        command_targets=command_targets,
        low_wait_max=args.low_wait_max,
        min_low_wait_covered_frac=args.min_low_wait_covered_frac,
        command_wait_gap_tolerance=args.command_wait_gap_tolerance,
    )
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
