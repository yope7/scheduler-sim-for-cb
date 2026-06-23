"""PF 上の (cost, wait) を desired_return command に変換し、PF 近傍 command Eval を行う。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.agents.pcn_agent import get_non_dominated_inds_minimize


def objectives_to_command(cost: float, avg_wait: float, n_jobs: int) -> np.ndarray:
    nj = max(1, int(n_jobs))
    return np.array([-float(avg_wait) * nj, -float(cost)], dtype=np.float32)


def dedupe_pf(points: np.ndarray, *, decimals: int = 3) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    nd = get_non_dominated_inds_minimize(pts)
    pf = pts[nd] if len(nd) else pts
    return np.unique(np.round(pf, decimals=decimals), axis=0)


def load_ref_pf(path: Path, key: str = "pareto_front") -> np.ndarray:
    data = np.load(path)
    if key in data:
        pts = np.asarray(data[key], dtype=np.float64)
    elif "points" in data:
        pts = np.asarray(data["points"], dtype=np.float64)
    else:
        raise KeyError(f"{path} has no {key} or points")
    return dedupe_pf(pts)


def pf_points_to_commands(
    pf: np.ndarray,
    n_jobs: int,
    *,
    jitter_frac: float = 0.0,
    dedupe: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, float]]:
    """PF 点ごとに (desired_return, horizon=n_jobs) を生成。"""
    pf = dedupe_pf(np.asarray(pf, dtype=np.float64)) if dedupe else np.asarray(pf, dtype=np.float64)
    if pf.size == 0:
        return []
    rng = rng or np.random.default_rng()
    hz = float(max(1, int(n_jobs)))
    out: List[Tuple[np.ndarray, float]] = []
    for cost, wait in pf:
        c, w = float(cost), float(wait)
        if jitter_frac > 0:
            c *= float(rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac))
            w *= float(rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac))
            c = max(0.0, c)
            w = max(0.0, w)
        dr = objectives_to_command(c, w, n_jobs)
        out.append((dr, hz))
    return out


def _stratified_sample_axis(
    pf: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    axis: int,
) -> np.ndarray:
    if pf.size == 0 or n <= 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(pf) == 1:
        return np.repeat(pf, n, axis=0)
    order = np.argsort(pf[:, axis])
    ordered = pf[order]
    edges = np.linspace(0, len(ordered), n + 1, dtype=int)
    picks = []
    for i in range(n):
        lo = int(edges[i])
        hi = int(max(edges[i + 1], lo + 1))
        hi = min(hi, len(ordered))
        picks.append(ordered[int(rng.integers(lo, hi))])
    return np.asarray(picks, dtype=np.float64)


def _low_wait_subset(
    pf: np.ndarray,
    *,
    low_wait_frac: float,
    low_wait_max: float,
) -> np.ndarray:
    if pf.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if low_wait_max > 0:
        band = pf[pf[:, 1] <= float(low_wait_max)]
        if band.size:
            return band
    if low_wait_frac <= 0:
        return np.empty((0, 2), dtype=np.float64)
    n_tail = max(1, int(np.ceil(len(pf) * min(float(low_wait_frac), 1.0))))
    order = np.argsort(pf[:, 1])
    return pf[order[:n_tail]]


def _append_unique(rows: List[np.ndarray], seen: set, candidates: np.ndarray) -> None:
    for row in np.asarray(candidates, dtype=np.float64).reshape(-1, 2):
        key = tuple(np.round(row, decimals=3))
        if key in seen:
            continue
        rows.append(row)
        seen.add(key)


def stratified_sample_pf(
    pf: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None,
    *,
    low_wait_frac: float = 0.0,
    low_wait_quota: int = 0,
    low_wait_max: float = 0.0,
    include_extremes: bool = True,
) -> np.ndarray:
    """PF から command 用の点を層化サンプルする。

    既定は cost 軸の層化。low_wait_* を指定すると、低 wait 側にも batch 内の
    予約枠を持たせる。これは cost 層化だけでは高 cost/低 wait の command が
    薄くなり、方策が少数 attractor に落ちるケースの診断・学習用。
    """
    pf = dedupe_pf(np.asarray(pf, dtype=np.float64))
    if pf.size == 0 or n <= 0:
        return np.empty((0, 2), dtype=np.float64)
    rng = rng or np.random.default_rng()
    if len(pf) == 1:
        return np.repeat(pf, n, axis=0)

    rows: List[np.ndarray] = []
    seen: set = set()
    if include_extremes:
        extreme_idx = [int(np.argmin(pf[:, 0])), int(np.argmin(pf[:, 1]))]
        _append_unique(rows, seen, pf[np.unique(extreme_idx)])

    low_quota = max(int(low_wait_quota), 0)
    if low_wait_frac > 0:
        low_quota = max(low_quota, int(np.ceil(float(n) * float(low_wait_frac))))
    low_quota = min(low_quota, n)
    if low_quota > 0 and len(rows) < n:
        low_band = _low_wait_subset(
            pf,
            low_wait_frac=max(float(low_wait_frac), 0.0),
            low_wait_max=max(float(low_wait_max), 0.0),
        )
        low_take = max(0, min(low_quota, n - len(rows)))
        _append_unique(
            rows,
            seen,
            _stratified_sample_axis(low_band, low_take, rng, axis=0),
        )

    remaining = max(0, n - len(rows))
    _append_unique(
        rows,
        seen,
        _stratified_sample_axis(pf, remaining, rng, axis=0),
    )

    if len(rows) < n:
        fill = _stratified_sample_axis(pf, n - len(rows), rng, axis=0)
        rows.extend([row for row in fill])
    return np.asarray(rows[:n], dtype=np.float64)


def merge_pf_pools(*pools: np.ndarray) -> np.ndarray:
    if not pools:
        return np.empty((0, 2), dtype=np.float64)
    parts = [dedupe_pf(p) for p in pools if p is not None and np.asarray(p).size]
    if not parts:
        return np.empty((0, 2), dtype=np.float64)
    return dedupe_pf(np.vstack(parts))


def subsample_pf_stratified(pf: np.ndarray, max_points: int) -> np.ndarray:
    """cost 軸で層化し max_points まで間引く。"""
    return stratified_sample_pf(pf, max_points)
