"""Eval PF と Archive PF の帯域ギャップ検出 → 学習 replay 重みブースト用。"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.agents.pcn_agent import get_non_dominated_inds_minimize

# analyze_pf_bulge / pf_left_tail_goal と同系の帯域
DEFAULT_GAP_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("low", 0.0, 5e5),
    ("low_slope", 0.0, 1.2e6),
    ("mid_1e6", 5e5, 1.5e6),
    ("knee", 1.5e6, 2.5e6),
)

# 左上先端向け FB: 膝・中域まで一括ブーストすると全体 PF が崩れやすい
LEFT_TAIL_GAP_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("low", 0.0, 5e5),
    ("low_slope", 0.0, 1.2e6),
)


def gap_bands_from_env(
    bands: Sequence[Tuple[str, float, float]] = DEFAULT_GAP_BANDS,
) -> Tuple[Tuple[str, float, float], ...]:
    """PCN_EVAL_GAP_BAND_NAMES + PCN_EVAL_GAP_BANDS_FRAC で cost 帯をワークロード相対指定。"""
    frac_raw = os.environ.get("PCN_EVAL_GAP_BANDS_FRAC", "").strip()
    names_raw = os.environ.get("PCN_EVAL_GAP_BAND_NAMES", "").strip()
    cost_max = float(os.environ.get("PCN_WORKLOAD_COST_MAX", "0") or "0")
    if frac_raw and names_raw and cost_max > 0:
        fracs = [float(x.strip()) for x in frac_raw.split(",") if x.strip()]
        names = [n.strip() for n in names_raw.split(",") if n.strip()]
        if len(fracs) >= 2 and len(fracs) % 2 == 0:
            picked: List[Tuple[str, float, float]] = []
            for i, name in enumerate(names):
                if 2 * i + 1 >= len(fracs):
                    break
                lo = cost_max * float(fracs[2 * i])
                hi = cost_max * float(fracs[2 * i + 1])
                picked.append((name, lo, hi))
            if picked:
                return tuple(picked)
    raw = os.environ.get("PCN_EVAL_GAP_BAND_NAMES", "").strip()
    if not raw:
        if os.environ.get("PCN_EVAL_GAP_LEFT_TAIL_ONLY", "0") == "1":
            return LEFT_TAIL_GAP_BANDS
        return tuple(bands)
    want = {n.strip() for n in raw.split(",") if n.strip()}
    picked = tuple(b for b in bands if b[0] in want)
    return picked if picked else LEFT_TAIL_GAP_BANDS


def pf_gap_in_band(
    eval_pf: np.ndarray,
    archive_pf: np.ndarray,
    cost_lo: float,
    cost_hi: float,
) -> Dict[str, Any]:
    if not eval_pf.size or not archive_pf.size:
        return {"n": 0, "mean_gap": float("nan"), "max_gap": float("nan")}
    gaps = []
    for c, w in eval_pf:
        if not (cost_lo <= c <= cost_hi):
            continue
        band = archive_pf[(archive_pf[:, 0] >= cost_lo) & (archive_pf[:, 0] <= cost_hi)]
        if not band.size:
            band = archive_pf
        gaps.append(float(w - band[:, 1].min()))
    if not gaps:
        return {"n": 0, "mean_gap": float("nan"), "max_gap": float("nan")}
    gaps_arr = np.asarray(gaps, dtype=np.float64)
    return {
        "n": int(len(gaps)),
        "mean_gap": float(gaps_arr.mean()),
        "max_gap": float(gaps_arr.max()),
    }


def summarize_band_gaps(
    eval_pf: np.ndarray,
    archive_pf: np.ndarray,
    bands: Sequence[Tuple[str, float, float]] = DEFAULT_GAP_BANDS,
) -> Dict[str, Dict[str, Any]]:
    return {
        name: pf_gap_in_band(eval_pf, archive_pf, lo, hi) for name, lo, hi in bands
    }


def gap_band_boosts(
    gap_summary: Dict[str, Dict[str, Any]],
    bands: Sequence[Tuple[str, float, float]] = DEFAULT_GAP_BANDS,
    *,
    ref_gap: Optional[float] = None,
    max_boost: Optional[float] = None,
    min_boost: float = 1.0,
) -> List[Tuple[float, float, float]]:
    """mean_gap > ref_gap の帯域に multiplier を付与。(cost_lo, cost_hi, mult)。"""
    ref_frac = os.environ.get("PCN_EVAL_GAP_REF_FRAC", "").strip()
    wait_max = float(os.environ.get("PCN_WORKLOAD_WAIT_MAX", "0") or "0")
    if ref_frac and wait_max > 0:
        ref = float(ref_frac) * wait_max
    else:
        ref = float(ref_gap if ref_gap is not None else os.environ.get("PCN_EVAL_GAP_REF_GAP", "1200"))
    cap = float(max_boost if max_boost is not None else os.environ.get("PCN_EVAL_GAP_BOOST_MAX", "3.0"))
    boosts: List[Tuple[float, float, float]] = []
    for name, lo, hi in bands:
        st = gap_summary.get(name, {})
        n = int(st.get("n", 0))
        mg = st.get("mean_gap", float("nan"))
        if n <= 0 or not np.isfinite(mg) or mg <= ref:
            continue
        mult = min(cap, min_boost + (mg - ref) / max(ref, 1.0))
        boosts.append((float(lo), float(hi), float(mult)))
    return boosts


def exploration_points_from_replay_entries(entries) -> np.ndarray:
    """replay 各エピソードの (cost, avgwait)。"""
    points = []
    for entry in entries:
        episode = entry[2]
        if not episode:
            continue
        first = episode[0]
        if hasattr(first, "objective_values") and first.objective_values is not None:
            obj = first.objective_values
            points.append([float(obj[0]), float(obj[2])])
        else:
            r = np.asarray(first.reward, dtype=np.float64)
            if r.size >= 2:
                points.append([float(-r[1]), float(-r[0])])
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def archive_pf_from_replay_entries(entries) -> np.ndarray:
    """experience_replay エントリから Archive 相当の (cost, wait) 点列。"""
    points = []
    for entry in entries:
        episode = entry[2]
        if not episode:
            continue
        first = episode[0]
        if hasattr(first, "objective_values") and first.objective_values is not None:
            obj = first.objective_values
            points.append([float(obj[0]), float(obj[2])])
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    nd = get_non_dominated_inds_minimize(pts)
    return pts[nd] if len(nd) else pts


def _uniform_grid_commands(
    agent,
    n_jobs: int,
    grid: int,
    extend: float,
    *,
    low_tail_frac: Optional[float] = None,
    low_tail_extra: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, float]], np.ndarray, np.ndarray, np.ndarray]:
    """均等格子の command 列と archive/exploration 点を返す。"""
    from src.utils.pf_uniform_plot import build_r1_grid, live_uniform_pf_low_tail

    entries = agent._valid_replay_entries()
    exploration = exploration_points_from_replay_entries(entries)
    ref_pts = archive_pf_from_replay_entries(entries)
    if ref_pts.size:
        cost_max = float(ref_pts[:, 0].max())
        wait_max = float(ref_pts[:, 1].max())
    else:
        target = agent.network if agent.use_enhanced_model else agent.model
        scale = target.desired_return_scale.detach().cpu().numpy().astype(np.float64)
        cost_max = float(scale[1])
        wait_max = float(scale[0]) / max(1, n_jobs)

    if low_tail_frac is None or low_tail_extra is None:
        ltf, lte = live_uniform_pf_low_tail()
        low_tail_frac = ltf if low_tail_frac is None else low_tail_frac
        low_tail_extra = lte if low_tail_extra is None else low_tail_extra

    r1 = build_r1_grid(
        cost_max,
        extend,
        grid,
        low_tail_frac=float(low_tail_frac),
        low_tail_extra=int(low_tail_extra),
    )
    r0 = np.linspace(0.0, -wait_max * n_jobs * extend, grid)
    commands: List[Tuple[np.ndarray, float]] = []
    for r0v in r0:
        for r1v in r1:
            dr = np.array([float(r0v), float(r1v)], dtype=np.float32)
            commands.append((dr, float(n_jobs)))
    return commands, ref_pts, exploration, r0


def _run_uniform_grid_commands(
    agent,
    env,
    commands: Sequence[Tuple[np.ndarray, float]],
    actors: Optional[Sequence[Any]] = None,
    weights_ref: Optional[Any] = None,
) -> np.ndarray:
    """格子 command を評価。actors があれば Ray Actor に分割（結果順序は commands と同じ）。"""
    if not commands:
        return np.empty((0, 2), dtype=np.float64)

    use_actors = (
        actors is not None
        and len(actors) > 0
        and os.environ.get("DISTRIBUTED_PCN_UNIFORM_PF_DISTRIBUTED", "1") == "1"
    )
    if use_actors:
        import ray

        n_actors = len(actors)
        chunks: List[List[Tuple[list, float]]] = [[] for _ in range(n_actors)]
        index_map: List[Tuple[int, int]] = []
        for i, (dr, hz) in enumerate(commands):
            slot = i % n_actors
            local_i = len(chunks[slot])
            chunks[slot].append((dr.tolist(), float(hz)))
            index_map.append((slot, local_i))
        futures = []
        for slot, chunk in enumerate(chunks):
            if chunk:
                futures.append(
                    (
                        slot,
                        actors[slot].eval_uniform_grid_batch.remote(chunk, weights_ref),
                    )
                )
        slot_results: Dict[int, List[List[float]]] = {}
        for slot, fut in futures:
            slot_results[slot] = ray.get(fut)
        achieved = [None] * len(commands)
        for i, (slot, local_i) in enumerate(index_map):
            achieved[i] = slot_results[slot][local_i]
        return np.asarray(achieved, dtype=np.float64)

    if agent is None or env is None:
        raise ValueError("serial uniform grid requires agent and env")
    max_return = np.full(2, np.inf, dtype=np.float32)
    achieved = []
    for dr, hz in commands:
        _, _, _, _, _, val = agent._run_episode(
            env, dr.copy(), np.float32(hz), max_return, eval_mode=True
        )
        achieved.append([float(val[0]), float(val[1])])
    return np.asarray(achieved, dtype=np.float64)


def uniform_command_eval_pf(
    agent,
    env,
    n_jobs: int,
    grid: int = 12,
    extend: float = 1.10,
    *,
    low_tail_frac: Optional[float] = None,
    low_tail_extra: Optional[int] = None,
    actors: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """均等 command 格子で Eval PF を得る。戻り値: (eval_pf, archive_pf, all_pts, exploration)。"""
    commands, ref_pts, exploration, _r0 = _uniform_grid_commands(
        agent,
        n_jobs,
        grid,
        extend,
        low_tail_frac=low_tail_frac,
        low_tail_extra=low_tail_extra,
    )
    pts = _run_uniform_grid_commands(agent, env, commands, actors=actors)
    nd = get_non_dominated_inds_minimize(pts)
    eval_pf = pts[nd] if len(nd) else pts
    return eval_pf, ref_pts, pts, exploration


def _maybe_write_live_uniform_pf(
    *,
    plot_dir: Optional[str],
    plot_iteration: Optional[int],
    plot_label: Optional[str],
    all_pts: np.ndarray,
    eval_pf: np.ndarray,
    archive_pf: np.ndarray,
    exploration: np.ndarray,
) -> Optional[str]:
    from src.utils.pf_uniform_plot import (
        live_uniform_pf_enabled,
        print_live_pf_saved,
        write_uniform_pf_plot,
    )

    if not live_uniform_pf_enabled() or not plot_dir:
        return None
    label = plot_label or os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LABEL", "live")
    path = write_uniform_pf_plot(
        all_pts,
        archive_pf,
        exploration,
        plot_dir,
        label,
        iteration=plot_iteration,
    )
    print_live_pf_saved(path, plot_iteration)
    return path


def compute_eval_gap_feedback(
    agent,
    env,
    n_jobs: int,
    *,
    grid: Optional[int] = None,
    bands: Optional[Sequence[Tuple[str, float, float]]] = None,
    plot_dir: Optional[str] = None,
    plot_iteration: Optional[int] = None,
    plot_label: Optional[str] = None,
    actors: Optional[Sequence[Any]] = None,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, Dict[str, Any]], Optional[str]]:
    g = int(grid if grid is not None else os.environ.get("PCN_EVAL_GAP_FEEDBACK_GRID", "12"))
    band_list = gap_bands_from_env() if bands is None else tuple(bands)
    eval_pf, archive_pf, all_pts, exploration = uniform_command_eval_pf(
        agent, env, n_jobs, grid=g, actors=actors
    )
    summary = summarize_band_gaps(eval_pf, archive_pf, band_list)
    boosts = gap_band_boosts(summary, band_list)
    plot_path = _maybe_write_live_uniform_pf(
        plot_dir=plot_dir,
        plot_iteration=plot_iteration,
        plot_label=plot_label,
        all_pts=all_pts,
        eval_pf=eval_pf,
        archive_pf=archive_pf,
        exploration=exploration,
    )
    return boosts, summary, plot_path


def save_live_uniform_pf_plot(
    agent,
    env,
    n_jobs: int,
    plot_dir: str,
    *,
    iteration: Optional[int] = None,
    plot_label: Optional[str] = None,
    grid: Optional[int] = None,
    actors: Optional[Sequence[Any]] = None,
) -> Optional[str]:
    """Eval ギャップ FB なしでも均等格子 PF 図だけ保存。"""
    from src.utils.pf_uniform_plot import live_uniform_pf_enabled, live_uniform_pf_grid

    if not live_uniform_pf_enabled():
        return None
    g = int(grid if grid is not None else live_uniform_pf_grid())
    eval_pf, archive_pf, all_pts, exploration = uniform_command_eval_pf(
        agent, env, n_jobs, grid=g, actors=actors
    )
    return _maybe_write_live_uniform_pf(
        plot_dir=plot_dir,
        plot_iteration=iteration,
        plot_label=plot_label,
        all_pts=all_pts,
        eval_pf=eval_pf,
        archive_pf=archive_pf,
        exploration=exploration,
    )
