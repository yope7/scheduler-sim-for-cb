"""均等 command PF 図（eval_uniform_command_pf と同系）。"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.pcn_agent import get_non_dominated_inds_minimize


def live_uniform_pf_enabled() -> bool:
    return os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF", "0") == "1"


def build_r1_grid(
    cost_max: float,
    extend: float,
    grid: int,
    *,
    focus_frac: float = 0.0,
    focus_extra: int = 0,
    focus_half_width_frac: float = 0.04,
    low_tail_frac: float = 0.0,
    low_tail_extra: int = 0,
) -> np.ndarray:
    r1 = np.linspace(0.0, -cost_max * extend, grid)
    if focus_frac > 0 and focus_extra > 0:
        center = -focus_frac * cost_max
        half = focus_half_width_frac * cost_max
        extra = np.linspace(center - half, center + half, focus_extra)
        r1 = np.concatenate([r1, extra])
    if low_tail_frac > 0 and low_tail_extra > 0:
        tail = np.linspace(0.0, -low_tail_frac * cost_max * extend, low_tail_extra)
        r1 = np.concatenate([r1, tail])
    return np.unique(r1).astype(np.float64)


def live_uniform_pf_grid() -> int:
    return int(os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_GRID", "12"))


def live_uniform_pf_low_tail() -> Tuple[float, int]:
    frac = float(os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LOW_TAIL_FRAC", "0.18"))
    extra = int(os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LOW_TAIL_EXTRA", "16"))
    return frac, extra


def write_uniform_pf_plot(
    achieved_pts: np.ndarray,
    archive_pf: np.ndarray,
    exploration_pts: np.ndarray,
    out_dir: str | Path,
    label: str,
    *,
    iteration: Optional[int] = None,
    also_latest: bool = True,
) -> str:
    """均等 command 到達点の PF 図を保存。iter 指定時は固定名も書く。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(achieved_pts, dtype=np.float64)
    if pts.size == 0:
        pts = np.empty((0, 2), dtype=np.float64)
    nd = get_non_dominated_inds_minimize(pts) if pts.size else np.array([], dtype=int)
    pf = pts[nd] if len(nd) else pts

    fig, ax = plt.subplots(figsize=(8, 6))
    if exploration_pts.size:
        ax.scatter(
            exploration_pts[:, 0],
            exploration_pts[:, 1],
            c="cyan",
            s=14,
            alpha=0.35,
            marker="o",
            label=f"Exploration ({len(exploration_pts)})",
            zorder=2,
        )
    if archive_pf.size:
        ax.scatter(
            archive_pf[:, 0],
            archive_pf[:, 1],
            c="cyan",
            s=40,
            marker="D",
            edgecolors="teal",
            linewidths=0.4,
            alpha=0.9,
            label=f"Archive PF ({len(archive_pf)})",
            zorder=4,
        )
    if pts.size:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c="blue",
            s=24,
            alpha=0.4,
            label=f"Achieved uniform ({len(pts)})",
            zorder=3,
        )
    if pf.size:
        order = np.lexsort((pf[:, 1], pf[:, 0]))
        ax.plot(
            pf[order, 0],
            pf[order, 1],
            "r-o",
            lw=1.6,
            ms=5,
            label=f"Eval PF ({len(nd)})",
            zorder=5,
        )
    ax.set_xlabel("Cost")
    ax.set_ylabel("Average Waiting Time")
    iter_s = f" iter {iteration}" if iteration is not None else ""
    ax.set_title(f"Uniform-command PF — {label}{iter_s}\nPF={len(nd)}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if iteration is not None:
        path = out / f"uniform_cmd_pf_iter_{int(iteration):03d}.png"
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"uniform_cmd_pf_{label}_{ts}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    if also_latest:
        latest = out / "uniform_cmd_pf_latest.png"
        shutil.copy2(path, latest)

    # 計装: 到達点の「座標」を npz でも残す（PNG は n_pf しか残さないため、後から
    # 「途中で出せて最後に消えた点」を数値化できない）。図バイトは不変、追加ファイルのみ。
    # DISTRIBUTED_PCN_LIVE_UNIFORM_PF_NPZ=0 で無効化。
    if os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_NPZ", "1") != "0":
        if iteration is not None:
            npz_path = out / f"uniform_cmd_pts_iter_{int(iteration):03d}.npz"
        else:
            npz_path = out / f"uniform_cmd_pts_{label}.npz"
        np.savez_compressed(
            npz_path,
            achieved=pts,
            eval_pf=pf,
            archive_pf=np.asarray(archive_pf, dtype=np.float64),
            exploration=np.asarray(exploration_pts, dtype=np.float64),
            iteration=np.int64(-1 if iteration is None else int(iteration)),
        )

    stats: Dict[str, Any] = {
        "label": label,
        "iteration": iteration,
        "n_pf": int(len(nd)),
        "n_achieved": int(len(pts)),
        "plot": str(path.resolve()),
    }
    if iteration is not None:
        stats_path = out / f"uniform_cmd_stats_iter_{int(iteration):03d}.json"
    else:
        stats_path = out / f"uniform_cmd_stats_{label}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    return str(path.resolve())


def print_live_pf_saved(path: str, iteration: Optional[int]) -> None:
    if iteration is not None:
        print(f"[LIVE_PF] iter={iteration} {path}")
    else:
        print(f"[LIVE_PF] {path}")
