#!/usr/bin/env python3
"""Best-checkpoint PCN evaluation.

Phase3 の mode-collapse 対策: 最終 checkpoint だけを評価するのではなく、
学習中アーカイブ非支配数 (training_iteration_summary.json の pareto_front_size) が
豊かだった checkpoint を上位 K だけ本評価し、「passed を満たす中で点数×cost幅が最大」の
checkpoint を採用する。

出力 (exec_dir 直下):
  - pf_score.json            … 採用 checkpoint のスコア (+ "selected_checkpoint")
  - eval_pf_points.npz       … 採用 checkpoint の achieved / pareto_front / command_targets
  - best_checkpoint_selection.json … 全候補のスコア一覧と選択理由
  - eval_pf_best.png         … 採用 checkpoint の PF 図

env で run_trace24_pcn_scratch.sh と同じノブ (PCN_EVAL_COMMAND_MODE,
PCN_PF_COMMAND_LOW_WAIT_*, PCN_SCORE_*, EVAL_GRID 等) を読む。
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

from scripts.eval_uniform_command_pf import run
from scripts.score_eval_pf_vs_reference import load_npz_pf, score_eval_vs_reference


def _iter_of(ckpt: str) -> int:
    # .../model_iter_030.pth -> 30
    base = os.path.basename(ckpt)
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else -1


def _archive_pf_by_iter(exec_dir: Path) -> dict[int, int]:
    f = exec_dir / "training_iteration_summary.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    out: dict[int, int] = {}
    for r in d.get("rows", []):
        it = r.get("iteration")
        pf = r.get("pareto_front_size")
        if it is not None and pf is not None:
            out[int(it)] = int(pf)
    return out


def _nearest_archive_size(it: int, arch: dict[int, int]) -> int:
    if not arch:
        return 0
    if it in arch:
        return arch[it]
    k = min(arch.keys(), key=lambda x: abs(x - it))
    return arch[k]


def select_candidates(exec_dir: Path, top_k: int, min_iter_frac: float) -> list[Path]:
    ckpts = sorted(glob.glob(str(exec_dir / "**" / "model_iter_*.pth"), recursive=True), key=_iter_of)
    ckpts = [Path(c) for c in ckpts if _iter_of(c) >= 0]
    if not ckpts:
        raise SystemExit(f"no model_iter_*.pth under {exec_dir}")
    arch = _archive_pf_by_iter(exec_dir)
    max_iter = _iter_of(str(ckpts[-1]))
    cutoff = max_iter * min_iter_frac
    # 早期(高アーカイブだが gap が悪い領域)を除外しつつアーカイブ非支配数で上位 K
    pool = [c for c in ckpts if _iter_of(str(c)) >= cutoff] or ckpts
    pool_sorted = sorted(pool, key=lambda c: _nearest_archive_size(_iter_of(str(c)), arch), reverse=True)
    chosen = list(pool_sorted[: max(1, top_k)])
    # 最終 checkpoint は必ず比較対象に含める
    if ckpts[-1] not in chosen:
        chosen.append(ckpts[-1])
    # iter 昇順で返す
    return sorted(set(chosen), key=lambda c: _iter_of(str(c)))


def eval_one(ckpt: Path, snap: Path | None, exec_dir: Path, ref, ref_npz: Path,
             grid: int, command_mode: str, config_path: str | None) -> dict:
    label = f"ckpt_iter_{_iter_of(str(ckpt)):03d}"
    stats = run(ckpt, snap, exec_dir, label, grid=grid, device="cpu",
                config_path=config_path, command_mode=command_mode, ref_pf_npz=ref_npz)
    all_pts = np.asarray(stats["achieved_points"], dtype=np.float64)
    score_pts = (np.asarray(stats.get("score_points"), dtype=np.float64)
                 if stats.get("score_points") is not None else all_pts)
    cmd_targets = np.asarray(stats.get("command_targets") or [], dtype=np.float64)
    is_pf_cmd = stats.get("command_mode") in ("pf_ref", "pf_archive")
    score = score_eval_vs_reference(
        all_pts if is_pf_cmd else score_pts, ref,
        eval_is_pf_commands=is_pf_cmd,
        command_targets=cmd_targets if len(cmd_targets) else None,
        low_wait_max=float(os.environ.get("PCN_SCORE_LOW_WAIT_MAX", "0")),
        min_low_wait_covered_frac=float(os.environ.get("PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC", "0")),
    )
    return {
        "checkpoint": str(ckpt),
        "iter": _iter_of(str(ckpt)),
        "label": label,
        "score": score,
        "achieved_points": all_pts,
        "pareto_front": score_pts,
        "command_targets": cmd_targets,
        "plot": stats.get("plot"),
        "command_mode": stats.get("command_mode"),
    }


def _rank_key(cand: dict):
    s = cand["score"]
    passed = 1 if s.get("passed") else 0
    n = int(s.get("eval_pf_unique_n", 0) or 0)
    span = float(s.get("cost_span", 0.0) or 0.0)
    frac_bad = float(s.get("frac_bad", 1.0) or 1.0)
    # passed 優先 → 点数 → -frac_bad → cost幅 (width と count のバランス)
    return (passed, n, -frac_bad, span)


def main() -> int:
    exec_dir = Path(os.environ["EXEC_DIR"])
    ref_npz = Path(os.environ["REF_NPZ"])
    config_path = os.environ.get("DISTRIBUTED_PCN_CONFIG")
    grid = int(os.environ.get("EVAL_GRID", "32"))
    command_mode = os.environ.get("PCN_EVAL_COMMAND_MODE", "pf_ref")
    # 既定で全 checkpoint を評価（eval ~18s/ckpt と安価、early/late どちらの collapse でも
    # 豊かな checkpoint を取りこぼさない）。proxy cutoff は早期collapse runで害になるため 0。
    top_k = int(os.environ.get("BEST_CKPT_TOPK", "99"))
    min_iter_frac = float(os.environ.get("BEST_CKPT_MIN_ITER_FRAC", "0.0"))

    snap = exec_dir / "learner_replay_snapshot.pkl.gz"
    snap = snap if snap.exists() else None
    ref = load_npz_pf(ref_npz)

    candidates = select_candidates(exec_dir, top_k, min_iter_frac)
    print(f"[best-ckpt] candidates: {[_iter_of(str(c)) for c in candidates]}")

    results = []
    for c in candidates:
        try:
            results.append(eval_one(c, snap, exec_dir, ref, ref_npz, grid, command_mode, config_path))
            s = results[-1]["score"]
            print(f"[best-ckpt] iter={results[-1]['iter']:>3} "
                  f"passed={s.get('passed')} n={s.get('eval_pf_unique_n')} "
                  f"cost_span={s.get('cost_span'):.0f} mean_gap={s.get('mean_gap', -1):.1f} "
                  f"frac_bad={s.get('frac_bad', -1):.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"[best-ckpt] iter={_iter_of(str(c))} eval FAILED: {e}")

    if not results:
        raise SystemExit("all candidate evals failed")

    best = max(results, key=_rank_key)
    print(f"[best-ckpt] SELECTED iter={best['iter']} -> {best['checkpoint']}")

    # 採用 checkpoint の成果物を確定名で出力
    np.savez(exec_dir / "eval_pf_points.npz",
             points=best["achieved_points"], pareto_front=best["pareto_front"],
             command_targets=best["command_targets"])
    score_out = dict(best["score"])
    score_out["selected_checkpoint"] = best["checkpoint"]
    score_out["selected_iter"] = best["iter"]
    (exec_dir / "pf_score.json").write_text(json.dumps(score_out, indent=2) + "\n")
    if best.get("plot") and Path(best["plot"]).exists():
        shutil.copy(best["plot"], exec_dir / "eval_pf_best.png")

    summary = {
        "selected_iter": best["iter"],
        "selected_checkpoint": best["checkpoint"],
        "candidates": [
            {"iter": r["iter"], **{k: r["score"].get(k) for k in
             ("passed", "eval_pf_unique_n", "cost_span", "mean_gap", "frac_bad")}}
            for r in sorted(results, key=lambda r: r["iter"])
        ],
    }
    (exec_dir / "best_checkpoint_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if best["score"].get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
