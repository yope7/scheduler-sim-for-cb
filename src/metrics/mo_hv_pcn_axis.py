"""
PCN の mo_hv レポート用: 学習イテレーション軸（EVAL_INTERVAL 刻み）。
pymoo 等に依存しない（plot や後処理のみの環境でも import 可能）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def time_to_solution_on_axis(
    axis: Sequence[float],
    hypervolumes: Sequence[float],
    threshold: float,
) -> Optional[float]:
    """横軸がエピソードや学習イテレーションのときの「初めて閾値を超えた x」。"""
    for t, hv in zip(axis, hypervolumes):
        if hv >= threshold:
            return float(t)
    return None


def pcn_training_iteration_axis(
    n_points: int,
    n_iterations: int,
    eval_interval: int,
) -> List[float]:
    if n_points <= 0:
        return []
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    n_periodic = n_iterations // eval_interval
    if n_points == n_periodic:
        return [float((i + 1) * eval_interval) for i in range(n_periodic)]
    if n_points == n_periodic + 1:
        xs = [float((i + 1) * eval_interval) for i in range(n_periodic)]
        xs.append(float(n_iterations))
        return xs
    if n_points < n_periodic:
        return [float((i + 1) * eval_interval) for i in range(n_points)]
    xs = [float((i + 1) * eval_interval) for i in range(n_periodic)]
    for _ in range(n_points - len(xs)):
        xs.append(float(n_iterations))
    return xs


def augment_report_pcn_training_axis(report: Dict[str, Any]) -> None:
    pac = report.get("param_algorithm_compare") or {}
    mo = pac.get("mo_hypervolume_benchmark") or {}
    dpc = pac.get("distributed_pcn") or {}
    entry = report.get("algorithms", {}).get("pcn_distributed")
    if not entry:
        return
    hvs = entry.get("hypervolume_series") or []
    if not hvs:
        return
    eval_interval = int(mo.get("pcn_eval_interval", 5))
    n_iter = int(dpc.get("n_iterations", 200))
    axis = pcn_training_iteration_axis(len(hvs), n_iter, eval_interval)
    entry["pcn_eval_interval"] = eval_interval
    entry["pcn_n_iterations"] = n_iter
    entry["training_iteration_at_eval"] = axis
    thr = report.get("hv_threshold_effective")
    if thr is not None:
        tts_ep = time_to_solution_on_axis(axis, hvs, float(thr))
        entry["tts_training_iteration"] = tts_ep
        entry["tts_episode"] = tts_ep
    entry["note"] = (
        "HV は学習ループ内で EVAL_INTERVAL（既定 5）イテレーションごとに記録。"
        " training_iteration_at_eval はその評価が走った学習イテレーション番号（例: 5,10,…）。"
        " 初期ランダム収集フェーズのエピソード数は別カウント。"
    )
