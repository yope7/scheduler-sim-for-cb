#!/usr/bin/env python3
"""
mo_hv_benchmark.json を読み、HV を学習世代／学習イテレーション軸でプロットする。
PCN は training_iteration_at_eval（EVAL_INTERVAL 刻み）を augment_report_pcn_training_axis で埋める。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.mo_hv_pcn_axis import augment_report_pcn_training_axis, time_to_solution_on_axis


def _x_axis_nsga2(entry: Dict[str, Any]) -> List[float]:
    g = entry.get("generation_index")
    if isinstance(g, list) and g:
        return [float(i + 1) for i in g]
    hvs = entry.get("hypervolume_series") or []
    return [float(i + 1) for i in range(len(hvs))]


def _x_axis_pcn(entry: Dict[str, Any]) -> List[float]:
    t = entry.get("training_iteration_at_eval")
    if isinstance(t, list) and len(t) == len(entry.get("hypervolume_series") or []):
        return [float(x) for x in t]
    raise RuntimeError("PCN に training_iteration_at_eval がありません。augment を先に実行してください。")


def plot_episode_lines(
    report: Dict[str, Any],
    out_png: Path,
    hv_threshold: Optional[float],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    algos = report.get("algorithms") or {}
    labels = {"nsga2": "NSGA-II", "pcn_distributed": "PCN"}
    for key, label in labels.items():
        e = algos.get(key)
        if not e:
            continue
        hvs = e.get("hypervolume_series") or []
        if not hvs:
            continue
        if key == "nsga2":
            x = _x_axis_nsga2(e)
        else:
            x = _x_axis_pcn(e)
        ax.plot(x, hvs, marker="o", ms=3, label=label)
    if hv_threshold is not None:
        ax.axhline(hv_threshold, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Training iteration / generation (1-based)")
    ax.set_ylabel("Hypervolume (solution space)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_tts_summary(
    report: Dict[str, Any],
    hv_threshold: float,
) -> Dict[str, Any]:
    algos = report.get("algorithms") or {}
    out: Dict[str, Any] = {
        "source_json": report.get("config_path", ""),
        "hv_threshold": hv_threshold,
        "note": (
            "PCN の tts_episode は training_iteration_at_eval 上で閾値初達（EVAL_INTERVAL に整合）。"
            " NSGA-II は世代番号 1..G。"
        ),
        "algorithms": {},
    }
    for key, lab in (("nsga2", "NSGA-II"), ("pcn_distributed", "PCN")):
        e = algos.get(key)
        if not e:
            continue
        hvs = e.get("hypervolume_series") or []
        times = e.get("time_seconds") or []
        if key == "nsga2":
            x = _x_axis_nsga2(e)
            last_ep = float(x[-1]) if x else 0.0
        else:
            x = _x_axis_pcn(e)
            last_ep = float(x[-1]) if x else 0.0
        tts_t = None
        if times and len(times) == len(hvs):
            tts_t = time_to_solution_on_axis(times, hvs, hv_threshold)
        tts_e = None
        if x and len(x) == len(hvs):
            tts_e = time_to_solution_on_axis(x, hvs, hv_threshold)
        out["algorithms"][key] = {
            "label": lab,
            "tts_time_sec": tts_t,
            "tts_episode": tts_e,
            "final_hv": float(hvs[-1]) if hvs else None,
            "n_points": len(hvs),
            "last_time_sec": float(times[-1]) if times else None,
            "last_episode": last_ep,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot HV vs training iteration from mo_hv_benchmark.json")
    p.add_argument("json_path", type=str)
    p.add_argument("--out-png", type=str, default=None)
    p.add_argument("--out-summary", type=str, default=None)
    p.add_argument("--hv-threshold", type=float, default=None, help="既定: JSON の hv_threshold_effective")
    p.add_argument("--write-json", action="store_true", help="augment 後のレポートを json_path に上書き")
    args = p.parse_args()

    path = Path(args.json_path)
    with open(path, encoding="utf-8") as f:
        report: Dict[str, Any] = json.load(f)

    augment_report_pcn_training_axis(report)
    thr: Optional[float] = args.hv_threshold
    if thr is None:
        thr = report.get("hv_threshold_effective")
    if thr is None:
        raise SystemExit("hv_threshold が指定できません（--hv-threshold または JSON 内）")

    if args.hv_threshold is not None:
        pcn = report.get("algorithms", {}).get("pcn_distributed")
        if pcn and pcn.get("training_iteration_at_eval") and pcn.get("hypervolume_series"):
            tts_e = time_to_solution_on_axis(
                pcn["training_iteration_at_eval"],
                pcn["hypervolume_series"],
                float(args.hv_threshold),
            )
            pcn["tts_episode"] = tts_e
            pcn["tts_training_iteration"] = tts_e
            pcn["tts_episode_threshold"] = float(args.hv_threshold)

    if args.write_json:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)

    stem = path.parent / path.stem
    out_png = Path(args.out_png or f"{stem}_episode_axis.png")
    out_summary = Path(args.out_summary or f"{stem}_tts_summary_episode_axis.json")

    plot_episode_lines(report, out_png, float(thr))
    summary = build_tts_summary(report, float(thr))
    summary["source_json"] = str(path.resolve())
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
