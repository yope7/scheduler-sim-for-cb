#!/usr/bin/env python3
"""
分散PCN（学習フルパイプライン）の壁時計計測: Cビットマップ観測 vs イベント観測。

- 同一エントリポイント ``python -m src.distributed.distributed_pcn``
- 差分は import 前にサブプロセスで設定する ``DISTRIBUTED_PCN_USE_EVENT_OBS`` のみ（公平比較）
- 長時間化を防ぐため ``DISTRIBUTED_PCN_QUICK`` + オプションで追加予算上書き
- ``--through-phase2``: Phase3 の反復を空にし（``DISTRIBUTED_PCN_N_ITERATIONS=0``）、
  最終可視化をオフ（``DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0``）。既定で Phase2 特徴重要度もオフ。
  観測コスト差はフェーズ1+2 のログ秒で見やすい。
- 1実行あたり ``--timeout`` で強制終了（打ち切りは timeout_hit と記録）
- 子プロセス（``distributed_pcn``）のログは**既定で標準出力にそのまま流す**。抑制する場合は ``--quiet``。

出力: JSON + .summary.txt（既定 ``0511/``）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_command_streaming(
    cmd: Sequence[str],
    *,
    cwd: str,
    env: Dict[str, str],
    timeout_sec: int,
) -> Tuple[int, str]:
    """subprocess は stdout にファイルオブジェクトを渡すと fileno が必要なため、PIPE + 読み取りスレッドでターミナルへ流す。"""
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    parts: List[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            parts.append(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
        th.join(timeout=120)
        out = "".join(parts)
        raise subprocess.TimeoutExpired(list(cmd), timeout_sec, output=out, stderr=None) from None
    th.join(timeout=120)
    out = "".join(parts)
    rc = proc.returncode
    if rc is None:
        rc = 0
    return int(rc), out


def _extract_phase_durations(output: str) -> Dict[str, float | None]:
    phases: Dict[str, float | None] = {}
    patterns = [
        ("phase1", r"フェーズ1完了.*?経過時間: ([\d.]+)秒"),
        ("phase2", r"フェーズ2完了.*?経過時間: ([\d.]+)秒"),
        ("phase3", r"フェーズ3完了.*?経過時間: ([\d.]+)秒"),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, output, re.DOTALL)
        phases[key] = float(m.group(1)) if m else None
    return phases


def _build_budget_env(
    quick: bool,
    n_iterations: int | None,
    n_actors: int | None,
    initial_episodes: int | None,
    supervised_epochs: int | None,
    *,
    through_phase2: bool = False,
    with_phase2_importance: bool = False,
) -> Dict[str, str]:
    e: Dict[str, str] = {}
    if quick:
        e["DISTRIBUTED_PCN_QUICK"] = "1"
    if n_iterations is not None:
        e["DISTRIBUTED_PCN_N_ITERATIONS"] = str(n_iterations)
    if n_actors is not None:
        e["DISTRIBUTED_PCN_N_ACTORS"] = str(n_actors)
    if initial_episodes is not None:
        e["DISTRIBUTED_PCN_INITIAL_EPISODES"] = str(initial_episodes)
    if supervised_epochs is not None:
        e["DISTRIBUTED_PCN_SUPERVISED_EPOCHS"] = str(supervised_epochs)
    if through_phase2:
        e["DISTRIBUTED_PCN_ENABLE_VISUALIZATION"] = "0"
        if not with_phase2_importance:
            e["DISTRIBUTED_PCN_PHASE2_IMPORTANCE"] = "0"
    return e


def run_one_mode(
    mode: str,
    n_jobs: int,
    n_onprem: int,
    n_cloud: int,
    timeout_sec: int,
    budget_env: Dict[str, str],
    quiet: bool,
) -> Dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env["DISTRIBUTED_PCN_JOBS"] = str(n_jobs)
    env["DISTRIBUTED_PCN_ONPREM"] = str(n_onprem)
    env["DISTRIBUTED_PCN_CLOUD"] = str(n_cloud)
    env["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1" if mode == "event_c" else "0"
    for k, v in budget_env.items():
        env[k] = v

    cmd = [sys.executable, "-m", "src.distributed.distributed_pcn"]
    t0 = datetime.now().isoformat(timespec="seconds")
    t_start = datetime.now().timestamp()
    try:
        if quiet:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            stdout_full = proc.stdout or ""
            stderr_full = proc.stderr or ""
            rc = proc.returncode
        else:
            rc, out = _run_command_streaming(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                timeout_sec=timeout_sec,
            )
            stdout_full = out
            stderr_full = ""
        t_end = datetime.now().timestamp()
        phases = _extract_phase_durations(out)
        p1, p2 = phases.get("phase1"), phases.get("phase2")
        phase12_sec = (p1 + p2) if p1 is not None and p2 is not None else None
        return {
            "mode": mode,
            "success": rc == 0,
            "returncode": rc,
            "elapsed_wall_sec": t_end - t_start,
            "timeout_hit": False,
            "started_iso": t0,
            "phase1_sec": p1,
            "phase2_sec": p2,
            "phase12_sec": phase12_sec,
            "phase3_sec": phases.get("phase3"),
            "stdout_tail": stdout_full[-4000:],
            "stderr_tail": stderr_full[-2000:],
        }
    except subprocess.TimeoutExpired as ex:
        t_end = datetime.now().timestamp()
        if quiet:
            out = ""
            if ex.stdout:
                out += ex.stdout
            if ex.stderr:
                out += "\n" + ex.stderr
            stdout_tail_src = ex.stdout or ""
            stderr_tail_src = ex.stderr or ""
        else:
            out = ex.output if getattr(ex, "output", None) else ""
            stdout_tail_src = out
            stderr_tail_src = ""
        phases = _extract_phase_durations(out)
        p1, p2 = phases.get("phase1"), phases.get("phase2")
        phase12_sec = (p1 + p2) if p1 is not None and p2 is not None else None
        return {
            "mode": mode,
            "success": False,
            "returncode": None,
            "elapsed_wall_sec": t_end - t_start,
            "timeout_hit": True,
            "error": f"timeout_after_{timeout_sec}s",
            "started_iso": t0,
            "phase1_sec": p1,
            "phase2_sec": p2,
            "phase12_sec": phase12_sec,
            "phase3_sec": phases.get("phase3"),
            "stdout_tail": stdout_tail_src[-4000:] if stdout_tail_src else "",
            "stderr_tail": stderr_tail_src[-2000:] if stderr_tail_src else "",
        }


def _parse_nodes(spec: str) -> Tuple[str, int, int]:
    spec = spec.strip().lower()
    if "x" not in spec:
        raise ValueError(f"invalid node spec: {spec}")
    a, b = spec.split("x", 1)
    return spec, int(a), int(b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCN 学習込み wall-clock（bitmap_c vs event_c）"
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default="64",
        help="カンマ区切りジョブ数",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default="256x1024",
        help='カンマ区切り「オンプレxクラウド」',
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="1 Run（1モード×1ワークロード）あたりの秒数上限",
    )
    parser.add_argument(
        "--quick",
        dest="quick",
        action="store_true",
        default=True,
        help="DISTRIBUTED_PCN_QUICK=1（既定）",
    )
    parser.add_argument(
        "--no-quick",
        dest="quick",
        action="store_false",
        help="クイックモードをオフ",
    )
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--n-actors", type=int, default=None)
    parser.add_argument("--initial-episodes", type=int, default=None)
    parser.add_argument("--supervised-epochs", type=int, default=None)
    parser.add_argument(
        "--through-phase2",
        action="store_true",
        help="Phase3 をスキップ（N_ITERATIONS=0）、可視化オフ。Phase2 特徴重要度は既定オフ",
    )
    parser.add_argument(
        "--with-phase2-importance",
        action="store_true",
        help="--through-phase2 時も DISTRIBUTED_PCN_PHASE2_IMPORTANCE=1 を維持",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="結果ディレクトリ（既定: 0511）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="子プロセスのログを標準出力に出さない（フェーズ時間抽出のみバッファ）",
    )
    args = parser.parse_args()

    n_iterations = args.n_iterations
    if args.through_phase2 and n_iterations is None:
        n_iterations = 0

    budget_env = _build_budget_env(
        args.quick,
        n_iterations,
        args.n_actors,
        args.initial_episodes,
        args.supervised_epochs,
        through_phase2=args.through_phase2,
        with_phase2_importance=args.with_phase2_importance,
    )

    job_list = [int(x.strip()) for x in args.jobs.split(",") if x.strip()]
    node_specs: List[Tuple[str, int, int]] = [
        _parse_nodes(p) for p in args.nodes.split(",") if p.strip()
    ]

    out_dir = Path(args.output_dir or (PROJECT_ROOT / "0511"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"pcn_training_wallclock_{stamp}.json"
    summary_path = out_dir / f"pcn_training_wallclock_{stamp}.summary.txt"

    modes = ["bitmap_c", "event_c"]
    rows: List[Dict[str, Any]] = []

    meta = {
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "entrypoint": "python -m src.distributed.distributed_pcn",
        "fairness_note": (
            "DISTRIBUTED_PCN_USE_EVENT_OBS のみ変更。その他の予算環境変数は両モード同一。"
        ),
        "timeout_sec_per_run": args.timeout,
        "budget_env": budget_env,
        "quick_mode": args.quick,
        "through_phase2": args.through_phase2,
        "with_phase2_importance": args.with_phase2_importance,
        "quiet_subprocess": args.quiet,
    }

    for node_label, nop, ncl in node_specs:
        for nj in job_list:
            pair: Dict[str, Any] = {
                "workload": {
                    "n_jobs": nj,
                    "n_on_premise_node": nop,
                    "n_cloud_node": ncl,
                    "node_label": node_label,
                },
            }
            for mode in modes:
                print(
                    f"\n>>> PCN  wall_clock  mode={mode}  jobs={nj}  nodes={node_label}  "
                    f"timeout={args.timeout}s",
                    flush=True,
                )
                pair[mode] = run_one_mode(
                    mode,
                    nj,
                    nop,
                    ncl,
                    timeout_sec=args.timeout,
                    budget_env=budget_env,
                    quiet=args.quiet,
                )
            b = pair["bitmap_c"]
            e = pair["event_c"]
            ratio = None
            if b.get("elapsed_wall_sec") and e.get("elapsed_wall_sec"):
                ratio = e["elapsed_wall_sec"] / max(b["elapsed_wall_sec"], 1e-9)
            pair["ratio_event_vs_bitmap_wall"] = ratio
            ratio_p12 = None
            bp12, ep12 = b.get("phase12_sec"), e.get("phase12_sec")
            if bp12 is not None and ep12 is not None:
                ratio_p12 = ep12 / max(bp12, 1e-9)
            pair["ratio_event_vs_bitmap_phase12"] = ratio_p12
            rows.append(pair)

    payload = {"meta": meta, "results": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "PCN 学習込み wall-clock（bitmap_c vs event_c）",
        f"timeout/run={args.timeout}s  quick={args.quick}  through_phase2={args.through_phase2}  "
        f"budget_env={json.dumps(budget_env, ensure_ascii=False)}",
        "",
    ]
    if args.through_phase2:
        hdr = (
            f"{'jobs':>6} {'nodes':>14} {'bitmap_s':>10} {'event_s':>10} {'event/bitmap':>12} "
            f"{'p12_b':>8} {'p12_e':>8} {'p12_ratio':>10} {'b_ok':>5} {'e_ok':>5} {'note':>20}"
        )
    else:
        hdr = (
            f"{'jobs':>6} {'nodes':>14} {'bitmap_s':>10} {'event_s':>10} {'event/bitmap':>12} "
            f"{'b_ok':>5} {'e_ok':>5} {'note':>20}"
        )
    lines.append(hdr)
    lines.append("-" * (len(hdr) + 4))
    for r in rows:
        w = r["workload"]
        b = r["bitmap_c"]
        ev = r["event_c"]
        ratio = r.get("ratio_event_vs_bitmap_wall")
        rs = f"{ratio:.4f}" if ratio is not None else "n/a"
        note = ""
        if b.get("timeout_hit") or ev.get("timeout_hit"):
            note = "timeout"
        elif not b.get("success") or not ev.get("success"):
            note = "fail"
        if args.through_phase2:
            bp12 = b.get("phase12_sec")
            ep12 = ev.get("phase12_sec")
            rp12 = r.get("ratio_event_vs_bitmap_phase12")
            s_b12 = f"{bp12:.2f}" if bp12 is not None else "n/a"
            s_e12 = f"{ep12:.2f}" if ep12 is not None else "n/a"
            s_r12 = f"{rp12:.4f}" if rp12 is not None else "n/a"
            lines.append(
                f"{w['n_jobs']:>6} {w['node_label']:>14} "
                f"{b.get('elapsed_wall_sec', 0):>10.2f} {ev.get('elapsed_wall_sec', 0):>10.2f} {rs:>12} "
                f"{s_b12:>8} {s_e12:>8} {s_r12:>10} "
                f"{str(b.get('success')):>5} {str(ev.get('success')):>5} {note:>20}"
            )
        else:
            lines.append(
                f"{w['n_jobs']:>6} {w['node_label']:>14} "
                f"{b.get('elapsed_wall_sec', 0):>10.2f} {ev.get('elapsed_wall_sec', 0):>10.2f} {rs:>12} "
                f"{str(b.get('success')):>5} {str(ev.get('success')):>5} {note:>20}"
            )
    lines.append("")
    lines.append(f"JSON: {json_path}")
    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
