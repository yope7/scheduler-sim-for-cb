#!/usr/bin/env python3
"""
distributed_pcn.py を外側から実行し、
ジョブ数×ノード構成の組み合わせでベンチマークを実施し、実行時間を記録する。

組み合わせ:
  - ジョブ数: 16, 32, 64
  - ノード構成（小さい順）: (256,1024), (512,2048)

使用方法:
  python scripts/run_distributed_pcn_sweep.py
  python scripts/run_distributed_pcn_sweep.py --quick   # 短時間モードでテスト
  python scripts/run_distributed_pcn_sweep.py -o results.json
  python scripts/run_distributed_pcn_sweep.py --jobs 16,32 --onprem 256 --cloud 1024
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# デフォルト値（--jobs / --onprem / --cloud で上書き可能）
# デフォルトは (256,1024) と (512,2048) の2構成のみ
DEFAULT_JOB_COUNTS = [16, 32, 64]
DEFAULT_NODE_CONFIGS = [
    ("(256,1024)", 256, 1024),
    ("(512,2048)", 512, 2048),
]


def run_distributed_pcn(
    n_jobs: int,
    n_on_prem: int,
    n_cloud: int,
    quick: bool = False,
    timeout: int = 3600,
    stream_output: bool = True,
) -> dict:
    """
    distributed_pcn をサブプロセスで実行し、実行時間を返す。
    stream_output=True のとき、サブプロセスの出力をリアルタイムで表示する。
    """
    env = os.environ.copy()
    env["DISTRIBUTED_PCN_JOBS"] = str(n_jobs)
    env["DISTRIBUTED_PCN_ONPREM"] = str(n_on_prem)
    env["DISTRIBUTED_PCN_CLOUD"] = str(n_cloud)
    if quick:
        env["DISTRIBUTED_PCN_QUICK"] = "1"

    cmd = [sys.executable, "-m", "src.distributed.distributed_pcn"]
    start = time.time()
    output_lines: list[str] = []
    timed_out = [False]  # mutable for closure

    def kill_after_timeout():
        time.sleep(timeout)
        if proc.poll() is None:
            timed_out[0] = True
            proc.kill()

    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if stream_output:
        timer = threading.Thread(target=kill_after_timeout, daemon=True)
        timer.start()
        try:
            for line in iter(proc.stdout.readline, "") if proc.stdout else []:
                line = line.rstrip()
                output_lines.append(line)
                print(f"    │ {line}")
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    else:
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            output_lines = (stdout or "").splitlines()
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = time.time() - start
            return {
                "success": False,
                "elapsed_sec": elapsed,
                "error": "timeout",
            }

    elapsed = time.time() - start
    output = "\n".join(output_lines)

    if timed_out[0]:
        return {
            "success": False,
            "elapsed_sec": elapsed,
            "error": "timeout",
        }

    if proc.returncode != 0:
        err_msg = output[-2000:] if output else ""
        return {
            "success": False,
            "elapsed_sec": elapsed,
            "returncode": proc.returncode,
            "error": err_msg,
        }

    phases = _extract_phase_durations(output)
    return {
        "success": True,
        "elapsed_sec": elapsed,
        "phase1_sec": phases.get("phase1"),
        "phase2_sec": phases.get("phase2"),
        "phase3_sec": phases.get("phase3"),
    }


def _extract_phase_durations(output: str) -> dict:
    """distributed_pcn の標準出力からフェーズごとの経過時間を抽出"""
    phases = {}
    patterns = [
        ("phase1", r"フェーズ1完了.*?経過時間: ([\d.]+)秒"),
        ("phase2", r"フェーズ2完了.*?経過時間: ([\d.]+)秒"),
        ("phase3", r"フェーズ3完了.*?経過時間: ([\d.]+)秒"),
    ]
    for key, pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            phases[key] = float(match.group(1))
    return phases


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="distributed_pcn のジョブ数×ノード構成スイープ")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="短時間モード（DISTRIBUTED_PCN_QUICK=1）で実行",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="結果を保存するJSONファイル（省略時はタイムスタンプ付きで自動生成）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600000,
        help="1実行あたりのタイムアウト秒（デフォルト: 3600）",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=None,
        help="ジョブ数のリスト（カンマ区切り、例: 16,32,64）",
    )
    parser.add_argument(
        "--onprem",
        type=str,
        default=None,
        help="オンプレミスノード数のリスト（カンマ区切り、例: 256,512）",
    )
    parser.add_argument(
        "--cloud",
        type=str,
        default=None,
        help="クラウドノード数のリスト（カンマ区切り、例: 1024,2048）",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="distributed_pcn の出力をリアルタイム表示しない（完了後にまとめて表示）",
    )
    args = parser.parse_args()

    # ジョブ数・ノード構成の決定
    job_counts = _parse_int_list(args.jobs) if args.jobs else DEFAULT_JOB_COUNTS
    if args.onprem is not None and args.cloud is not None:
        onprem_list = _parse_int_list(args.onprem)
        cloud_list = _parse_int_list(args.cloud)
        if len(onprem_list) != len(cloud_list):
            parser.error("--onprem と --cloud の要素数は同じにしてください")
        node_configs = [
            (f"({op},{cl})", op, cl)
            for op, cl in zip(onprem_list, cloud_list)
        ]
    else:
        node_configs = DEFAULT_NODE_CONFIGS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"execution_times_{timestamp}.json"

    records = []
    total_runs = len(job_counts) * len(node_configs)
    run_idx = 0
    sweep_start = time.time()
    run_times: list[float] = []

    print("=" * 70)
    print("distributed_pcn ジョブ数×ノード構成スイープ")
    print(f"  ジョブ数: {job_counts}")
    print(f"  ノード構成: {[nc[0] for nc in node_configs]}")
    print(f"  総実行数: {total_runs}")
    print(f"  クイックモード: {args.quick}")
    print(f"  出力先: {output_path}")
    print(f"  開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for n_jobs in job_counts:
        for node_name, n_on_prem, n_cloud in node_configs:
            run_idx += 1
            run_start = time.time()
            run_start_str = datetime.now().strftime("%H:%M:%S")

            # 進捗・ETA
            pct = 100 * (run_idx - 1) / total_runs if total_runs > 0 else 0
            eta_str = ""
            if run_times and run_idx < total_runs:
                avg_time = sum(run_times) / len(run_times)
                remaining = total_runs - run_idx
                eta_sec = avg_time * remaining
                eta_str = f"  [残り約{eta_sec/60:.1f}分]"

            print(f"\n{'─' * 70}")
            print(f"[{run_idx}/{total_runs}] ({pct:.0f}%) n_jobs={n_jobs}, {node_name} (onprem={n_on_prem}, cloud={n_cloud})")
            print(f"  開始: {run_start_str}{eta_str}")
            print(f"  distributed_pcn 実行中...")

            r = run_distributed_pcn(
                n_jobs=n_jobs,
                n_on_prem=n_on_prem,
                n_cloud=n_cloud,
                quick=args.quick,
                timeout=args.timeout,
                stream_output=not args.no_stream,
            )
            run_times.append(r["elapsed_sec"])
            record = {
                "n_jobs": n_jobs,
                "node_config": node_name,
                "n_on_premise": n_on_prem,
                "n_cloud": n_cloud,
                "elapsed_sec": r["elapsed_sec"],
                "success": r["success"],
            }
            if r.get("phase1_sec") is not None:
                record["phase1_sec"] = r["phase1_sec"]
            if r.get("phase2_sec") is not None:
                record["phase2_sec"] = r["phase2_sec"]
            if r.get("phase3_sec") is not None:
                record["phase3_sec"] = r["phase3_sec"]
            if not r["success"]:
                record["error"] = r.get("error", "unknown")
            records.append(record)

            status = "OK" if r["success"] else "FAIL"
            phase_str = ""
            if all(r.get(f"phase{i}_sec") is not None for i in (1, 2, 3)):
                phase_str = f"  [P1:{r['phase1_sec']:.1f}s P2:{r['phase2_sec']:.1f}s P3:{r['phase3_sec']:.1f}s]"
            run_end_str = datetime.now().strftime("%H:%M:%S")
            print(f"  → 完了: {run_end_str}  {status}  経過: {r['elapsed_sec']:.1f}秒{phase_str}")

    # 結果を保存
    result = {
        "timestamp": timestamp,
        "quick_mode": args.quick,
        "records": records,
    }
    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = project_root / out_file
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # サマリーファイル（人間が読みやすい形式）も出力
    summary_path = out_file.with_suffix(".txt")
    lines = [
        "=" * 80,
        f"実行時間記録 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"クイックモード: {args.quick}",
        "=" * 80,
        "",
        "# phase1=初期エピソード収集, phase2=教師あり学習, phase3=改良された経験の実現",
        "n_jobs  node_config              total(秒)  phase1(秒)  phase2(秒)  phase3(秒)  status",
        "-" * 80,
    ]
    for rec in records:
        status = "OK" if rec["success"] else "FAIL"
        p1 = f"{rec['phase1_sec']:.1f}" if rec.get("phase1_sec") is not None else "-"
        p2 = f"{rec['phase2_sec']:.1f}" if rec.get("phase2_sec") is not None else "-"
        p3 = f"{rec['phase3_sec']:.1f}" if rec.get("phase3_sec") is not None else "-"
        lines.append(
            f"{rec['n_jobs']:6d}  {rec['node_config']:20s}  "
            f"{rec['elapsed_sec']:8.1f}  {p1:>10}  {p2:>10}  {p3:>10}  {status}"
        )
    lines.extend(["", "=" * 80])
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    sweep_elapsed = time.time() - sweep_start
    ok_count = sum(1 for rec in records if rec["success"])
    print(f"\n{'=' * 70}")
    print("スイープ完了")
    print(f"  成功: {ok_count}/{total_runs}")
    print(f"  総所要時間: {sweep_elapsed/60:.1f}分 ({sweep_elapsed:.0f}秒)")
    print(f"  結果を保存しました:")
    print(f"    JSON: {out_file}")
    print(f"    サマリー: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
