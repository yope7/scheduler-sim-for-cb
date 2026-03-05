#!/usr/bin/env python3
"""
distributed_pcn.py を外側から実行し、
ジョブ数×ノード構成の組み合わせでベンチマークを実施し、実行時間を記録する。

組み合わせ:
  - ジョブ数: 16, 32, 64
  - ノード構成:
    - オンプレ: on-prem=256, cloud=0
    - (256, 1024): on-prem=256, cloud=1024
    - (512, 2048): on-prem=512, cloud=2048

使用方法:
  python scripts/run_distributed_pcn_sweep.py
  python scripts/run_distributed_pcn_sweep.py --quick   # 短時間モードでテスト
  python scripts/run_distributed_pcn_sweep.py -o results.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# ジョブ数の候補
JOB_COUNTS = [16, 32, 64]

# ノード構成: (名前, on_premise, cloud)
NODE_CONFIGS = [
    ("オンプレ", 256, 0),
    ("クラウド(256,1024)", 256, 1024),
    ("クラウド(512,2048)", 512, 2048),
]


def run_distributed_pcn(
    n_jobs: int,
    n_on_prem: int,
    n_cloud: int,
    quick: bool = False,
    timeout: int = 3600,
) -> dict:
    """
    distributed_pcn をサブプロセスで実行し、実行時間を返す。
    """
    env = os.environ.copy()
    env["DISTRIBUTED_PCN_JOBS"] = str(n_jobs)
    env["DISTRIBUTED_PCN_ONPREM"] = str(n_on_prem)
    env["DISTRIBUTED_PCN_CLOUD"] = str(n_cloud)
    if quick:
        env["DISTRIBUTED_PCN_QUICK"] = "1"

    cmd = [sys.executable, "-m", "src.distributed.distributed_pcn"]
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "success": False,
            "elapsed_sec": elapsed,
            "error": "timeout",
        }
    elapsed = time.time() - start

    if result.returncode != 0:
        err_msg = (result.stderr or result.stdout or "")[-2000:]
        return {
            "success": False,
            "elapsed_sec": elapsed,
            "returncode": result.returncode,
            "error": err_msg,
        }

    # フェーズごとの時間を出力から抽出
    output = (result.stdout or "") + (result.stderr or "")
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
        default=3600,
        help="1実行あたりのタイムアウト秒（デフォルト: 3600）",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"execution_times_{timestamp}.json"

    records = []
    total_runs = len(JOB_COUNTS) * len(NODE_CONFIGS)
    run_idx = 0

    print("=" * 70)
    print("distributed_pcn ジョブ数×ノード構成スイープ")
    print(f"  ジョブ数: {JOB_COUNTS}")
    print(f"  ノード構成: {[nc[0] for nc in NODE_CONFIGS]}")
    print(f"  クイックモード: {args.quick}")
    print(f"  出力先: {output_path}")
    print("=" * 70)

    for n_jobs in JOB_COUNTS:
        for node_name, n_on_prem, n_cloud in NODE_CONFIGS:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] n_jobs={n_jobs}, {node_name} (onprem={n_on_prem}, cloud={n_cloud})")
            r = run_distributed_pcn(
                n_jobs=n_jobs,
                n_on_prem=n_on_prem,
                n_cloud=n_cloud,
                quick=args.quick,
                timeout=args.timeout,
            )
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
            print(f"  -> {status} 経過時間: {r['elapsed_sec']:.1f}秒{phase_str}")

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

    print(f"\n結果を保存しました:")
    print(f"  JSON: {out_file}")
    print(f"  サマリー: {summary_path}")


if __name__ == "__main__":
    main()
