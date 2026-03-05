#!/usr/bin/env python3
"""
分散PCNのスケーリングベンチマーク

複数の規模でプロファイリングを実行し、スケーリング特性を評価する。

使用方法:
  python scripts/scale_benchmark.py                    # 全スケール実行
  python scripts/scale_benchmark.py --scale large      # 大規模のみ
  python scripts/scale_benchmark.py --output results.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# スケール定義: (N_JOBS, n_on_premise, n_cloud, 説明)
# config.ymlの値を上書きするため、環境変数で渡す
SCALES = {
    "small": {
        "N_JOBS": 32,
        "n_on_premise_node": 256,
        "n_cloud_node": 1024,
        "desc": "小規模（従来）",
    },
    "medium": {
        "N_JOBS": 64,
        "n_on_premise_node": 512,
        "n_cloud_node": 2048,
        "desc": "中規模",
    },
    "large": {
        "N_JOBS": 128,
        "n_on_premise_node": 512,
        "n_cloud_node": 2048,
        "desc": "大規模（現行config）",
    },
}


def run_benchmark(scale_name: str, scale_config: dict) -> dict:
    """1スケールでベンチマークを実行"""
    env = os.environ.copy()
    env["DISTRIBUTED_PCN_PROFILE"] = "1"
    env["DISTRIBUTED_PCN_QUICK"] = "1"
    env["DISTRIBUTED_PCN_JOBS"] = str(scale_config["N_JOBS"])
    env["DISTRIBUTED_PCN_ONPREM"] = str(scale_config["n_on_premise_node"])
    env["DISTRIBUTED_PCN_CLOUD"] = str(scale_config["n_cloud_node"])

    cmd = [sys.executable, "-m", "src.distributed.distributed_pcn"]
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    wall_time = time.time() - start

    # 出力からタイミングを抽出
    output = result.stdout + result.stderr
    patterns = {
        "phase1_duration": r"フェーズ1完了.*?経過時間: ([\d.]+)秒",
        "phase2_duration": r"フェーズ2完了.*?経過時間: ([\d.]+)秒",
        "phase3_duration": r"フェーズ3完了.*?経過時間: ([\d.]+)秒",
        "phase1_episodes": r"フェーズ1完了.*?生成エピソード数: (\d+)",
        "env_step_ms": r"env\.step loop: ([\d.]+)s \((\d+) steps, ([\d.]+)ms/step\)",
        "get_episodes": r"get_episodes=([\d.]+)s",
        "add_episodes": r"add_episodes=([\d.]+)s",
        "update": r"update=([\d.]+)s",
        "actor_time": r"\[PROFILE Iter \d+\] Actor実行: ([\d.]+)s",
        "learner_time": r"\[PROFILE Iter \d+\] Learner実行: ([\d.]+)s",
    }

    extracted = {"scale": scale_name, "desc": scale_config["desc"], "wall_time": wall_time}
    if result.returncode != 0:
        extracted["error"] = result.stderr[-2000:] if result.stderr else "Unknown error"
        return extracted

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.DOTALL)
        if match:
            if key == "env_step_ms":
                extracted["env_step_total_s"] = float(match.group(1))
                extracted["env_steps"] = int(match.group(2))
                extracted["env_step_ms_per_step"] = float(match.group(3))
            elif key in ("phase1_episodes", "env_steps"):
                extracted[key] = int(match.group(1))
            else:
                extracted[key] = float(match.group(1))

    # 複数イテレーションの平均を取る
    actor_matches = re.findall(r"\[PROFILE Iter \d+\] Actor実行: ([\d.]+)s", output)
    if actor_matches:
        vals = [float(m) for m in actor_matches]
        extracted["actor_time_avg"] = sum(vals) / len(vals)
        extracted["actor_time_max"] = max(vals)
    learner_matches = re.findall(r"\[PROFILE Iter \d+\] Learner実行: ([\d.]+)s", output)
    if learner_matches:
        vals = [float(m) for m in learner_matches]
        extracted["learner_time_avg"] = sum(vals) / len(vals)
        extracted["learner_time_max"] = max(vals)
    get_ep_matches = re.findall(r"\[PROFILE Learner\] get_episodes=([\d.]+)s", output)
    if get_ep_matches:
        vals = [float(m) for m in get_ep_matches]
        extracted["get_episodes_avg"] = sum(vals) / len(vals)
    update_matches = re.findall(r"update=([\d.]+)s", output)
    if update_matches:
        vals = [float(m) for m in update_matches]
        extracted["update_avg"] = sum(vals) / len(vals)

    # 観測サイズの計算
    obs_size = (
        scale_config["n_on_premise_node"] * 30
        + scale_config["n_cloud_node"] * 30
        + 40
    )
    extracted["obs_size"] = obs_size
    extracted["n_jobs"] = scale_config["N_JOBS"]

    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=list(SCALES.keys()) + ["all"], default="all")
    parser.add_argument("--output", "-o", type=str, default="scale_benchmark_results.json")
    parser.add_argument("--repeat", type=int, default=1, help="各スケールの繰り返し回数")
    args = parser.parse_args()

    scales = SCALES if args.scale == "all" else {args.scale: SCALES[args.scale]}
    results = []

    for scale_name, scale_config in scales.items():
        print(f"\n{'='*60}")
        print(f"スケール: {scale_name} ({scale_config['desc']})")
        print(f"  N_JOBS={scale_config['N_JOBS']}, onprem={scale_config['n_on_premise_node']}, cloud={scale_config['n_cloud_node']}")
        print("=" * 60)

        for run in range(args.repeat):
            if args.repeat > 1:
                print(f"  Run {run+1}/{args.repeat}...")
            r = run_benchmark(scale_name, scale_config)
            if "error" in r:
                print(f"  ERROR: {r['error'][:500]}")
            else:
                print(f"  Wall time: {r['wall_time']:.1f}s")
                if "env_step_ms_per_step" in r:
                    print(f"  env.step: {r['env_step_ms_per_step']:.2f}ms/step")
                if "update_avg" in r:
                    print(f"  Learner update avg: {r['update_avg']:.3f}s")
            results.append(r)

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n結果を {args.output} に保存しました")

    # 簡易スケーリングレポート
    if len(results) >= 2 and not any("error" in r for r in results):
        print("\n--- スケーリング概要 ---")
        for r in results:
            if "wall_time" in r and "error" not in r:
                print(f"  {r['scale']}: {r['wall_time']:.1f}s, obs={r.get('obs_size',0)}")


if __name__ == "__main__":
    main()
