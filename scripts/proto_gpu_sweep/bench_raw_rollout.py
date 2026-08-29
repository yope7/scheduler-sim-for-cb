"""raw_rollout_kernel の速度実測(設計書 Task8)。

5万ジョブ(weekB head50000, cap 48000/192000)を B 本並列で回し、wall-clock と「本/分」を出す。
行動列は Bernoulli p を 0.2..1.0 の等間隔で B 本に散らして生成(混雑度のばらつきを含める)。

usage:
  CUDA_HOME=$PWD/tools/nvcc122/nvidia/cuda_nvcc CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      scripts/proto_gpu_sweep/bench_raw_rollout.py [--B 64,256,1024] [--tpb 32] \
      [--emax 16384] [--k 128] [--two-gpu]
  --two-gpu: B を半分ずつ CUDA_VISIBLE_DEVICES=0/1 の subprocess 2本に投げ、合算スループットを出す。
  (内部利用) --worker: subprocess 側のエントリ。単一デバイスで B 本回して1行 JSON を出力。

比較基準(設計書): CPU64コア=11本/分, 既存GPU工場=2.5本/分。
注: numba CUDA はドライバ(CUDA 12.2)対応の nvvm が必要。/usr/local/cuda が 12.5 の
このマシンはドライバが CUDA 12.2 / システムが 12.5 で PTX が不整合になるため、12.2 の
nvcc 断片(tools/nvcc122)へ CUDA_HOME を向ける。未構築なら tools/setup_nvcc122.sh を実行。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

BIG_CFG = "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml"
N_JOBS = 50000
SEED = 12345


def load_trace_jobs():
    from scripts.pcn_replay_snapshot import load_config, create_eval_env

    cfg = load_config(BIG_CFG)
    env = create_eval_env(cfg, job_seed=0, n_jobs=N_JOBS)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    return jobs, int(env.n_on_premise_node), int(env.n_cloud_node)


def gen_actions_spread(n_jobs: int, B: int, seed: int, p_lo=0.2, p_hi=1.0) -> np.ndarray:
    """行動列: p を [p_lo, p_hi] の等間隔で B 本に割り振った Bernoulli。"""
    rng = np.random.default_rng(seed)
    ps = np.linspace(p_lo, p_hi, B)
    return (rng.random((B, n_jobs)) < ps[:, None]).astype(np.int8)


def bench_one(jobs, n_on, n_cl, B, e_max, k, tpb, warmup=True):
    from raw_rollout_kernel import run_raw_rollout

    actions = gen_actions_spread(jobs.shape[0], B, SEED)
    if warmup:
        # JITコンパイル+CUDA初期化をB=1の1本(短くはないが正確な計測のため)で先に済ませる…
        # は5万ジョブでは高価すぎるので、先頭256ジョブのミニ問題でコンパイルだけ暖める。
        run_raw_rollout(jobs[:256], actions[:1, :256], n_on, n_cl, e_max=1024, k=k, tpb=tpb)
    t0 = time.time()
    out = run_raw_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=k, tpb=tpb)
    dt = time.time() - t0
    return dt, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", default="64,256,1024")
    ap.add_argument("--tpb", type=int, default=1,
                    help="1エピソード=1スレッドで分岐が完全にバラけるため既定1(warp内分岐ゼロ)")
    ap.add_argument("--emax", type=int, default=16384)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--out", default="", help="結果1行/点を追記するファイル(空なら追記しない)")
    ap.add_argument("--two-gpu", action="store_true")
    ap.add_argument("--worker", action="store_true", help="(internal) single-device worker")
    args = ap.parse_args()
    b_list = [int(x) for x in args.B.split(",")]

    def _append(line: str) -> None:
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")

    jobs, n_on, n_cl = load_trace_jobs()

    if args.worker:
        # subprocess 側: 1デバイスで各Bを回し JSON 1行/B を stdout
        for B in b_list:
            dt, out = bench_one(jobs, n_on, n_cl, B, args.emax, args.k, args.tpb)
            print(json.dumps(dict(
                B=B, sec=dt, ovf=int(out["ovf"].sum()),
                peak_on=int(out["peak_ev_on"].max()), peak_cl=int(out["peak_ev_cl"].max()),
            )), flush=True)
        return 0

    if args.two_gpu:
        # B を半分ずつ 2 デバイスへ(subprocess 2本; ホスト側の総和スループット)
        print(f"[bench 2GPU] n_jobs={N_JOBS} B={b_list} tpb={args.tpb} E_MAX={args.emax} K={args.k}")
        for B in b_list:
            half = B // 2
            procs = []
            t0 = time.time()
            for dev in (0, 1):
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(dev)
                procs.append(subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__), "--worker",
                     "--B", str(half), "--tpb", str(args.tpb),
                     "--emax", str(args.emax), "--k", str(args.k)],
                    stdout=subprocess.PIPE, env=env,
                    cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                ))
            results = []
            failed = False
            for pr in procs:
                out_b, _ = pr.communicate()
                if pr.returncode != 0:
                    print(f"  worker rc={pr.returncode}")
                    failed = True
                    continue
                results.append(json.loads(out_b.decode().strip().splitlines()[-1]))
            if failed:
                for pr in procs:  # 生き残りを後始末(失敗時に相方を放置しない)
                    if pr.poll() is None:
                        pr.kill()
                return 1
            wall = time.time() - t0  # 2プロセスのロード込み(参考)
            kern = max(r["sec"] for r in results)  # カーネル時間はワーカー内計測の遅い方
            eps_min = B / kern * 60.0
            print(f"  B={B:5d} (={half}x2): kernel={kern:.1f}s wall(incl load)={wall:.1f}s "
                  f"-> {eps_min:.1f} 本/分  ovf={sum(r['ovf'] for r in results)} "
                  f"peak_on={max(r['peak_on'] for r in results)} "
                  f"peak_cl={max(r['peak_cl'] for r in results)}")
            _append(f"B={B} wall={kern:.1f}s eps_per_min={eps_min:.1f} mode=2gpu tpb={args.tpb}")
        return 0

    print(f"[bench] n_jobs={N_JOBS} n_on={n_on} n_cl={n_cl} B={b_list} "
          f"tpb={args.tpb} E_MAX={args.emax} K={args.k}")
    for B in b_list:
        dt, out = bench_one(jobs, n_on, n_cl, B, args.emax, args.k, args.tpb)
        eps_min = B / dt * 60.0
        print(f"  B={B:5d}: {dt:.1f}s -> {eps_min:.1f} 本/分 "
              f"({dt / B:.2f} 秒/本換算, thread実効 {dt:.1f}s/エピソード) "
              f"ovf={int(out['ovf'].sum())} "
              f"peak_ev_on={int(out['peak_ev_on'].max())} peak_ev_cl={int(out['peak_ev_cl'].max())}",
              flush=True)
        _append(f"B={B} wall={dt:.1f}s eps_per_min={eps_min:.1f} tpb={args.tpb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
