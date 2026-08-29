"""ブロック協調版カーネル(raw_rollout_kernel_block)を既存版(raw_rollout_kernel)と突き合わせる。

既存版の出力を正解として (start_times, waits, costs) を全件比較する。既存版自体は
verify_raw_rollout.py で CPU env と全件一致することが確認済みなので、ここが一致すれば
ブロック版も CPU env と一致する。

usage:
  source tools/cuda_env.sh
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:scripts/proto_gpu_sweep .venv/bin/python \
      scripts/proto_gpu_sweep/verify_raw_rollout_block.py [--big] [--tpb 128]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from raw_rollout_kernel import run_raw_rollout  # noqa: E402
from raw_rollout_kernel_block import run_raw_rollout_block  # noqa: E402
from verify_raw_rollout import gen_actions, gen_jobs  # noqa: E402

BIG_CFG = "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml"


def compare(ref, blk, label):
    ok = True
    for key in ("start_times", "waits", "costs"):
        n_mism = int((ref[key] != blk[key]).sum())
        print(f"  {label} {key}: mismatch={n_mism}")
        ok &= n_mism == 0
    if (blk["ovf"] != 0).any():
        print(f"  {label} ovf={blk['ovf'].tolist()}")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="実trace 5万ジョブで比較(速度も測る)")
    ap.add_argument("--tpb", type=int, default=128)
    ap.add_argument("--b", type=int, default=4)
    args = ap.parse_args()

    if not args.big:
        n_jobs, n_on, n_cl = 128, 64, 256
        jobs = gen_jobs(n_jobs, 12345)
        p_list = [0.3, 0.5, 0.7, 1.0]
        actions = gen_actions(n_jobs, p_list, 12346)
        e_max, k = 16384, 128
    else:
        from scripts.pcn_replay_snapshot import create_eval_env, load_config

        cfg = load_config(BIG_CFG)
        env = create_eval_env(cfg, job_seed=0, n_jobs=50000)
        env.reset()
        jobs = np.asarray(env.jobs, dtype=np.float64).copy()
        n_on, n_cl = int(env.n_on_premise_node), int(env.n_cloud_node)
        n_jobs = jobs.shape[0]
        p_list = [0.5] * args.b
        actions = gen_actions(n_jobs, p_list, 12346)
        e_max, k = 8192, 128

    print(f"[verify_block] n_jobs={n_jobs} B={len(p_list)} n_on={n_on} n_cl={n_cl} "
          f"tpb={args.tpb} e_max={e_max}", flush=True)

    t0 = time.perf_counter()
    ref = run_raw_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=k)
    t_ref = time.perf_counter() - t0
    print(f"[既存版 1スレッド/本] {t_ref:.2f}s  ({t_ref*1000/n_jobs:.3f} ms/step)", flush=True)

    t0 = time.perf_counter()
    blk = run_raw_rollout_block(jobs, actions, n_on, n_cl, e_max=e_max, k=k, tpb=args.tpb)
    t_blk = time.perf_counter() - t0
    print(f"[ブロック版 {args.tpb}スレッド/本] {t_blk:.2f}s  ({t_blk*1000/n_jobs:.3f} ms/step)  "
          f"→ {t_ref/max(t_blk,1e-9):.2f}倍", flush=True)

    ok = compare(ref, blk, "[cmp]")
    print("[verify_block] " + ("PASS: 全件一致" if ok else "FAIL: 差分あり"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
