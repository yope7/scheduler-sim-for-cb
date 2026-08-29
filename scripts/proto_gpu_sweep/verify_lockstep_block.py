"""lockstep のブロック協調版(lockstep_kernel_block)を既存版(lockstep_kernel)と突き合わせる。

既存版は verify_lockstep.py で CPU env と一致することが確認済みなので、ここが一致すれば
ブロック版も CPU env と一致する。行動列は事前生成して渡す(NN非依存の経路で比較)。

usage:
  source tools/cuda_env.sh
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:scripts/proto_gpu_sweep .venv/bin/python \
      scripts/proto_gpu_sweep/verify_lockstep_block.py [--big] [--b 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lockstep_kernel as LK  # noqa: E402
import lockstep_kernel_block as LKB  # noqa: E402
from verify_raw_rollout import gen_actions, gen_jobs  # noqa: E402

BIG_CFG = "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--b", type=int, default=2)
    args = ap.parse_args()

    if not args.big:
        n_jobs, n_on, n_cl = 128, 64, 256
        jobs = gen_jobs(n_jobs, 12345)
        e_max, k = 16384, 128
    else:
        from scripts.pcn_replay_snapshot import create_eval_env, load_config

        cfg = load_config(BIG_CFG)
        env = create_eval_env(cfg, job_seed=0, n_jobs=50000)
        env.reset()
        jobs = np.asarray(env.jobs, dtype=np.float64).copy()
        n_on, n_cl = int(env.n_on_premise_node), int(env.n_cloud_node)
        n_jobs = jobs.shape[0]
        e_max, k = 8192, 128

    p_list = [0.5] * args.b
    actions = gen_actions(n_jobs, p_list, 12346)
    print(f"[verify_lockstep_block] n_jobs={n_jobs} B={args.b} n_on={n_on} n_cl={n_cl}", flush=True)

    t0 = time.perf_counter()
    ref = LK.run_lockstep_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=k)
    t_ref = time.perf_counter() - t0
    print(f"[既存版 1スレッド/本] {t_ref:.2f}s ({t_ref*1000/n_jobs:.3f} ms/step)", flush=True)

    t0 = time.perf_counter()
    blk = LKB.run_lockstep_rollout_block(jobs, actions, n_on, n_cl, e_max=e_max, k=k)
    t_blk = time.perf_counter() - t0
    print(f"[ブロック版 {LKB.TPB}スレッド/本] {t_blk:.2f}s ({t_blk*1000/n_jobs:.3f} ms/step) "
          f"→ {t_ref/max(t_blk,1e-9):.2f}倍", flush=True)

    ok = True
    for key in ("start_times", "waits", "costs"):
        if key in ref and key in blk:
            n_mism = int((np.asarray(ref[key]) != np.asarray(blk[key])).sum())
            print(f"  {key}: mismatch={n_mism}")
            ok &= n_mism == 0
    if "obs" in ref and "obs" in blk:
        d = np.abs(np.asarray(ref["obs"], dtype=np.float64) - np.asarray(blk["obs"], dtype=np.float64))
        print(f"  obs: max|diff|={d.max():.3e} (完全一致={bool((d == 0).all())})")
        ok &= bool((d == 0).all())
    if "ovf" in blk:
        n_ovf = int((np.asarray(blk["ovf"]) != 0).sum())
        print(f"  ovf: {n_ovf}")
        ok &= n_ovf == 0
    print("[verify_lockstep_block] " + ("PASS: 全件一致" if ok else "FAIL: 差分あり"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
