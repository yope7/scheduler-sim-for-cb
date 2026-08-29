"""lockstep の B スケーリング実測(「定額=B非依存」が本当かの検証)。

同一 checkpoint・同一ジョブ列で run_lockstep_greedy を B を変えて回し、
wall / us-per-step / 1エピソードあたり秒 を出す。T(=n_jobs)も指定できるので、
短い T で形(スケーリング曲線)を掴んでから実寸 T=50000 で確認する運用を想定。

usage:
  CKPT=<ckpt.pth> CUDA_HOME=$PWD/tools/nvcc122/nvidia/cuda_nvcc CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:scripts/proto_gpu_sweep \
      .venv/bin/python scripts/proto_gpu_sweep/bench_lockstep_bscale.py --nj 5000 --blist 16,64,256
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
os.environ.setdefault("SCHEDULER_OBS_EFFICIENCY", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("PCN_OBS_LOG", "1")
os.environ.setdefault("PCN_FOURIER_CMD", "1")
os.environ.setdefault("PCN_FC_DEPTH", "4")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get(
    "CKPT",
    "experiments/distributed_pcn/run_j50000_gpu_v6/20260824_033100/iteration_014/model_iter_014.pth",
)


def build_commands(ckpt_path: str, b: int, zlo: float = -2.0, zhi: float = -0.2) -> np.ndarray:
    """checkpoint の指令正規化バッファから学習分布内の指令 B 本(z=等間隔・負側)。"""
    st = th.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    center = sd["desired_return_center"].numpy().astype(np.float32)
    scale = sd["desired_return_scale"].numpy().astype(np.float32)
    zs = np.linspace(zlo, zhi, b, dtype=np.float32)
    return np.stack([center + z * scale for z in zs]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nj", type=int, default=5000)
    ap.add_argument("--blist", type=str, default="16,64,256")
    ap.add_argument("--emax", type=int, default=65536)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--return-obs", action="store_true",
                    help="本番同様に obs を記録する(帯域・メモリを含めた実コスト)")
    ap.add_argument("--zlo", type=float, default=-2.0)
    ap.add_argument("--zhi", type=float, default=-0.2,
                    help="zhi を 0 に近づけると cheap端(混雑)側の重い指令が入る")
    ap.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sample"])
    args = ap.parse_args()

    from scripts.pcn_replay_snapshot import load_config, create_eval_env  # noqa: E402
    from lockstep_nn import load_policy_model, run_lockstep_greedy  # noqa: E402

    cfg = load_config(
        "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml")
    env = create_eval_env(cfg, job_seed=0, n_jobs=50000)
    env.reset()
    jobs_full = np.asarray(env.jobs, dtype=np.float64).copy()
    n_on = int(env.n_on_premise_node)
    n_cl = int(env.n_cloud_node)

    nj = args.nj
    jobs = np.ascontiguousarray(jobs_full[:nj])
    model = load_policy_model(CKPT, 50000, device="cuda")
    print(f"[bench] nj={nj} e_max={args.emax} k={args.k} mode={args.mode} "
          f"obs={args.return_obs} z=[{args.zlo},{args.zhi}]", flush=True)

    # warmup(JITコンパイル)を計測から外す
    run_lockstep_greedy(jobs[:256], model, build_commands(CKPT, 1), n_on, n_cl,
                        n_window=100, horizons=np.array([256.0], dtype=np.float32),
                        e_max=2048, k=args.k, tpb=1, mode="greedy", return_obs=False)

    print(f"{'B':>6} {'wall_s':>9} {'us/step':>10} {'s/ep':>9} {'ep/min':>9}", flush=True)
    for b in [int(x) for x in args.blist.split(",") if x.strip()]:
        cmds = build_commands(CKPT, b, args.zlo, args.zhi)
        hz = np.full(b, float(nj), dtype=np.float32)
        th.cuda.synchronize()
        t0 = time.time()
        out = run_lockstep_greedy(
            jobs, model, cmds, n_on, n_cl, n_window=100, horizons=hz,
            e_max=args.emax, k=args.k, tpb=1, mode=args.mode,
            return_obs=args.return_obs)
        th.cuda.synchronize()
        dt = time.time() - t0
        ovf = int(np.asarray(out["ovf"]).sum()) if "ovf" in out else -1
        cf = float(np.asarray(out["actions"]).mean()) if "actions" in out else float("nan")
        pk_on = int(np.asarray(out["peak_ev_on"]).max()) if "peak_ev_on" in out else -1
        pk_cl = int(np.asarray(out["peak_ev_cl"]).max()) if "peak_ev_cl" in out else -1
        mem = th.cuda.max_memory_allocated() / 2**30
        print(f"{b:>6} {dt:>9.1f} {dt / nj * 1e6:>10.1f} {dt / b:>9.2f} "
              f"{b / dt * 60:>9.1f}   ovf={ovf} cloud_frac={cf:.3f} peakGB={mem:.1f} "
              f"peak_ev=({pk_on},{pk_cl})", flush=True)
        del out
        th.cuda.empty_cache(); th.cuda.reset_peak_memory_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
