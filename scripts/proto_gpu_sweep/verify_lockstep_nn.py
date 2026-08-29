"""verify_lockstep_nn — B-3(NN接続)の検証器。

同一 checkpoint・同一指令(desired_return, horizon)で
  (a) CPUリファレンス: pcn_agent._run_episode(eval_mode=True, greedy) を実trace env で実行
  (b) GPUロックステップ: lockstep_nn.run_lockstep_greedy (obs構築+配置+NN全てGPU)
の行動列と目的値 (cost, avg_wait) を突き合わせる。

合格条件(設計書 B-3): 行動列・(cost,wait) の完全一致が理想。fp32 演算順序差
(CPU BLAS vs cuBLAS)で行動が割れる場合は「行動一致率>99.9% かつ目的値の相対差<1e-5」
を合格とし、差異事例(step, CPUロジット差)を報告する。

checkpoint: run_j50000_gpu_v5 iter14 (obs_dim=224, 5万ジョブ用の観測正規化なので
検証も5万ジョブ trace で行う。--nj で短縮スモーク可能だが正規化がズレるため参考値)。

usage:
  CUDA_HOME=$PWD/tools/nvcc122/nvidia/cuda_nvcc CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      scripts/proto_gpu_sweep/verify_lockstep_nn.py [--nj 50000] [--b 2] [--skip-cpu]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 観測+モデルの環境変数は env/pcn_agent の import 前に確定させる(学習時条件、
# eval_jscale_c3.sh のヘッダと同一)。
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
os.environ.setdefault("SCHEDULER_OBS_EFFICIENCY", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("PCN_FOURIER_CMD", "1")
os.environ.setdefault("PCN_FC_DEPTH", "4")
os.environ.setdefault("PCN_COND_ADD_SCALE", "0.25")
os.environ.setdefault("PCN_COMMAND_BALANCE", "1")
os.environ.setdefault("PCN_OBS_LOG", "1")

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CKPT = os.environ.get(
    "CKPT",
    "experiments/distributed_pcn/run_j50000_gpu_v5/20260821_213149/iteration_014/model_iter_014.pth",
)
CFG = os.environ.get(
    "CFG", "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml"
)


def build_commands(ckpt_path: str, b: int) -> np.ndarray:
    """checkpoint の指令正規化バッファから学習分布内の指令 B 本を作る(z=等間隔)。

    dr の規約は [-wait*nj, -cost](objectives_to_command)で常に負。center=0 の
    checkpoint では z は負のみ使う(正の dr は分布外の無意味な指令になる)。
    """
    st = th.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    center = sd["desired_return_center"].numpy().astype(np.float64)
    scale = sd["desired_return_scale"].numpy().astype(np.float64)
    zs = np.linspace(-0.6, -0.15, b)
    cmds = np.stack([center + z * scale for z in zs]).astype(np.float32)
    print(f"[cmd] center={center.tolist()} scale={scale.tolist()}")
    for i, c in enumerate(cmds):
        print(f"[cmd] {i}: dr={c.tolist()} (z={zs[i]:+.2f})")
    return cmds


def run_cpu_reference(env, ckpt_path: str, commands: np.ndarray, nj: int):
    """eval_b2_compare._winit/_ep と同一手順の CPU リファレンス(行動列も回収)。"""
    from src.agents.pcn_agent import PCN

    st = th.load(ckpt_path, map_location="cpu", weights_only=False)
    ag = PCN(
        env, device="cpu", state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1.0, 1.0, 1.0 / max(1, nj)], dtype=np.float32),
        learning_rate=1e-3, batch_size=512,
        hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
        project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False,
    )
    tg = ag.model
    tg.load_state_dict(st.get("model_state_dict", st), strict=False)
    tg.eval()
    mx = np.full(2, np.inf, dtype=np.float32)

    actions_all, objs = [], []
    for i, dr in enumerate(commands):
        t0 = time.time()
        r = ag._run_episode(env, np.asarray(dr, dtype=np.float32), np.float32(nj), mx,
                            eval_mode=True)
        acts = np.array([int(t.action) for t in r[0]], dtype=np.int8)
        cost, wait = float(r[5][0]), float(r[5][1])
        print(f"[cpu] cmd={i}: {time.time()-t0:.1f}s cost={cost:.6g} wait={wait:.6g} "
              f"cloud_frac={acts.mean():.4f}", flush=True)
        actions_all.append(acts)
        objs.append([cost, wait])
    return np.stack(actions_all), np.array(objs)


def bench_nn_forward(model, b: int, obs_dim: int, device: str = "cuda", iters: int = 1000):
    """JIT forward 単体のバッチ実行時間(μs/step)を CUDA イベントで実測。"""
    obs = th.rand((b, obs_dim), dtype=th.float32, device=device)
    dr = th.zeros((b, 2), dtype=th.float32, device=device)
    h = th.full((b, 1), 100.0, dtype=th.float32, device=device)
    with th.no_grad():
        jit = th.jit.trace(model, (obs, dr, h))
        for _ in range(50):
            jit(obs, dr, h)
        th.cuda.synchronize()
        ev0, ev1 = th.cuda.Event(True), th.cuda.Event(True)
        ev0.record()
        for _ in range(iters):
            jit(obs, dr, h)
        ev1.record()
        th.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1000.0 / iters  # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nj", type=int, default=50000)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--skip-cpu", action="store_true",
                    help="GPU経路のみ実行(スモーク/計時用)")
    args = ap.parse_args()

    from scripts.pcn_replay_snapshot import create_eval_env, load_config
    from lockstep_nn import load_policy_model, run_lockstep_greedy

    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=0, n_jobs=args.nj)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    n_on = int(env.n_on_premise_node)
    n_cl = int(env.n_cloud_node)
    n_window = int(env.n_window)
    nj = jobs.shape[0]
    obs_dim = int(env.observation_space.shape[0])
    print(f"[verify_nn] ckpt={CKPT}")
    print(f"[verify_nn] nj={nj} n_on={n_on} n_cl={n_cl} n_window={n_window} "
          f"obs_dim={obs_dim} B={args.b}")

    commands = build_commands(CKPT, args.b)

    # --- (b) GPU ロックステップ ---
    model = load_policy_model(CKPT, nj, device="cuda")
    # ウォームアップ(numba JIT コンパイル: 先頭256ジョブのミニ問題)
    run_lockstep_greedy(jobs[:256], model, commands[:1], n_on, n_cl,
                        n_window=n_window, e_max=1024, k=16)
    timing = {}
    t0 = time.time()
    gpu = run_lockstep_greedy(jobs, model, commands, n_on, n_cl,
                              n_window=n_window, e_max=16384, k=128, timing=timing,
                              progress=10000)
    print(f"[gpu] lockstep+NN: {time.time()-t0:.1f}s "
          f"(loop {timing['total_s']:.1f}s = {timing['per_step_us']:.0f}us/step, B={args.b}) "
          f"ovf={gpu['ovf'].tolist()}", flush=True)
    if gpu["ovf"].any():
        print(f"[verify_nn] FAIL: ovf set {gpu['ovf'].tolist()}")
        return 1
    for i in range(args.b):
        print(f"[gpu] cmd={i}: cost={gpu['objectives'][i,0]:.6g} "
              f"wait={gpu['objectives'][i,1]:.6g} cloud_frac={gpu['actions'][i].mean():.4f}")

    nn_us = bench_nn_forward(model, args.b, obs_dim)
    print(f"[gpu] NN forward単体(JIT, B={args.b}): {nn_us:.1f}us/step")

    if args.skip_cpu:
        print("[verify_nn] --skip-cpu: 比較なしで終了")
        return 0

    # --- (a) CPU リファレンス ---
    print(f"[cpu] running {args.b} reference episodes (数分/本)...", flush=True)
    cpu_actions, cpu_objs = run_cpu_reference(env, CKPT, commands, nj)

    # --- 突き合わせ ---
    ok = True
    for i in range(args.b):
        ga, ca = gpu["actions"][i], cpu_actions[i]
        n_mism = int((ga != ca).sum())
        match_rate = 1.0 - n_mism / nj
        go, co = gpu["objectives"][i], cpu_objs[i]
        rel_cost = abs(go[0] - co[0]) / max(1.0, abs(co[0]))
        rel_wait = abs(go[1] - co[1]) / max(1.0, abs(co[1]))
        exact = (n_mism == 0) and (rel_cost == 0.0) and (rel_wait == 0.0)
        loose = (match_rate > 0.999) and (rel_cost < 1e-5) and (rel_wait < 1e-5)
        verdict = "EXACT" if exact else ("PASS(loose)" if loose else "FAIL")
        print(f"[cmp] cmd={i}: action match {nj-n_mism}/{nj} ({match_rate*100:.4f}%) "
              f"cost gpu={go[0]:.6g} cpu={co[0]:.6g} (rel {rel_cost:.2e}) "
              f"wait gpu={go[1]:.6g} cpu={co[1]:.6g} (rel {rel_wait:.2e}) -> {verdict}")
        if n_mism:
            idx = np.flatnonzero(ga != ca)
            print(f"  差異step(先頭10): {idx[:10].tolist()} (total {n_mism})")
        if not (exact or loose):
            ok = False

    print("\n[verify_lockstep_nn] " + ("PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
