#!/usr/bin/env python3
"""b2(fourier) Eval（FiLM条件付き greedy ロールアウト）の単体ベンチ + プロファイル。

eval_uniform_command_pf.run と同じ「コマンドごとに _run_episode を逐次回す」評価の中身を計測する。
DEVICE=cuda で ms/command と forward/env.step の比率（torch.profiler）を出す。
NJ を変えてスケール依存（ジョブ10倍）も見る。

usage:
  NJ=128 G=64 DEVICE=cuda PYTHONPATH=. .venv/bin/python scripts/bench_eval_b2_128.py
  TORCH_PROFILE=1 NJ=128 G=24 DEVICE=cuda PYTHONPATH=. .venv/bin/python scripts/bench_eval_b2_128.py
"""
import os, sys, time
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
os.environ.setdefault("PCN_FOURIER_CMD", "1")
os.environ.setdefault("PCN_FOURIER_BANDS", "4")
os.environ.setdefault("PCN_FILM", "1")
os.environ.setdefault("PCN_OBS_LOG", "1")

import numpy as np
import torch as th
from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import PCN

NJ = int(os.environ.get("NJ", "128"))
G = int(os.environ.get("G", "64"))
DEVICE = os.environ.get("DEVICE", "cuda" if th.cuda.is_available() else "cpu")
EXEC = os.environ.get("EXEC", "experiments/distributed_pcn/run_synth128_fourier128/20260606_020951")
CKPT = os.environ.get("CKPT", f"{EXEC}/iteration_100/model_iter_100.pth")
CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_synthetic_pcn.yml")

config = load_config(CFG)
env = create_eval_env(config, job_seed=0, n_jobs=NJ)
state = th.load(CKPT, map_location="cpu", weights_only=False)
ag = PCN(env, device=DEVICE, state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32),
         learning_rate=1e-3, batch_size=512, hidden_dim=512,
         project_name="t", experiment_name="bench", log=False, use_enhanced_model=False)
tg = ag.model
tg.load_state_dict(state["model_state_dict"], strict=False)
tg.eval()

# uniform 風コマンド: cost を linspace、wait は固定端、horizon=NJ
scale = tg.desired_return_scale.detach().cpu().numpy().astype(np.float64)
cost_max = float(scale[1]) if scale[1] > 0 else 90000.0
wait_max_tot = float(scale[0]) if scale[0] > 0 else 220.0 * NJ
r1_grid = np.linspace(0.0, -cost_max * 1.10, G)
cmds = [(-wait_max_tot, float(r1), float(NJ)) for r1 in r1_grid]
maxret = np.full(2, np.inf, dtype=np.float32)

def run_commands(n):
    out = []
    for r0, r1, hz in cmds[:n]:
        dr = np.array([r0, r1], dtype=np.float32)
        res = ag._run_episode(env, dr.copy(), np.float32(hz), maxret, eval_mode=True)
        out.append(res[5])  # [cost, avgwait]
    return out

# warmup
run_commands(min(4, G))
if DEVICE != "cpu":
    th.cuda.synchronize()

if os.environ.get("TORCH_PROFILE") == "1":
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE != "cpu" else [])
    with profile(activities=acts) as prof:
        run_commands(G)
        if DEVICE != "cpu":
            th.cuda.synchronize()
    key = "self_cuda_time_total" if DEVICE != "cpu" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=22))
    raise SystemExit(0)

t0 = time.perf_counter()
vals = run_commands(G)
if DEVICE != "cpu":
    th.cuda.synchronize()
dt = time.perf_counter() - t0
ep_steps = NJ  # 近似（1ジョブ1ステップ）
import hashlib
# 達成点(cost,avgwait)を丸めずビット列でハッシュ → fast=0/1 で完全一致を確認（結果不変）
ach = np.asarray(vals, dtype=np.float64)
ahash = hashlib.sha1(np.ascontiguousarray(ach).tobytes()).hexdigest()[:16]
print(f"RESULT fast_env={os.environ.get('PCN_FAST_ENV','1')} device={DEVICE} NJ={NJ} G={G} "
      f"total={dt:.3f}s per_cmd_ms={dt / G * 1000:.2f} per_step_ms~{dt / G / ep_steps * 1000:.3f} "
      f"cost[min,max]=[{min(v[0] for v in vals):.0f},{max(v[0] for v in vals):.0f}] "
      f"ahash={ahash}", flush=True)
