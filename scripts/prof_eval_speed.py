#!/usr/bin/env python3
"""eval_uniform_command_pf の各フェーズ所要時間を計測（律速特定用）。"""
import os, time
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
import numpy as np, torch as th

def t(): return time.perf_counter()
T0 = t()
from scripts.pcn_replay_snapshot import (create_eval_env, load_config, eval_n_jobs,
    load_learner_replay_snapshot, episode_objectives_from_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN
print(f"[imports] {t()-T0:.1f}s")

CFG = "experiments/distributed_pcn/job_trace_1024_pcn.yml"
CKPT = "experiments/distributed_pcn/run1024_amplog_b/20260604_122948/iteration_100/model_iter_100.pth"
SNAP = "experiments/distributed_pcn/run1024_amplog_b/20260604_122948/learner_replay_snapshot.pkl.gz"

a = t(); config = load_config(CFG); print(f"[load_config] {t()-a:.1f}s")
a = t(); snap = load_learner_replay_snapshot(SNAP); print(f"[load_snapshot] {t()-a:.1f}s  meta={snap.get('metadata',{})}")
n_jobs = int(snap.get("metadata", {}).get("n_jobs", eval_n_jobs(config)))
print(f"  => n_jobs(eval) = {n_jobs}")
a = t(); env = create_eval_env(config, job_seed=0, n_jobs=n_jobs); print(f"[build_env] {t()-a:.1f}s  obs_dim={env.observation_space.shape[0]}")
a = t(); state = th.load(CKPT, map_location="cpu", weights_only=False)
h_scale = 1.0 / max(1, n_jobs)
agent = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
            scaling_factor=np.array([1.0,1.0,h_scale],dtype=np.float32), learning_rate=1e-3, batch_size=512,
            hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
            use_enhanced_model=(state.get("model_type","")=="EnhancedPCNModel"))
tgt = agent.network if agent.use_enhanced_model else agent.model
tgt.load_state_dict(state.get("model_state_dict", state), strict=False); tgt.eval()
print(f"[load_model+agent] {t()-a:.1f}s")
a = t(); arch = archive_pf_from_snapshot(snap, n_jobs); print(f"[archive_pf] {t()-a:.1f}s  n_archive_nd={len(arch)}")

# per-command timing: 3 commands
cost_max = 1.8e9; wait_max = 1.6e6
max_return = np.full(2, np.inf, dtype=np.float32)
for k in range(3):
    r1 = -cost_max * (0.3 + 0.3*k); r0 = -wait_max * n_jobs * 0.5
    dr = np.array([r0, r1], dtype=np.float32)
    a = t()
    out = agent._run_episode(env, dr.copy(), np.float32(n_jobs), max_return, eval_mode=True)
    val = out[5]
    print(f"[cmd {k}] {t()-a:.2f}s  achieved cost={val[0]:.3e} wait={val[1]:.3e}  (ep_len~{len(out[0])})")
print(f"[TOTAL] {t()-T0:.1f}s")
