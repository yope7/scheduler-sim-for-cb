#!/usr/bin/env python3
"""ユーザ仮説の検証: 低wait(高クラウド) target で方策がクラウド(action=1)を
   序盤に"早食い"して後半0しか選べなくなっているか、を行動系列で確認する。"""
import os, time
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from scripts.pcn_replay_snapshot import (create_eval_env, load_config, eval_n_jobs,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import dedupe_pf, objectives_to_command

CFG="experiments/distributed_pcn/job_trace_1024_pcn.yml"
CKPT="experiments/distributed_pcn/run1024_amplog_b/20260604_122948/iteration_100/model_iter_100.pth"
SNAP="experiments/distributed_pcn/run1024_amplog_b/20260604_122948/learner_replay_snapshot.pkl.gz"

config=load_config(CFG); snap=load_learner_replay_snapshot(SNAP)
n_jobs=int(snap.get("metadata",{}).get("n_jobs", eval_n_jobs(config)))
env=create_eval_env(config, job_seed=0, n_jobs=n_jobs)
state=th.load(CKPT, map_location="cpu", weights_only=False)
agent=PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
          scaling_factor=np.array([1.0,1.0,1.0/max(1,n_jobs)],dtype=np.float32), learning_rate=1e-3,
          batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
          use_enhanced_model=(state.get("model_type","")=="EnhancedPCNModel"))
tgt=agent.network if agent.use_enhanced_model else agent.model
tgt.load_state_dict(state.get("model_state_dict", state), strict=False); tgt.eval()

arch=dedupe_pf(archive_pf_from_snapshot(snap, n_jobs))
order=np.argsort(arch[:,1])  # by wait
picks={"low_wait(high cost)":arch[order[0]], "mid":arch[order[len(order)//2]], "high_wait(low cost)":arch[order[-1]]}
max_return=np.full(2, np.inf, dtype=np.float32)

fig, ax=plt.subplots(figsize=(10,6))
print(f"n_jobs={n_jobs} archive_nd={len(arch)}")
for lbl,(ct,wt) in picks.items():
    dr=objectives_to_command(float(ct),float(wt),n_jobs).astype(np.float32)
    out=agent._run_episode(env, dr, np.float32(n_jobs), max_return, eval_mode=True)
    acts=np.array([int(t.action) for t in out[0]])
    ca,wa=float(out[5][0]), float(out[5][1])
    cloud=(acts==1).astype(float)
    cum=np.cumsum(cloud)
    x=np.linspace(0,1,len(acts))
    third=max(1,len(acts)//3)
    er,lr=cloud[:third].mean(), cloud[-third:].mean()
    print(f"[{lbl}] target(cost={ct:.2e},wait={wt:.2e}) -> achieved(cost={ca:.2e},wait={wa:.2e})  "
          f"cloud_rate early={er:.2f} late={lr:.2f}  total_cloud={cloud.mean():.2f}  ep_len={len(acts)}")
    ax.plot(x, cum/max(1,cum[-1]), lw=2, label=f"{lbl}: early={er:.2f} late={lr:.2f}")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.6,label="uniform pacing (ideal spread)")
ax.set_xlabel("episode progress (job index / total)"); ax.set_ylabel("cumulative cloud(=1) fraction")
ax.set_title("Is the policy 'spending' cloud early? (curve above diagonal = front-loaded = greedy)\namplog_b iter100")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("pf_1024_action_timing.png", dpi=115, bbox_inches="tight")
print("saved pf_1024_action_timing.png")
