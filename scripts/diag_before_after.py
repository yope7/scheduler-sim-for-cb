#!/usr/bin/env python3
"""before/after フロント比較: amplog_b iter100 (urgency OFF, 220) vs urgency_a iterXX (urgency ON, 221)。
   同じ (cost,wait) target 群を両方策に与え、到達点を重ねる。下/左に寄るほど良い。"""
import os, glob
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import (create_eval_env, load_config,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.utils.pf_command_eval import dedupe_pf, stratified_sample_pf, objectives_to_command

CFG="experiments/distributed_pcn/job_trace_1024_pcn.yml"
AMP="experiments/distributed_pcn/run1024_amplog_b/20260604_122948/iteration_100/model_iter_100.pth"
AMP_SNAP="experiments/distributed_pcn/run1024_amplog_b/20260604_122948/learner_replay_snapshot.pkl.gz"
URG=os.environ.get("DIAG_URG_CKPT") or sorted(glob.glob("experiments/distributed_pcn/run1024_urgency_a/*/iteration_0*/model_iter_0*.pth"))[-1]
N_JOBS=24
config=load_config(CFG)
snap=load_learner_replay_snapshot(AMP_SNAP)
arch=dedupe_pf(archive_pf_from_snapshot(snap, N_JOBS))
targets=stratified_sample_pf(arch, 9, rng=np.random.default_rng(1), include_extremes=True)
print(f"urgency ckpt = {URG}")

def eval_policy(ckpt, urgency_on):
    os.environ["SCHEDULER_OBS_URGENCY"]="1" if urgency_on else "0"
    env=create_eval_env(config, job_seed=0, n_jobs=N_JOBS)
    state=th.load(ckpt, map_location="cpu", weights_only=False)
    ag=PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
           scaling_factor=np.array([1.0,1.0,1.0/max(1,N_JOBS)],dtype=np.float32), learning_rate=1e-3,
           batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
           use_enhanced_model=(state.get("model_type","")=="EnhancedPCNModel"))
    t=ag.network if ag.use_enhanced_model else ag.model
    t.load_state_dict(state.get("model_state_dict", state), strict=False); t.eval()
    mx=np.full(2,np.inf,dtype=np.float32); out=[]
    for ct,wt in targets:
        dr=objectives_to_command(float(ct),float(wt),N_JOBS).astype(np.float32)
        r=ag._run_episode(env, dr, np.float32(N_JOBS), mx, eval_mode=True)
        out.append([float(r[5][0]), float(r[5][1])])
    return np.array(out)

amp=eval_policy(AMP, False)
urg=eval_policy(URG, True)
for nm,a in [("amplog",amp),("urgency",urg)]:
    print(f"{nm}: " + " ".join(f"({c:.2e},{w:.2e})" for c,w in a))

fig,ax=plt.subplots(figsize=(9.5,6.5))
ax.plot(arch[np.argsort(arch[:,0]),0], arch[np.argsort(arch[:,0]),1], "-", color="0.6", lw=1, label=f"discovered archive front ({len(arch)})")
ax.scatter(amp[:,0],amp[:,1], s=90, marker="s", color="#d62728", label="amplog (urgency OFF) achieved", zorder=5)
ax.scatter(urg[:,0],urg[:,1], s=90, marker="o", color="#2ca02c", label="urgency_a (urgency ON) achieved", zorder=6)
for i in range(len(targets)):
    ax.plot([amp[i,0],urg[i,0]],[amp[i,1],urg[i,1]], "-", color="0.8", lw=0.7, zorder=1)
ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title("before/after: urgency obs pulls the achieved front DOWN toward optimal\n(same command targets; lower=better. trace1024)")
fig.tight_layout(); fig.savefig("pf_1024_before_after.png", dpi=120, bbox_inches="tight")
print("saved pf_1024_before_after.png")
