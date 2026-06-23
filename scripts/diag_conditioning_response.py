#!/usr/bin/env python3
"""conditioning応答の特性化: archive PF を target command として与え、
   target→achieved の対応を記録・可視化。collapse がフロントのどこで・どう起きるか。"""
import os, time, json
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from scripts.pcn_replay_snapshot import (create_eval_env, load_config, eval_n_jobs,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.utils.pf_command_eval import dedupe_pf, stratified_sample_pf, objectives_to_command

CFG = "experiments/distributed_pcn/job_trace_1024_pcn.yml"
CKPT = "experiments/distributed_pcn/run1024_amplog_b/20260604_122948/iteration_100/model_iter_100.pth"
SNAP = "experiments/distributed_pcn/run1024_amplog_b/20260604_122948/learner_replay_snapshot.pkl.gz"
N_TARGETS = 40

t0=time.perf_counter()
config = load_config(CFG)
snap = load_learner_replay_snapshot(SNAP)
n_jobs = int(snap.get("metadata", {}).get("n_jobs", eval_n_jobs(config)))
env = create_eval_env(config, job_seed=0, n_jobs=n_jobs)
state = th.load(CKPT, map_location="cpu", weights_only=False)
agent = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
            scaling_factor=np.array([1.0,1.0,1.0/max(1,n_jobs)],dtype=np.float32), learning_rate=1e-3,
            batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
            use_enhanced_model=(state.get("model_type","")=="EnhancedPCNModel"))
tgt = agent.network if agent.use_enhanced_model else agent.model
tgt.load_state_dict(state.get("model_state_dict", state), strict=False); tgt.eval()

arch = dedupe_pf(archive_pf_from_snapshot(snap, n_jobs))
targets = stratified_sample_pf(arch, N_TARGETS, rng=np.random.default_rng(0), include_extremes=True)
print(f"[setup] {time.perf_counter()-t0:.1f}s  archive_nd={len(arch)} targets={len(targets)} n_jobs={n_jobs}")

max_return = np.full(2, np.inf, dtype=np.float32)
rows=[]
for i,(ct,wt) in enumerate(targets):
    dr = objectives_to_command(float(ct), float(wt), n_jobs)  # [-wait*nj, -cost]
    out = agent._run_episode(env, dr.astype(np.float32), np.float32(n_jobs), max_return, eval_mode=True)
    ca, wa = float(out[5][0]), float(out[5][1])
    rows.append((float(ct),float(wt),ca,wa))
arr=np.array(rows)  # cols: cost_t, wait_t, cost_a, wait_a
np.savez("pf_1024_conditioning_response.npz", target=arr[:,:2], achieved=arr[:,2:])

# metrics: wait-axis tracking (do achieved waits follow commanded waits?)
wt, wa = arr[:,1], arr[:,3]
ct, ca = arr[:,0], arr[:,2]
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>0 and b.std()>0 else float('nan')
print(f"[track] cost corr(target,achieved)={corr(ct,ca):.3f}  wait corr(target,achieved)={corr(wt,wa):.3f}")
lowwait = arr[arr[:,1] <= np.quantile(arr[:,1],0.33)]  # low-wait targets
print(f"[low-wait targets] commanded wait mean={lowwait[:,1].mean():.3e}  ACHIEVED wait mean={lowwait[:,3].mean():.3e}  (ratio {lowwait[:,3].mean()/max(1,lowwait[:,1].mean()):.2f}x)")

fig, axes = plt.subplots(1,2, figsize=(15,6))
ax=axes[0]
for ctv,wtv,cav,wav in rows:
    ax.annotate("", xy=(cav,wav), xytext=(ctv,wtv),
                arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8, alpha=0.8))
ax.scatter(arr[:,0],arr[:,1], s=45, c="#2ca02c", label="commanded target (on archive PF)", zorder=4)
ax.scatter(arr[:,2],arr[:,3], s=30, c="#d62728", label="achieved", zorder=5)
ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title("target -> achieved (arrow). Long arrows = conditioning miss")
ax=axes[1]
ax.scatter(wt, wa, s=40, c=ct, cmap="viridis")
lo,hi=min(wt.min(),wa.min()), max(wt.max(),wa.max())
ax.plot([lo,hi],[lo,hi],"k--",lw=1,label="perfect tracking")
ax.set_xlabel("commanded wait target"); ax.set_ylabel("achieved wait")
ax.set_title(f"wait-axis tracking (corr={corr(wt,wa):.2f}); color=cost target"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.suptitle("amplog_b iter100 conditioning response", y=1.01)
fig.tight_layout(); fig.savefig("pf_1024_conditioning_response.png", dpi=115, bbox_inches="tight")
print(f"[done] {time.perf_counter()-t0:.1f}s saved pf_1024_conditioning_response.png")
