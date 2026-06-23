#!/usr/bin/env python3
"""front-loading の機序検証: 方策のクラウド判断は『ジョブの緊急度(オンプレ予測待ち)』に
   依るのか、それとも単に『位置(序盤=予算潤沢)』に依るのか。
   緊急度に依る → 観測に信号がある。位置に依る → 信号が無く obs拡張が効く。"""
import os, time
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS","1"); os.environ.setdefault("SCHEDULER_LEARNER_BITMAP","0")
import numpy as np, torch as th
from scripts.pcn_replay_snapshot import (create_eval_env, load_config, eval_n_jobs,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import dedupe_pf, objectives_to_command

CFG=os.environ.get("DIAG_CFG", "experiments/distributed_pcn/job_trace_1024_pcn.yml")
CKPT=os.environ.get("DIAG_CKPT", "experiments/distributed_pcn/run1024_amplog_b/20260604_122948/iteration_100/model_iter_100.pth")
SNAP=os.environ.get("DIAG_SNAP", "experiments/distributed_pcn/run1024_amplog_b/20260604_122948/learner_replay_snapshot.pkl.gz")
import os.path as _osp
config=load_config(CFG)
snap=load_learner_replay_snapshot(SNAP) if _osp.isfile(SNAP) else None
n_jobs=int(os.environ.get("DIAG_NJOBS", str(snap.get("metadata",{}).get("n_jobs",24) if snap else 24)))
env=create_eval_env(config, job_seed=0, n_jobs=n_jobs)
state=th.load(CKPT, map_location="cpu", weights_only=False)
agent=PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
          scaling_factor=np.array([1.0,1.0,1.0/max(1,n_jobs)],dtype=np.float32), learning_rate=1e-3,
          batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
          use_enhanced_model=(state.get("model_type","")=="EnhancedPCNModel"))
tgt=agent.network if agent.use_enhanced_model else agent.model
tgt.load_state_dict(state.get("model_state_dict", state), strict=False); tgt.eval()

if snap is not None:
    arch=dedupe_pf(archive_pf_from_snapshot(snap, n_jobs)); order=np.argsort(arch[:,1])
    ct,wt=arch[order[len(order)//2]]  # mid target
else:
    ct,wt=float(os.environ.get("DIAG_COST","6.62e8")), float(os.environ.get("DIAG_WAIT","5.61e5"))
dr=objectives_to_command(float(ct),float(wt),n_jobs).astype(np.float32); hz=np.float32(n_jobs)

obs=env.reset(); done=False; desired=dr.copy(); horizon=hz
rows=[]  # (step, predicted_onprem_wait, action)
step=0
while not done:
    j=int(env.index_next_job)
    if j>=len(env.jobs): break
    raw=env.jobs[j]; job=env._to_queue_job(raw); arr=int(raw[0])
    _,onp=env._find_event_allocation(job,False,arr)
    pw=int(onp)-arr
    pobs=agent._obs_for_policy(env, obs)
    action=int(agent._act(pobs, desired, horizon, eval_mode=True))
    obs,reward,scheduled,wts,done=env.step(action)
    desired=desired-np.array(reward,dtype=np.float32)
    if scheduled: horizon=np.float32(max(float(horizon)-1,1.0))
    rows.append((step,pw,action)); step+=1

cost_a,_,wait_a=env.calc_objective_values()
print(f"ACHIEVED: cost={cost_a:.3e} wait={wait_a:.3e}  (target cost={ct:.2e} wait={wt:.2e})  [amplog baseline @mid: wait=8.63e5]")
a=np.array(rows,dtype=np.float64)
pos=a[:,0]/max(1,a[:,0].max()); pw=a[:,1]; act=a[:,2]
def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else float('nan')
# urgency response: among jobs, do clouded ones have higher predicted on-prem wait?
hi=pw>np.median(pw); lo=~hi
print(f"target mid: cost={ct:.2e} wait={wt:.2e}  n_steps={len(a)}  total_cloud={act.mean():.2f}")
print(f"corr(action, predicted_onprem_wait) = {corr(act,pw):+.3f}   <- urgency response")
print(f"corr(action, position)              = {corr(act,pos):+.3f}   <- front-loading/budget")
print(f"cloud rate | HIGH urgency (pw>median) = {act[hi].mean():.2f}")
print(f"cloud rate | LOW  urgency (pw<=median)= {act[lo].mean():.2f}")
print(f"predicted_wait mean | clouded={pw[act==1].mean():.1f}  onprem={pw[act==0].mean() if (act==0).any() else float('nan'):.1f}")
