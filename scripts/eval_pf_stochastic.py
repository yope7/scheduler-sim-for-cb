#!/usr/bin/env python3
"""(b1) 確率的eval + best-of-k。greedy(argmax) が隠している方策の到達可能集合を、
   各指令で K 回サンプリングして非支配を取ることで引き出す。
   greedy の distinct vs stochastic の distinct を比較し、方策に余力があるか診断。
   usage: CKPT=<pth> EXEC=<dir> CFG=<yml> NJ=128 NCMD=80 KSAMP=16 PCN_FILM=1 OUT=stoch.png python scripts/eval_pf_stochastic.py"""
import os, glob, re
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
if os.environ.get("OBS_URGENCY", "1") == "1":
    os.environ["SCHEDULER_OBS_URGENCY"] = "1"
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import (create_eval_env, load_config,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.utils.pf_command_eval import dedupe_pf, objectives_to_command

CKPT=os.environ["CKPT"]; CFG=os.environ["CFG"]; NJ=int(os.environ.get("NJ","128"))
NCMD=int(os.environ.get("NCMD","80")); K=int(os.environ.get("KSAMP","16"))
DEVICE=os.environ.get("DEVICE","cuda" if th.cuda.is_available() else "cpu")
OUT=os.environ.get("OUT","stoch.png"); LABEL=os.environ.get("LABEL",f"{NJ}-job")
EXEC=os.environ.get("EXEC", re.sub(r"/iteration_\d+/.*$","",CKPT))

config=load_config(CFG)
snap=load_learner_replay_snapshot(glob.glob(EXEC+"/learner_replay_snapshot.pkl.gz")[0])
arch=dedupe_pf(archive_pf_from_snapshot(snap,NJ)); arch=arch[np.argsort(arch[:,0])]
# dense interpolated commands across archive cost range
cg=np.linspace(float(arch[:,0].min()),float(arch[:,0].max()),NCMD)
wg=np.interp(cg,arch[:,0],arch[:,1])
env=create_eval_env(config,job_seed=0,n_jobs=NJ)
state=th.load(CKPT,map_location="cpu",weights_only=False)
ag=PCN(env,device=DEVICE,state_dim=env.observation_space.shape[0],
       scaling_factor=np.array([1.,1.,1./max(1,NJ)],dtype=np.float32),learning_rate=1e-3,
       batch_size=512,hidden_dim=512,project_name="t",experiment_name="PCN",log=False,
       use_enhanced_model=(state.get("model_type","")=="EnhancedPCNModel"))
tg=ag.network if ag.use_enhanced_model else ag.model
tg.load_state_dict(state.get("model_state_dict",state),strict=False); tg.eval()
mx=np.full(2,np.inf,dtype=np.float32)

greedy=[]; stoch=[]
for cc,ww in zip(cg,wg):
    dr=objectives_to_command(float(cc),float(ww),NJ).astype(np.float32)
    r=ag._run_episode(env,dr,np.float32(NJ),mx,eval_mode=True)   # greedy
    greedy.append([float(r[5][0]),float(r[5][1])])
    for _ in range(K):
        r=ag._run_episode(env,dr,np.float32(NJ),mx,eval_mode=False)  # sample
        stoch.append([float(r[5][0]),float(r[5][1])])
greedy=np.array(greedy); stoch=np.array(stoch)

def distinct(P,frac=0.005):
    nd=get_non_dominated_inds_minimize(P); pf=P[nd]
    g=P[:,0].max()*frac
    return pf, len(np.unique(np.round(pf/max(1e-9,g),0),axis=0))
gpf,gd=distinct(greedy); spf,sd=distinct(stoch)
print(f"{LABEL}: greedy distinct={gd} (from {len(greedy)} cmds)")
print(f"{LABEL}: stochastic distinct={sd} (from {len(stoch)} = {NCMD}cmd x {K} samples), non-dom={len(spf)}")
print(f"  → Δdistinct = {sd-gd:+d}  ({'方策に隠れた余力あり→(b2)価値大' if sd>=gd+8 else '余力少→分散/条件付けの真の限界'})")

fig,ax=plt.subplots(figsize=(11,6.5))
ax.scatter(stoch[:,0],stoch[:,1],s=6,c="#9ecae1",alpha=0.3,zorder=0,label=f"stochastic samples ({len(stoch)})")
ax.plot(arch[:,0],arch[:,1],"-",color="#888",lw=1.2,zorder=2,label=f"discovered PF ({len(arch)})")
spf2=spf[np.argsort(spf[:,0])]; ax.plot(spf2[:,0],spf2[:,1],"-",color="#1a73e8",lw=1.8,zorder=3,label=f"STOCHASTIC PF (distinct~{sd})")
ax.scatter(greedy[:,0],greedy[:,1],s=26,c="#d62728",edgecolor="k",lw=0.3,zorder=4,label=f"greedy achieved (distinct~{gd})")
ax.set_xlabel("Cost"); ax.set_ylabel("Avg Wait"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title(f"(b1) greedy vs stochastic-best-of-{K} — {LABEL}\ngreedy distinct={gd} -> stochastic distinct={sd}  ({'hidden capacity!' if sd>=gd+8 else 'no hidden capacity'})")
fig.tight_layout(); fig.savefig(OUT,dpi=120,bbox_inches="tight")
np.savez(OUT.replace(".png",".npz"),greedy=greedy,stoch=stoch,arch=arch)
print(f"saved {OUT}")
