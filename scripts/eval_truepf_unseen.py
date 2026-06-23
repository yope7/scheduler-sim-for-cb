#!/usr/bin/env python3
"""未知ジョブ(seed)の「真PF」を軽く推定して、方策の達成を重ねる正しい汎化テスト。
   真PF推定 = ランダムp掃引(独立) ＋ 方策best-of-k の非支配（その instance で到達可能な最良フロント）。
   それに方策の greedy 達成（=実運用の1発）を重ね、未知 instance の PF を再現できているか見る。
   usage: CKPT=.. EXEC=.. CFG=.. NJ=128 SEEDS=1,2 NCMD=40 KSAMP=10 PCN_FILM=1 python scripts/eval_truepf_unseen.py"""
import os, glob, re
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS","1"); os.environ.setdefault("SCHEDULER_LEARNER_BITMAP","0")
if os.environ.get("OBS_URGENCY","1")=="1": os.environ["SCHEDULER_OBS_URGENCY"]="1"
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.utils.pf_command_eval import objectives_to_command

CKPT=os.environ["CKPT"]; CFG=os.environ["CFG"]; NJ=int(os.environ.get("NJ","128"))
SEEDS=[int(x) for x in os.environ.get("SEEDS","1,2").split(",")]
NCMD=int(os.environ.get("NCMD","40")); K=int(os.environ.get("KSAMP","10"))
DEVICE=os.environ.get("DEVICE","cuda" if th.cuda.is_available() else "cpu")
config=load_config(CFG)
state=th.load(CKPT,map_location="cpu",weights_only=False)

def randp_sweep(env, nP=40):
    pts=[]
    for i,p in enumerate(np.linspace(0,1,nP)):
        rng=np.random.default_rng(1000+i); env.reset(); tw=tc=0.0; n=0; done=False; st=0
        while not done and st<NJ+5:
            a=1 if rng.random()<p else 0
            r=env.step(a); tw+=-float(r[1][0]); tc+=-float(r[1][1]); n+=1 if r[2] else 0; done=r[-1]; st+=1
        pts.append([tc, tw/max(1,n)])
    return np.array(pts)

fig,axes=plt.subplots(1,len(SEEDS),figsize=(7*len(SEEDS),5.8)); axes=np.atleast_1d(axes)
for ax,sd in zip(axes,SEEDS):
    env=create_eval_env(config,job_seed=sd,n_jobs=NJ)
    ag=PCN(env,device=DEVICE,state_dim=env.observation_space.shape[0],
           scaling_factor=np.array([1.,1.,1./max(1,NJ)],dtype=np.float32),learning_rate=1e-3,
           batch_size=512,hidden_dim=512,project_name="t",experiment_name="PCN",log=False,
           use_enhanced_model=(state.get("model_type","")=="EnhancedPCNModel"))
    tg=ag.network if ag.use_enhanced_model else ag.model
    tg.load_state_dict(state.get("model_state_dict",state),strict=False); tg.eval()
    mx=np.full(2,np.inf,dtype=np.float32)
    rp=randp_sweep(env)                                   # 独立な探索
    cmin,cmax=float(rp[:,0].min()),float(rp[:,0].max())
    cg=np.linspace(cmin,cmax,NCMD); wg=np.interp(cg,np.sort(rp[:,0]),rp[np.argsort(rp[:,0]),1])
    greedy=[]; allp=[rp]
    for cc,ww in zip(cg,wg):
        dr=objectives_to_command(float(cc),float(ww),NJ).astype(np.float32)
        r=ag._run_episode(env,dr,np.float32(NJ),mx,eval_mode=True); greedy.append([float(r[5][0]),float(r[5][1])])
        samp=[]
        for _ in range(K):
            r=ag._run_episode(env,dr,np.float32(NJ),mx,eval_mode=False); samp.append([float(r[5][0]),float(r[5][1])])
        allp.append(np.array(samp))
    greedy=np.array(greedy); allp=np.vstack(allp)
    tnd=get_non_dominated_inds_minimize(allp); truepf=allp[tnd]; truepf=truepf[np.argsort(truepf[:,0])]  # 真PF推定
    gnd=get_non_dominated_inds_minimize(greedy); gpf=greedy[gnd]; gpf=gpf[np.argsort(gpf[:,0])]
    # greedy が真PFからどれだけ上(劣)か
    ex=np.clip(greedy[:,1]-np.interp(greedy[:,0],truepf[:,0],truepf[:,1]),0,None)/max(1e-9,truepf[:,1].ptp())
    ax.scatter(rp[:,0],rp[:,1],s=14,c="#cfe8cf",alpha=0.6,zorder=0,label=f"random-p sweep ({len(rp)})")
    ax.plot(truepf[:,0],truepf[:,1],"-",color="#2ca02c",lw=2.0,zorder=2,label=f"estimated TRUE PF ({len(truepf)})")
    ax.scatter(greedy[:,0],greedy[:,1],s=26,c="#d62728",edgecolor="k",lw=0.3,zorder=4,label="policy greedy (deploy)")
    ax.plot(gpf[:,0],gpf[:,1],"-",color="#d62728",lw=1.2,alpha=0.6,zorder=3)
    ax.set_title(f"UNSEEN seed={sd}: gap-to-true={ex.mean():.3f}\nred(policy) on green(true PF)? = generalizes well",fontsize=11)
    ax.set_xlabel("Cost"); ax.set_ylabel("Avg Wait"); ax.grid(alpha=0.3); ax.legend(fontsize=8,loc="upper right")
    print(f"seed={sd}: true PF pts={len(truepf)}, greedy gap-to-true(mean wait excess, norm)={ex.mean():.3f}")
fig.suptitle("Proper generalization: policy (red) vs each UNSEEN instance's OWN true PF (green)",fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("pf_truepf_unseen.png",dpi=120,bbox_inches="tight")
print("saved pf_truepf_unseen.png")
