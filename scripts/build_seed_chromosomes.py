#!/usr/bin/env python3
"""安い端注入用の軽い種chromosome npzを構築(NSGA不要・O(N)・任意スケール)。
全オンプレ(安い端)+全クラウド+p掃引。cost=Σ(cloud時 pt×nodes)で正確に並べ替え(env eval不要)。
usage: N=1024 OUT=results/eval_pf/seed_synth_1024.npz SYNTH_MAX_NODES=16 python scripts/build_seed_chromosomes.py"""
import os, numpy as np
from src.utils.job_gen.poason_job import JobSimulator
N=int(os.environ.get("N","1024"))
mx=int(os.environ.get("SYNTH_MAX_NODES","16"))
OUT=os.environ.get("OUT",f"results/eval_pf/seed_synth_{N}.npz")
np.random.seed(0)
jobs=JobSimulator(0,n_jobs=N,n_users=N,lam=0.2,max_required_nodes=mx).generate_jobs()
pt=jobs[:,1].astype(np.float64); nodes=jobs[:,2].astype(np.float64)  # [1]=pt,[2]=nodes
ps=[0.0,0.1,0.2,0.35,0.5,0.65,0.8,0.9,1.0]
chroms=[]; costs=[]
rng=np.random.default_rng(12345)
for p in ps:
    a=(rng.random(N)<p).astype(np.int8) if 0<p<1 else np.full(N,int(p),dtype=np.int8)
    cost=float((pt[a==1]*nodes[a==1]).sum())  # cloud cost = pt*nodes
    chroms.append(a); costs.append(cost)
chroms=np.array(chroms,dtype=np.int8)
pf=np.stack([np.array(costs), np.zeros(len(costs))],axis=1)  # wait=placeholder(順序はcostのみ使用)
os.makedirs(os.path.dirname(OUT),exist_ok=True)
np.savez(OUT, chromosomes=chroms, pf=pf)
print(f"saved {OUT}: {len(chroms)}種 cost[{min(costs):.0f},{max(costs):.0f}] 全オンプレ種あり(cost0)")
