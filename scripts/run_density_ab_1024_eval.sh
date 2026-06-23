#!/usr/bin/env bash
# 1024 A/B 評価+図: baseline(手動帯) vs density(密度版一本) を未知seedで掃引し PF図を出す.
# OBS_URGENCY=1 = 1024学習レシピと一致(obs次元/モデル構造を合わせる). 1024 evalは重いので軽量化.
set -uo pipefail
ROOT=/home/noguchi/scheduler-sim-for-cb
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CFG=experiments/distributed_pcn/job_trace_1024_pcn.yml
NJ=1024
B_CKPT=$(ls "$ROOT"/experiments/distributed_pcn/density_ab_1024/baseline/20*/iteration_100/model_iter_100.pth 2>/dev/null | tail -1)
D_CKPT=$(ls "$ROOT"/experiments/distributed_pcn/density_ab_1024/density/20*/iteration_100/model_iter_100.pth 2>/dev/null | tail -1)
echo "[1024eval] B_CKPT=$B_CKPT"; echo "[1024eval] D_CKPT=$D_CKPT"
[ -z "$B_CKPT" ] && { echo "ERROR baseline ckpt missing"; exit 1; }
[ -z "$D_CKPT" ] && { echo "ERROR density ckpt missing"; exit 1; }

echo "[1024eval] sweep baseline $(date +%H:%M:%S)"
CKPT="$B_CKPT" CFG=$CFG NJ=$NJ SEEDS=1 NCMD=30 KSAMP=6 NPROC=32 OBS_URGENCY=1 \
  OUT="$ROOT/truepf_1024_baseline.npz" PYTHONPATH=. "$PY" scripts/eval_b2_compare.py 2>&1 | grep -aE "seed=|saved"
echo "[1024eval] sweep density $(date +%H:%M:%S)"
CKPT="$D_CKPT" CFG=$CFG NJ=$NJ SEEDS=1 NCMD=30 KSAMP=6 NPROC=32 OBS_URGENCY=1 \
  OUT="$ROOT/truepf_1024_density.npz" PYTHONPATH=. "$PY" scripts/eval_b2_compare.py 2>&1 | grep -aE "seed=|saved"

echo "[1024eval] plot + gap $(date +%H:%M:%S)"
PYTHONPATH=. "$PY" - <<'PY'
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.agents.pcn_agent import get_non_dominated_inds_minimize
B=np.load("truepf_1024_baseline.npz"); D=np.load("truepf_1024_density.npz")
def gap(g,pf):
    wq=np.interp(g[:,0],pf[:,0],pf[:,1]); rng=float(pf[:,1].max()-pf[:,1].min()) or 1.0
    return float(np.mean(np.clip(g[:,1]-wq,0,None))/rng)
sd=1
rp=B[f"rp_{sd}"]; bg=B[f"greedy_{sd}"]; dg=D[f"greedy_{sd}"]
ref=np.vstack([rp,B[f"samp_{sd}"],bg,D[f"samp_{sd}"],dg])
nd=get_non_dominated_inds_minimize(ref); pf=ref[nd]; pf=pf[np.argsort(pf[:,0])]
gb=gap(bg,pf); gd=gap(dg,pf)
fig,ax=plt.subplots(figsize=(8.5,6.2))
ax.scatter(rp[:,0],rp[:,1],s=10,c="#cccccc",alpha=0.5,label="random-p sweep")
ax.plot(pf[:,0],pf[:,1],"-",color="#2ca02c",lw=2,label=f"common TRUE PF (n={len(pf)})")
ax.scatter(bg[:,0],bg[:,1],s=36,c="#d62728",edgecolor="k",lw=0.3,label=f"baseline manual-bands  gap={gb:.3f}")
ax.scatter(dg[:,0],dg[:,1],s=36,c="#1a73e8",edgecolor="k",lw=0.3,marker="D",label=f"density-inverse (1 knob)  gap={gd:.3f}")
ax.axvline(0,color="gray",ls=":",lw=1)
ax.set_title(f"1024 jobs UNSEEN seed={sd}: density-inverse vs manual-bands replay weighting\n(cost=0 dotted = cheapest corner)")
ax.set_xlabel("Cost"); ax.set_ylabel("Avg Wait"); ax.grid(alpha=0.3); ax.legend(fontsize=9,loc="upper right")
fig.tight_layout(); fig.savefig("pf_1024_density_compare.png",dpi=120,bbox_inches="tight")
print(f"saved pf_1024_density_compare.png")
print(f"RESULT baseline_gap={gb:.4f}  density_gap={gd:.4f}  delta={gd-gb:+.4f}")
print(f"cost-span baseline[{bg[:,0].min():.0f},{bg[:,0].max():.0f}] n={len(bg)} | density[{dg[:,0].min():.0f},{dg[:,0].max():.0f}] n={len(dg)}")
PY
echo "[1024eval] DONE $(date +%H:%M:%S)"
