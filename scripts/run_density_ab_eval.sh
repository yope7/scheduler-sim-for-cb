#!/usr/bin/env bash
# A/B 評価: baseline(現行手動帯) vs density(密度版一本) を未知seedで掃引し gap-to-true を比較。
# OBS_URGENCY=0 / FILM・FOURIER無し = scale128 学習レシピと一致(obs次元・モデル構造を合わせる)。
set -uo pipefail
ROOT=/home/noguchi/scheduler-sim-for-cb
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CFG=config/config.yml
BASE_CKPT=$(ls "$ROOT"/experiments/distributed_pcn/density_ab/baseline/20*/iteration_050/model_iter_050.pth 2>/dev/null | tail -1)
DENS_CKPT=$(ls "$ROOT"/experiments/distributed_pcn/density_ab/density/20*/iteration_050/model_iter_050.pth 2>/dev/null | tail -1)
echo "[eval] BASE_CKPT=$BASE_CKPT"
echo "[eval] DENS_CKPT=$DENS_CKPT"
[ -z "$BASE_CKPT" ] && { echo "ERROR: baseline ckpt missing"; exit 1; }
[ -z "$DENS_CKPT" ] && { echo "ERROR: density ckpt missing"; exit 1; }

echo "[eval] sweep baseline $(date +%H:%M:%S)"
CKPT="$BASE_CKPT" CFG=$CFG NJ=128 SEEDS=1,2 NCMD=40 KSAMP=10 NPROC=32 OBS_URGENCY=0 \
  OUT="$ROOT/truepf_density_baseline.npz" PYTHONPATH=. "$PY" scripts/eval_b2_compare.py 2>&1 | grep -aE "seed=|saved"
echo "[eval] sweep density $(date +%H:%M:%S)"
CKPT="$DENS_CKPT" CFG=$CFG NJ=128 SEEDS=1,2 NCMD=40 KSAMP=10 NPROC=32 OBS_URGENCY=0 \
  OUT="$ROOT/truepf_density_on.npz" PYTHONPATH=. "$PY" scripts/eval_b2_compare.py 2>&1 | grep -aE "seed=|saved"

echo "[eval] === gap-to-true (共通真PF=両方策の rp/samp/greedy 非支配) ==="
PYTHONPATH=. "$PY" - <<'PY'
import numpy as np
from src.agents.pcn_agent import get_non_dominated_inds_minimize
B=np.load("truepf_density_baseline.npz"); D=np.load("truepf_density_on.npz")
def gap(g, pf):
    wq=np.interp(g[:,0], pf[:,0], pf[:,1]); rng=float(pf[:,1].max()-pf[:,1].min()) or 1.0
    return float(np.mean(np.clip(g[:,1]-wq,0,None))/rng)
tot_b=tot_d=0.0
for sd in (1,2):
    bg=B[f"greedy_{sd}"]; dg=D[f"greedy_{sd}"]
    ref=np.vstack([B[f"rp_{sd}"],B[f"samp_{sd}"],bg,D[f"samp_{sd}"],dg])
    nd=get_non_dominated_inds_minimize(ref); pf=ref[nd]; pf=pf[np.argsort(pf[:,0])]
    gb=gap(bg,pf); gd=gap(dg,pf); tot_b+=gb; tot_d+=gd
    win="density WIN" if gd<gb else "baseline win"
    print(f"seed{sd}: baseline_gap={gb:.4f}  density_gap={gd:.4f}  delta={gd-gb:+.4f}  -> {win}")
    print(f"        cost-span  baseline[{bg[:,0].min():.0f},{bg[:,0].max():.0f}] n={len(bg)} | density[{dg[:,0].min():.0f},{dg[:,0].max():.0f}] n={len(dg)}")
print(f"MEAN: baseline_gap={tot_b/2:.4f}  density_gap={tot_d/2:.4f}  delta={(tot_d-tot_b)/2:+.4f}")
PY
echo "[eval] DONE $(date +%H:%M:%S)"
