#!/usr/bin/env bash
# 統合ゲート: 全最適化ON（update + env + sweep）で b2/128 を実学習し「崩壊しない・PF同等」を実証。
# 直列実行（他の学習を起動しない）。GPUは Ray 任せ（両GPU空き前提）。
set -u
cd /home/noguchi/scheduler-sim-for-cb
NITER="${1:-100}"
CFG=experiments/distributed_pcn/job_synthetic_pcn.yml
BASE_CKPT=experiments/distributed_pcn/run_synth128_fourier128/20260606_020951/iteration_100/model_iter_100.pth
OUT=experiments/distributed_pcn/run_synth128_fourier128sweep

echo "[gate] TRAIN fourier128 (FAST_UPDATE+FAST_ENV+SWEEP all ON) NITER=$NITER START=$(date +%H:%M:%S)"
PCN_FAST_ENV_SWEEP=1 PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 \
  bash scripts/run_synthetic_urgency.sh fourier128sweep 128 "$NITER"
EXG=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*' | tail -1)
CKPT="$EXG/iteration_${NITER}/model_iter_${NITER}.pth"
echo "[gate] EXG=$EXG"
echo "[gate] === train.log: NaN/error/collapse markers (want 0) + phase timing ==="
grep -aciE "Traceback|エラー|nan に|非有限|Invalid device" "$OUT/train.log" 2>/dev/null
grep -aE "総経過時間|フェーズ[123]完了|学習エポック数" "$OUT/train.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -6
echo "[gate] === in-training command-follow (collapse なら match≈0/片隅) ==="
grep -aE "follow\(VALUE" "$OUT/train.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -3

echo "[gate] === eval seed0: sweep-trained vs baseline (gap to common true-PF) ==="
PCN_FILM=1 PCN_FOURIER_CMD=1 DEVICE=cuda CKPT="$CKPT" CFG=$CFG NJ=128 SEEDS=0 NCMD=40 KSAMP=10 \
  OUT=truepf_fourier128sweep_s0.npz OBS_URGENCY=1 PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py 2>&1 | grep -aE "seed=|saved"
PCN_FILM=1 PCN_FOURIER_CMD=1 DEVICE=cuda CKPT="$BASE_CKPT" CFG=$CFG NJ=128 SEEDS=0 NCMD=40 KSAMP=10 \
  OUT=truepf_fourier128base_s0.npz OBS_URGENCY=1 PYTHONPATH=. .venv/bin/python scripts/eval_b2_compare.py 2>&1 | grep -aE "seed=|saved"

PYTHONPATH=. .venv/bin/python - <<'PY'
import numpy as np
from src.agents.pcn_agent import get_non_dominated_inds_minimize
def gap(npz):
    d=np.load(npz); g=d["greedy_0"]; rp=d["rp_0"]; sm=d["samp_0"]
    ref=np.vstack([rp,sm,g]); nd=get_non_dominated_inds_minimize(ref); pf=ref[nd]
    o=np.argsort(pf[:,0]); pf=pf[o]
    wq=np.interp(g[:,0], pf[:,0], pf[:,1])
    rng=float(pf[:,1].max()-pf[:,1].min()) or 1.0
    return float(np.mean(np.clip(g[:,1]-wq,0,None))/rng), g[:,0].min(), g[:,0].max(), len(g)
gs=gap("truepf_fourier128sweep_s0.npz"); gb=gap("truepf_fourier128base_s0.npz")
print(f"[gate] SWEEP gap={gs[0]:.4f} cost[{gs[1]:.0f},{gs[2]:.0f}] n={gs[3]}")
print(f"[gate] BASE  gap={gb[0]:.4f} cost[{gb[1]:.0f},{gb[2]:.0f}] n={gb[3]}")
ok = gs[0] <= gb[0] + 0.01 and gs[2] > gs[1]*1.5  # 同等以下のgap かつ cost域が潰れていない
print(f"[gate] VERDICT: {'PASS (no collapse, PF parity)' if ok else 'CHECK — review'}")
PY
echo "[gate] DONE=$(date +%H:%M:%S)"