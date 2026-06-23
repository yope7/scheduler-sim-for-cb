#!/usr/bin/env python3
"""凍結run(勾配nan→全スキップ)の単一プロセス再現 + anomaly detection で nan発生演算を特定する。

bench_update_b2_128.py を qd レシピ(FILM+Fourier+PF_W12+LS0.02)向けに改造。
checkpoint は読まず random init(本番の凍結状態を再現するのが目的)。

usage: SNAP=experiments/.../learner_replay_snapshot.pkl.gz CFG=experiments/distributed_pcn/job_trace_128_pcn.yml \
       NJ=128 SEEDS=0,1,2 PYTHONPATH=. .venv/bin/python scripts/debug_nan_grad.py
"""
import os, sys, heapq

for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v

# --- 本番(run_synthetic_urgency + qd LV)と同じ env を import 前に固定 ---
_ENV = {
    "DISTRIBUTED_PCN_USE_EVENT_OBS": "1", "DISTRIBUTED_PCN_USE_EVENT_NATIVE": "1",
    "SCHEDULER_LEARNER_BITMAP": "0", "SCHEDULER_OBS_URGENCY": "1",
    "PCN_FOURIER_CMD": "1", "PCN_FOURIER_BANDS": "4", "PCN_FILM": "1",
    "PCN_USE_AMP": "0", "PCN_OBS_LOG": "1",
    # --conditioning
    "PCN_CONDITIONING_SENS_WEIGHT": "0.03", "PCN_CONDITIONING_KL_MARGIN": "0.08",
    "PCN_COND_ADD_SCALE": "0.25", "PCN_S_EMB_DROPOUT": "0.08",
    "PCN_TRAIN_COST_ENDPOINT_WEIGHT": "8", "PCN_VALUE_REPRO_WEIGHT": "0",
    # --mid-core
    "PCN_COMMAND_BALANCE": "1", "PCN_TRAIN_MID_STEP_WEIGHT": "6",
    "PCN_TRAIN_EVALIKE_STEP_WEIGHT": "4", "PCN_TRAIN_EVALIKE_STEP_FRAC": "0.15",
    "PCN_TRAIN_MID_PF_WEIGHT": "4", "PCN_MID_BAND_COND_WEIGHT": "0.06",
    "PCN_MID_BAND_COND_WAIT_LEVELS": "5", "PCN_MID_BAND_COND_COST_LEVELS": "4",
    # run_synthetic_urgency.sh 固有
    "PCN_TRAIN_KNEE_PF_WEIGHT": "8", "PCN_TRAIN_LOW_SLOPE_PF_WEIGHT": "6",
    "PCN_TRAIN_LOW_WAIT_PF_WEIGHT": "10", "PCN_TRAIN_LOW_WAIT_MAX": "0",
    "PCN_TRAIN_LOW_WAIT_FRAC": "0.30", "PCN_TRAIN_PF_BALANCE_REF": "32",
    "PCN_TRAIN_PF_BALANCE_ALPHA": "0.5", "PCN_PHASE1_SWEEP_TRAIN_WEIGHT": "10",
    "PCN_PF_COMMAND_ANCHORS": "16", "PCN_EMA_DECAY": "0",
    # qd LV
    "PCN_TRAIN_PF_WEIGHT": "12", "PCN_LABEL_SMOOTH": "0.02",
}
for k, v in _ENV.items():
    os.environ.setdefault(k, v)

import numpy as np
import torch as th

from scripts.pcn_replay_snapshot import create_eval_env, load_config, load_learner_replay_snapshot
from src.agents.pcn_agent import PCN

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_128_pcn.yml")
NJ = int(os.environ.get("NJ", "128"))
SNAP = os.environ["SNAP"]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
N_UPD = int(os.environ.get("N_UPD", "30"))
NO_NORM = os.environ.get("NO_NORM", "0") == "1"   # 1: 正規化を未初期化のままにする(対照実験)
ANOMALY = os.environ.get("ANOMALY", "1") == "1"

config = load_config(CFG)
env = create_eval_env(config, job_seed=0, n_jobs=NJ)
snap = load_learner_replay_snapshot(SNAP)
episodes = snap.get("episodes", [])
print(f"[probe] snapshot episodes={len(episodes)} cfg={CFG} NJ={NJ} no_norm={NO_NORM}")

for seed in SEEDS:
    th.manual_seed(seed)
    np.random.seed(seed)
    ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
             scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32),
             learning_rate=1e-3, batch_size=2048, hidden_dim=512,
             project_name="t", experiment_name="probe", log=False, use_enhanced_model=False)
    ag.experience_replay = []
    for i, ep in enumerate(episodes):
        if ep:
            heapq.heappush(ag.experience_replay, (1, (i, id(ep)), ep))
    if not NO_NORM:
        ag.update_desired_return_normalization()
    print(f"[probe seed={seed}] norm center={ag.return_norm_center} scale={ag.return_norm_scale}")
    ag.build_training_batch_cache(on_device=False)
    ag.np_random = np.random.default_rng(seed)

    sk0, st0 = ag._nan_skip_total, ag._opt_step_total
    if ANOMALY:
        th.autograd.set_detect_anomaly(True)
    try:
        losses = []
        for u in range(N_UPD):
            out = ag.update()
            losses.append(float(out[0]) if isinstance(out, (tuple, list)) else float(out))
        sk, st = ag._nan_skip_total - sk0, ag._opt_step_total - st0
        rate = sk / max(1, sk + st)
        print(f"[probe seed={seed}] RESULT skip率={rate:.1%} (skip={sk}/step={st}) "
              f"loss_head={[round(l,3) for l in losses[:5]]}")
    except RuntimeError as e:
        print(f"[probe seed={seed}] ★ANOMALY捕捉: {e}")
        import traceback
        tb = traceback.format_exc()
        tail = [l for l in tb.splitlines() if "src/" in l or "Error" in l or "returned nan" in l]
        print("\n".join(tail[-15:]))
    finally:
        th.autograd.set_detect_anomaly(False)
