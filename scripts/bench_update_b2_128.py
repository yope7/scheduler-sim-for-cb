#!/usr/bin/env python3
"""b2(fourier)128 の Learner.update ループ単体ベンチ + 結果不変A/B。

実運用と同じ env（--conditioning --mid-core + run_synthetic_urgency）で PCN を構築し、
本番の checkpoint/replay snapshot を読み込んで update_many を回す。

- DEVICE=cuda: ms/update を計測（高速化の前後比較）。
- DEVICE=cpu : 決定論。PCN_FAST_UPDATE 0/1 で重みハッシュが一致する事を確認（結果不変の証明）。

usage:
  PCN_FAST_UPDATE=0 DEVICE=cpu  N=8  PYTHONPATH=. .venv/bin/python scripts/bench_update_b2_128.py
  PCN_FAST_UPDATE=1 DEVICE=cuda N=60 PYTHONPATH=. .venv/bin/python scripts/bench_update_b2_128.py
"""
import os, sys, time, heapq, hashlib

# --- 本番(b2 128)と同じ env を import 前に固定（--conditioning --mid-core + run_synthetic_urgency）---
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
    "PCN_CONDITIONING_SENS_WAIT_DR_THRESH": "0.002",
    "PCN_ADAPTIVE_RETURN_NORMALIZATION": "1",
    # run_synthetic_urgency 学習重み
    "PCN_TRAIN_KNEE_PF_WEIGHT": "8", "PCN_TRAIN_LOW_SLOPE_PF_WEIGHT": "6",
    "PCN_TRAIN_LOW_WAIT_PF_WEIGHT": "10", "PCN_TRAIN_LOW_WAIT_MAX": "0",
    "PCN_TRAIN_LOW_WAIT_FRAC": "0.30", "PCN_PHASE1_SWEEP_TRAIN_WEIGHT": "10",
    "PCN_PF_COMMAND_ANCHORS": "16", "PCN_CHOOSE_COMMANDS_MODE": "pf_archive",
}
for k, v in _ENV.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("PCN_FAST_UPDATE", "1")

import numpy as np
import torch as th

from scripts.pcn_replay_snapshot import (
    create_eval_env, load_config, load_learner_replay_snapshot,
)
from src.agents.pcn_agent import PCN

NJ = int(os.environ.get("NJ", "128"))
N = int(os.environ.get("N", "50"))
WARMUP = int(os.environ.get("WARMUP", "5"))
SEED = int(os.environ.get("SEED", "0"))
DEVICE = os.environ.get("DEVICE", "cuda" if th.cuda.is_available() else "cpu")
EXEC = os.environ.get(
    "EXEC", "experiments/distributed_pcn/run_synth128_fourier128/20260606_020951")
CKPT = os.environ.get("CKPT", f"{EXEC}/iteration_100/model_iter_100.pth")
SNAP = os.environ.get("SNAP", f"{EXEC}/learner_replay_snapshot.pkl.gz")
CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_synthetic_pcn.yml")

config = load_config(CFG)
env = create_eval_env(config, job_seed=0, n_jobs=NJ)
state = th.load(CKPT, map_location="cpu", weights_only=False)

ag = PCN(env, device=DEVICE, state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32),
         learning_rate=1e-3, batch_size=2048, hidden_dim=512,
         project_name="t", experiment_name="bench", log=False, use_enhanced_model=False)
tg = ag.model
tg.load_state_dict(state["model_state_dict"], strict=False)

# replay snapshot -> experience_replay（heap: (priority,(step,id),episode)）
snap = load_learner_replay_snapshot(SNAP)
episodes = snap.get("episodes", [])
ag.experience_replay = []
for i, ep in enumerate(episodes):
    if ep:
        heapq.heappush(ag.experience_replay, (1, (i, id(ep)), ep))
# 正規化を checkpoint buffer から復元（command balance 用）
ag.return_norm_center = tg.desired_return_center.detach().cpu().numpy().copy()
ag.return_norm_scale = tg.desired_return_scale.detach().cpu().numpy().copy()
steps = ag.build_training_batch_cache(on_device=(DEVICE != "cpu"))
print(f"[bench] device={DEVICE} replay_ep={len(ag.experience_replay)} cache_steps={steps} "
      f"obs_dim={env.observation_space.shape[0]} fast={os.environ['PCN_FAST_UPDATE']}", flush=True)

# seed（numpy=サンプリング/コマンド, torch=dropout/randperm）
ag.np_random = np.random.default_rng(SEED)
th.manual_seed(SEED)
if DEVICE != "cpu":
    th.cuda.manual_seed_all(SEED)

ag.update_many(WARMUP)
if DEVICE != "cpu":
    th.cuda.synchronize()

if os.environ.get("TORCH_PROFILE") == "1":
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE != "cpu" else [])
    with profile(activities=acts, record_shapes=False) as prof:
        ag.update_many(int(os.environ.get("PROF_N", "20")))
        if DEVICE != "cpu":
            th.cuda.synchronize()
    key = "self_cuda_time_total" if DEVICE != "cpu" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=25))
    raise SystemExit(0)

t0 = time.perf_counter()
mean_loss, last_metrics, losses = ag.update_many(N)
if DEVICE != "cpu":
    th.cuda.synchronize()
dt = time.perf_counter() - t0

h = hashlib.sha1()
for p in tg.parameters():
    h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
print(f"RESULT device={DEVICE} fast={os.environ['PCN_FAST_UPDATE']} "
      f"per_update_ms={dt / N * 1000:.3f} mean_loss={mean_loss:.6f} "
      f"loss_head={[round(float(x), 6) for x in losses[:4]]} "
      f"metrics_keys={sorted(last_metrics.keys())} "
      f"whash={h.hexdigest()[:16]}", flush=True)
