#!/usr/bin/env python3
"""event env の配置ロジックだけを高速検証＋計測（NN/GPU不要）。

変更したのは env.step の配置探索のみなので、固定アクション列で env を直接回し、
スケジュール（per-step reward/scheduled/wait + 最終 cost/avgwait）のハッシュを取る。
fast_env 0/1 で thash が一致すれば結果不変（変更点をピンポイント検証）。NN/GPU を介さない分、
多シード・多スケールを高速に回せる。

usage:
  PCN_FAST_ENV=1 NJ=512 P=0.7 EP=6 PYTHONPATH=. .venv/bin/python scripts/bench_env_alloc.py
"""
import os, sys, time, hashlib
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")

import numpy as np
from scripts.pcn_replay_snapshot import create_eval_env, load_config

NJ = int(os.environ.get("NJ", "512"))
P = float(os.environ.get("P", "0.7"))     # クラウド選択確率（0/1で片側に寄せ高競合に）
EP = int(os.environ.get("EP", "6"))
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_synthetic_pcn.yml")

cfg = load_config(CFG)
env = create_eval_env(cfg, job_seed=JOB_SEED, n_jobs=NJ)


def run_episode(action_seed: int) -> str:
    rng = np.random.default_rng(action_seed)
    env.reset()
    h = hashlib.sha1()
    done = False
    st = 0
    while not done and st < NJ + 16:
        a = 1 if rng.random() < P else 0
        obs, reward, scheduled, wt, done = env.step(a)
        h.update(np.asarray(reward, dtype=np.float64).tobytes())
        h.update(b"\x01" if scheduled else b"\x00")
        h.update(np.int64(int(wt)).tobytes())
        st += 1
    cost, mk, avgwt = env.calc_objective_values()
    h.update(np.asarray([cost, avgwt], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


run_episode(1)  # warm（jobs_set 生成・初回確保を計測から除外）
t0 = time.perf_counter()
hashes = [run_episode(1000 + e) for e in range(EP)]
dt = time.perf_counter() - t0
thash = hashlib.sha1("|".join(hashes).encode()).hexdigest()[:16]
print(f"RESULT fast_env={os.environ.get('PCN_FAST_ENV','1')} "
      f"alloc={os.environ.get('PCN_FAST_ENV_ALLOC', os.environ.get('PCN_FAST_ENV','1'))} "
      f"NJ={NJ} P={P} EP={EP} total={dt:.3f}s per_ep_ms={dt / EP * 1000:.1f} "
      f"per_step_us~{dt / EP / NJ * 1e6:.1f} thash={thash}", flush=True)
