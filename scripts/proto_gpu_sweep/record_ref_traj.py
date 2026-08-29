"""本体 env（SchedulingEnvEventNative）の参照軌跡を記録する（Phase 1 一致検証の地上真実）。

bench_env_alloc.py :28-46 の env 生成・step 呼びの流儀を流用:
  create_eval_env(cfg, JOB_SEED, NJ) + 固定 seed の Bernoulli(p) 行動列。
各 step の (start_time, nodes, waiting_time, cost, reward, scheduled, done) と
最終 (cost, avgwait) を npz に保存する。PCN_FAST_ENV=0/1 を切り替えて 2 回実行すれば
prune + sweep/C 経路の有無どちらの参照も取れる（verify_batch_env.py が突き合わせる）。

usage:
  PCN_FAST_ENV=1 NJ=128 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/record_ref_traj.py
  env: NJ=128 NEP=16 PS="0.3,0.7" JOB_SEED=0 OUT=scripts/proto_gpu_sweep/data
"""
from __future__ import annotations

import os
import sys
import time

# 学習系の既定に合わせる（bench_env_alloc.py と同じ）。SCHEDULER_OBS_URGENCY は
# 観測にしか影響しない（REUSE_URGENCY_ALLOC でも配置結果はビット一致検証済み）ため、
# 記録の高速化のみを目的に 0 とする（obs は Phase 1 のスコープ外）。
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "0")

import numpy as np

from scripts.pcn_replay_snapshot import create_eval_env, load_config

NJ = int(os.environ.get("NJ", "128"))
NEP = int(os.environ.get("NEP", "16"))          # p ごとの行動列本数
PS = [float(x) for x in os.environ.get("PS", "0.3,0.7").split(",")]
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_synthetic_pcn.yml")
OUT_DIR = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "data"))


def gen_actions(p: float, seed: int, n: int) -> np.ndarray:
    """bench_env_alloc.py run_episode と同じ抽選（seed 固定 Bernoulli）。"""
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p).astype(np.int32)


def main():
    fast = os.environ.get("PCN_FAST_ENV", "1")
    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=JOB_SEED, n_jobs=NJ)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()   # (NJ,8)
    assert jobs.shape == (NJ, 8), jobs.shape
    h_pad = int(jobs[:, 2].max())

    # 行動列: p ごとに seed 系列を分離（p0.3→1000+i, p0.7→2000+i, ...）
    episodes = []          # (p, seed, actions)
    for pi, p in enumerate(PS):
        for i in range(NEP):
            seed = 1000 * (pi + 1) + i
            episodes.append((p, seed, gen_actions(p, seed, NJ)))
    B = len(episodes)

    actions = np.stack([e[2] for e in episodes])            # (B,NJ)
    start = np.zeros((B, NJ), dtype=np.int64)
    wait = np.zeros((B, NJ), dtype=np.int64)
    cost = np.zeros((B, NJ), dtype=np.int64)
    rew = np.zeros((B, NJ, 2), dtype=np.float64)
    sched = np.zeros((B, NJ), dtype=bool)
    done = np.zeros((B, NJ), dtype=bool)
    nodes = np.full((B, NJ, h_pad), -1, dtype=np.int32)
    final_cost = np.zeros(B, dtype=np.float64)
    final_avgwait = np.zeros(B, dtype=np.float64)

    t0 = time.perf_counter()
    for b, (p, seed, acts) in enumerate(episodes):
        env.reset()
        assert np.array_equal(np.asarray(env.jobs, dtype=np.float64), jobs)
        for t in range(NJ):
            _, r, sc, wt, dn = env.step(int(acts[t]))
            rec = env._absolute_schedule_records[-1]
            nd = rec["nodes"]
            start[b, t] = int(rec["start"])
            nodes[b, t, : len(nd)] = nd
            wait[b, t] = int(wt)
            cost[b, t] = int(-r[1])
            rew[b, t] = np.asarray(r, dtype=np.float64)
            sched[b, t] = bool(sc)
            done[b, t] = bool(dn)
        assert done[b, -1], "episode did not finish in NJ steps"
        c, _, aw = env.calc_objective_values(calc_makespan=False)
        final_cost[b] = float(c)
        final_avgwait[b] = float(aw)
    dt = time.perf_counter() - t0
    per_step_us = dt / (B * NJ) * 1e6

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"ref_nj{NJ}_seed{JOB_SEED}_fast{fast}.npz")
    np.savez_compressed(
        out,
        jobs=jobs,
        actions=actions,
        start=start,
        wait=wait,
        cost=cost,
        rew=rew,
        sched=sched,
        done=done,
        nodes=nodes,
        final_cost=final_cost,
        final_avgwait=final_avgwait,
        ps=np.asarray([e[0] for e in episodes]),
        seeds=np.asarray([e[1] for e in episodes]),
        per_step_us=np.float64(per_step_us),
        fast_env=np.int32(int(fast)),
        n_on=np.int32(env.n_on_premise_node),
        n_cl=np.int32(env.n_cloud_node),
    )
    print(
        f"WROTE {out}  B={B} NJ={NJ} fast_env={fast} "
        f"total={dt:.2f}s per_step_us={per_step_us:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
