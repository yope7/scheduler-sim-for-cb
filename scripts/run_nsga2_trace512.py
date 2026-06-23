#!/usr/bin/env python
"""NSGA-II を PCN eval と同一の trace ワークロード(create_eval_env と同じ JobGenerator 経路)で実行し、
最終 PF・世代ごと PF 履歴・最終 PF の遺伝子を npz に保存する。

PCN の truepf_*.npz (greedy/rp) と同じ座標系 (cost, avg_wait) なので、そのまま重ね描きできる
（等価性は scripts/verify_nsga2_eval_equiv.py で照合済みであること）。

usage: CFG=experiments/distributed_pcn/job_trace_512_pcn.yml NJ=512 POP=200 GEN=150 \
       MUT=auto NPROC=32 OUT=results/eval_pf/nsga2_trace512_s0.npz \
       PYTHONPATH=. .venv/bin/python scripts/run_nsga2_trace512.py
   MUT=auto は 1/n_jobs（binary GA の標準）。既存実装の既定値を試すなら MUT=0.1。
"""
import os
import time

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

import numpy as np

from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.nsga2_agent import NSGA2Agent

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_512_pcn.yml")
NJ = int(os.environ.get("NJ", "512"))
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
POP = int(os.environ.get("POP", "200"))
GEN = int(os.environ.get("GEN", "150"))
MUT = os.environ.get("MUT", "auto")
CX = float(os.environ.get("CX", "0.9"))
NPROC = int(os.environ.get("NPROC", str(max(1, (os.cpu_count() or 2) // 2))))
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", f"results/eval_pf/nsga2_trace{NJ}_s{JOB_SEED}.npz")

config = load_config(CFG)
env = create_eval_env(config, job_seed=JOB_SEED, n_jobs=NJ)
mut = (1.0 / NJ) if MUT == "auto" else float(MUT)
print(f"NSGA-II: NJ={NJ} job_seed={JOB_SEED} pop={POP} gen={GEN} mut={mut:.4g} cx={CX} "
      f"nproc={NPROC} seed={SEED}", flush=True)

agent = NSGA2Agent(pop_size=POP, num_generations=GEN, crossover_prob=CX, mutation_prob=mut)
t0 = time.time()
res = agent.run(env, NJ, n_jobs=NPROC, seed=SEED, seed_extremes=True)
elapsed = time.time() - t0

pf = res["objectives"]
order = np.argsort(pf[:, 0])
pf = pf[order]
chroms = np.array([res["chromosomes"][i] for i in order], dtype=np.int8)
gen_pf = np.empty(len(agent.history["pareto_fronts"]), dtype=object)
for i, p in enumerate(agent.history["pareto_fronts"]):
    gen_pf[i] = p

np.savez(
    OUT,
    pf=pf,
    chromosomes=chroms,
    gen_pf=gen_pf,
    meta=np.array(
        [("pop", POP), ("gen", GEN), ("mut", mut), ("cx", CX), ("seed", SEED),
         ("job_seed", JOB_SEED), ("nj", NJ), ("elapsed_sec", elapsed),
         ("n_evaluations", agent.n_evaluations)],
        dtype=object,
    ),
)
print(f"saved {OUT}: PF {len(pf)}点 evals={agent.n_evaluations} elapsed={elapsed:.1f}s")
print(f"  cost range: {pf[:, 0].min():.0f} – {pf[:, 0].max():.0f}")
print(f"  wait range: {pf[:, 1].min():.2f} – {pf[:, 1].max():.2f}")
