#!/usr/bin/env python
"""大規模ジョブ数(1万〜5万)向け NSGA-II 参照PF作成。run_nsga2_trace512.py の大規模版。

run_nsga2_trace512.py との差分（すべて大規模で必要になったもの）:
  1. param_job.job_trace_n_jobs を NJ に合わせて上書き（元スクリプトは config の値のまま
     CSV を読むので NJ != config 値だと JobGenerator と env で件数が食い違う）
  2. p 掃引個体を初期集団に注入（PSEED=個数）。2目的PFはクラウド率 p のマクロで
     ほぼ決まるので、骨格を種で与えると初期世代から実用的な PF が立つ
  3. mp.Pool の chunksize を 1 に（既定 4 は POP<4*NPROC で並列度が chunk 数に頭打ちする。
     さらに個体ごとの評価時間が p 依存で最大 7 倍違うため静的 chunk は不利）
  4. 世代ごとに npz を上書き保存（CKPT_EVERY）。長時間ジョブを任意の世代で打ち切っても
     そこまでの PF が残る = 打ち切り基準を運用可能にする
  5. 世代ごとに壁時計と evals/s を出力（残り時間の実測見積もり）

usage:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CFG=experiments/distributed_pcn/job_trace_weekB_full_pcn.yml NJ=20000 \
  POP=200 GEN=12 PSEED=24 NPROC=64 OUT=results/eval_pf/nsga2_trace20000_s0.npz \
  PYTHONPATH=. .venv/bin/python scripts/run_nsga2_bigjobs.py
"""
import multiprocessing as mp
import os
import time

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

import numpy as np
import yaml

from scripts.pcn_replay_snapshot import create_eval_env
from src.agents import nsga2_agent as N
from src.agents.nsga2_agent import Individual, NSGA2Agent

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_weekB_full_pcn.yml")
NJ = int(os.environ.get("NJ", "20000"))
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
POP = int(os.environ.get("POP", "200"))
GEN = int(os.environ.get("GEN", "12"))
PSEED = int(os.environ.get("PSEED", "24"))
MUT = os.environ.get("MUT", "auto")
CX = float(os.environ.get("CX", "0.9"))
NPROC = int(os.environ.get("NPROC", str(max(1, (os.cpu_count() or 2)))))
SEED = int(os.environ.get("SEED", "0"))
CHUNK = int(os.environ.get("CHUNK", "1"))
OUT = os.environ.get("OUT", f"results/eval_pf/nsga2_trace{NJ}_s{JOB_SEED}.npz")
# 多族掃引(family_sweep.py)の個体を初期集団に種として入れる(既定ON)
FAMSEED = os.environ.get("FAMSEED", "1") not in ("0", "", "off")
FAMSEED_FAMILIES = os.environ.get(
    "FAMSEED_FAMILIES", "big,small,long,short,wide,fcfs,lifo,burst,random").split(",")
FAMSEED_FRACS = [float(x) for x in os.environ.get(
    "FAMSEED_FRACS", "0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.65,0.80,0.90").split(",")]

cfg = yaml.safe_load(open(CFG))
cfg["param_job"]["job_trace_n_jobs"] = NJ
cfg["param_env"]["n_jobs"] = NJ
env = create_eval_env(cfg, job_seed=JOB_SEED, n_jobs=NJ)
mut = (1.0 / NJ) if MUT == "auto" else float(MUT)
print(f"NSGA-II big: NJ={NJ} cap={cfg['param_env']['n_on_premise_node']}/"
      f"{cfg['param_env']['n_cloud_node']} pop={POP} gen={GEN} pseed={PSEED} "
      f"mut={mut:.4g} nproc={NPROC} chunk={CHUNK} out={OUT}", flush=True)


def _family_seed_chromosomes(nb_jobs):
    """多族掃引(scripts/family_sweep.py)の個体を種として作る。

    素の p 掃引だけを種にすると、変異率 1/20000 では「どのジョブを送るか」の並べ替えに
    実質到達できない(1世代あたり平均1ビットしか動かない)。族の骨格を初期集団に入れて
    そこから改良させる。build_order は family_sweep と共有(決定論的)。
    """
    from scripts.family_sweep import build_order

    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64)[:nb_jobs]
    out = []
    for fam in FAMSEED_FAMILIES:
        order = build_order(fam, jobs)
        for fr in FAMSEED_FRACS:
            k = int(round(fr * nb_jobs))
            a = np.zeros(nb_jobs, dtype=np.int32)
            a[order[:k]] = 1
            out.append(a)
    return out


class BigNSGA2(NSGA2Agent):
    """初期集団への p 掃引 + 多族種の注入 + chunksize 制御 + 世代ごと保存。探索規則は親と同一。"""

    def initialize_population(self, nb_jobs, seed_extremes=False):
        self.population = []
        if FAMSEED:
            fam_chroms = _family_seed_chromosomes(nb_jobs)
            for a in fam_chroms:
                self.population.append(Individual(chromosome=a.tolist()))
            print(f"  多族種を {len(fam_chroms)} 個体注入 "
                  f"(族{len(FAMSEED_FAMILIES)}×割合{len(FAMSEED_FRACS)})", flush=True)
        if PSEED > 0:
            rng = np.random.default_rng(20260815)
            # 端(全オンプレ/全クラウド)を含む p 掃引。PF の骨格を初期世代から与える
            for p in np.linspace(0.0, 1.0, PSEED):
                if p <= 0.0:
                    a = np.zeros(nb_jobs, dtype=np.int32)
                elif p >= 1.0:
                    a = np.ones(nb_jobs, dtype=np.int32)
                else:
                    a = (rng.random(nb_jobs) < p).astype(np.int32)
                self.population.append(Individual(chromosome=a.tolist()))
        elif seed_extremes:
            self.population.append(Individual(chromosome=[0] * nb_jobs))
            self.population.append(Individual(chromosome=[1] * nb_jobs))
        while len(self.population) < self.pop_size:
            self.population.append(
                Individual(chromosome=np.random.randint(0, 2, size=nb_jobs,
                                                        dtype=np.int32).tolist()))

    def _evaluate_population(self, env, n_jobs=-1):
        env_params = N._build_env_params(env)
        todo = []
        for ind in self.population:
            if ind.objectives is None:
                cached = self._eval_cache.get(tuple(ind.chromosome))
                if cached is not None:
                    ind.objectives = cached.copy()
                else:
                    todo.append(ind)
        if not todo:
            return
        self.n_evaluations += len(todo)
        n_proc = max(1, os.cpu_count() // 2) if n_jobs == -1 else max(1, int(n_jobs))
        if self._pool is None:
            self._pool = mp.Pool(processes=n_proc, initializer=N._pool_init,
                                 initargs=(env_params,))
        results = self._pool.map(N._pool_eval, [i.chromosome for i in todo],
                                 chunksize=CHUNK)
        for ind, obj in zip(todo, results):
            ind.objectives = obj
            self._eval_cache[tuple(ind.chromosome)] = obj.copy()


def save(agent, elapsed):
    res = agent.get_final_pareto_front()
    pf = res["objectives"]
    if len(pf) == 0:
        return 0
    order = np.argsort(pf[:, 0])
    pf = pf[order]
    chroms = np.array([res["chromosomes"][i] for i in order], dtype=np.int8)
    gen_pf = np.empty(len(agent.history["pareto_fronts"]), dtype=object)
    for i, p in enumerate(agent.history["pareto_fronts"]):
        gen_pf[i] = p
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    np.savez(OUT, pf=pf, chromosomes=chroms, gen_pf=gen_pf,
             meta=np.array([("pop", POP), ("gen", GEN), ("mut", mut), ("cx", CX),
                            ("seed", SEED), ("job_seed", JOB_SEED), ("nj", NJ),
                            ("pseed", PSEED), ("elapsed_sec", elapsed),
                            ("n_evaluations", agent.n_evaluations),
                            ("gens_done", len(agent.history["pareto_fronts"]) - 1)],
                           dtype=object))
    return len(pf)


agent = BigNSGA2(pop_size=POP, num_generations=GEN, crossover_prob=CX, mutation_prob=mut)
np.random.seed(SEED)
import random as _r  # noqa: E402

_r.seed(SEED)
t0 = time.time()
try:
    agent.initialize_population(NJ)
    agent._evaluate_population(env, n_jobs=NPROC)
    agent.non_dominated_sort()
    agent.calculate_crowding_distance()
    agent.save_pareto_front()
    print(f"[gen 0] evals={agent.n_evaluations} elapsed={time.time() - t0:.1f}s "
          f"pf={save(agent, time.time() - t0)}点", flush=True)
    for gen in range(1, GEN + 1):
        parents = agent.population.copy()
        agent.population = agent.create_offspring()
        agent._evaluate_population(env, n_jobs=NPROC)
        agent.population = parents + agent.population
        agent.non_dominated_sort()
        agent.calculate_crowding_distance()
        nxt, r = [], 1
        while len(nxt) < agent.pop_size:
            front = [i for i in agent.population if i.rank == r]
            if not front:
                break
            rem = agent.pop_size - len(nxt)
            if len(front) <= rem:
                nxt.extend(front)
            else:
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                nxt.extend(front[:rem])
            r += 1
        agent.population = nxt
        agent.save_pareto_front()
        el = time.time() - t0
        npts = save(agent, el)
        print(f"[gen {gen}/{GEN}] pf={npts}点 evals={agent.n_evaluations} "
              f"elapsed={el / 60:.1f}min  {agent.n_evaluations / el:.2f} evals/s  "
              f"ETA_all={el / gen * (GEN - gen) / 60:.0f}min", flush=True)
finally:
    agent._close_pool()

el = time.time() - t0
n = save(agent, el)
d = np.load(OUT, allow_pickle=True)
pf = d["pf"]
print(f"saved {OUT}: PF {n}点 evals={agent.n_evaluations} elapsed={el / 60:.1f}min")
print(f"  cost range: {pf[:, 0].min():.4g} – {pf[:, 0].max():.4g}")
print(f"  wait range: {pf[:, 1].min():.4g} – {pf[:, 1].max():.4g}")
