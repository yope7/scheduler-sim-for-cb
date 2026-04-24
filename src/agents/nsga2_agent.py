from __future__ import annotations

import datetime
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from morl_baselines.common.performance_indicators import hypervolume

try:
    import nsga2_core
    C_NSGA2_AVAILABLE = True
except Exception:
    C_NSGA2_AVAILABLE = False


def _build_env_params(env):
    return {
        "max_step": env.max_step,
        "n_window": env.n_window,
        "n_on_premise_node": env.n_on_premise_node,
        "n_cloud_node": env.n_cloud_node,
        "n_job_queue_obs": env.n_job_queue_obs,
        "n_job_queue_bck": env.n_job_queue_bck,
        "weight_wt": env.weight_wt,
        "weight_cost": env.weight_cost,
        "penalty_not_allocate": env.penalty_not_allocate,
        "penalty_invalid_action": env.penalty_invalid_action,
        "jobs_set": env.jobs_set,
        "flag": 0,
    }


def _evaluate_worker(args):
    chromosome, env_params = args
    from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized

    env = SchedulingEnvCacheOptimized(**env_params)
    env.reset()
    done = False
    step_idx = 0
    while not done:
        action = chromosome[step_idx] if step_idx < len(chromosome) else 0
        _, _, scheduled, _, done = env.step(action)
        if scheduled:
            step_idx += 1
        if done:
            env.finalize_window_history()
    cost, _, avg_waiting_time = env.calc_objective_values()
    return np.array([cost, avg_waiting_time], dtype=np.float64)


@dataclass
class Individual:
    chromosome: List[int]
    objectives: Optional[np.ndarray] = None
    rank: int = 0
    crowding_distance: float = 0.0

    def dominates(self, other: "Individual") -> bool:
        if self.objectives is None or other.objectives is None:
            return False
        better = False
        for a, b in zip(self.objectives, other.objectives):
            if a > b:
                return False
            if a < b:
                better = True
        return better


class NSGA2Agent:
    """Non-distributed NSGA-II (direct minimization of solution values)."""

    def __init__(
        self,
        pop_size: int = 200,
        num_generations: int = 150,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
        tournament_size: int = 2,
        eliminate_duplicates: bool = False,
    ):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.eliminate_duplicates = eliminate_duplicates
        self.population: List[Individual] = []
        self.history = {"pareto_fronts": [], "all_solutions": []}
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_dir = f"execution_{ts}"
        os.makedirs(self.execution_dir, exist_ok=True)

    def initialize_population(self, nb_jobs: int):
        self.population = []
        for _ in range(self.pop_size):
            chrom = np.random.randint(0, 2, size=nb_jobs, dtype=np.int32).tolist()
            self.population.append(Individual(chromosome=chrom))

    def _evaluate_population(self, env, n_jobs=-1):
        env_params = _build_env_params(env)
        tasks = [(ind.chromosome, env_params) for ind in self.population]
        if n_jobs == 1:
            results = [_evaluate_worker(t) for t in tasks]
        else:
            n_proc = max(1, os.cpu_count() // 2) if n_jobs == -1 else max(1, int(n_jobs))
            with mp.Pool(processes=n_proc) as pool:
                results = pool.map(_evaluate_worker, tasks)
        for ind, obj in zip(self.population, results):
            ind.objectives = obj

    def non_dominated_sort(self):
        n = len(self.population)
        dom_count = [0] * n
        dom_set = [[] for _ in range(n)]
        for i in range(n):
            self.population[i].rank = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.population[i].dominates(self.population[j]):
                    dom_set[i].append(j)
                elif self.population[j].dominates(self.population[i]):
                    dom_count[i] += 1
        rank = 1
        front = [i for i in range(n) if dom_count[i] == 0]
        for i in front:
            self.population[i].rank = rank
        while front:
            next_front = []
            for i in front:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        self.population[j].rank = rank + 1
                        next_front.append(j)
            rank += 1
            front = next_front

    def calculate_crowding_distance(self):
        for ind in self.population:
            ind.crowding_distance = 0.0
        ranks = sorted(set(ind.rank for ind in self.population if ind.objectives is not None))
        for r in ranks:
            front = [ind for ind in self.population if ind.rank == r and ind.objectives is not None]
            if len(front) <= 2:
                for ind in front:
                    ind.crowding_distance = float("inf")
                continue
            objs = np.array([ind.objectives for ind in front], dtype=np.float64)
            if C_NSGA2_AVAILABLE:
                d = nsga2_core.calculate_crowding_distance(objs)
            else:
                d = np.zeros(len(front), dtype=np.float64)
                for k in range(objs.shape[1]):
                    order = np.argsort(objs[:, k])
                    d[order[0]] = np.inf
                    d[order[-1]] = np.inf
                    span = objs[order[-1], k] - objs[order[0], k]
                    if span == 0:
                        continue
                    for i in range(1, len(front) - 1):
                        d[order[i]] += (objs[order[i + 1], k] - objs[order[i - 1], k]) / span
            for ind, dist in zip(front, d):
                ind.crowding_distance = float(dist)

    def _tournament(self):
        cand = random.sample(self.population, self.tournament_size)
        cand.sort(key=lambda x: (x.rank, -x.crowding_distance))
        return cand[0]

    def crossover(self, p1: Individual, p2: Individual):
        if random.random() > self.crossover_prob:
            return Individual(p1.chromosome.copy()), Individual(p2.chromosome.copy())
        n = len(p1.chromosome)
        point = random.randint(1, n - 1)
        c1 = p1.chromosome[:point] + p2.chromosome[point:]
        c2 = p2.chromosome[:point] + p1.chromosome[point:]
        return Individual(c1), Individual(c2)

    def mutation(self, ind: Individual):
        arr = np.array(ind.chromosome, dtype=np.int32)
        mask = np.random.random(len(arr)) < self.mutation_prob
        arr[mask] = 1 - arr[mask]
        ind.chromosome = arr.tolist()

    def create_offspring(self):
        offspring: List[Individual] = []
        while len(offspring) < self.pop_size:
            p1 = self._tournament()
            p2 = self._tournament()
            c1, c2 = self.crossover(p1, p2)
            self.mutation(c1)
            self.mutation(c2)
            offspring.extend([c1, c2])
        return offspring[: self.pop_size]

    def save_pareto_front(self):
        pf = [ind.objectives for ind in self.population if ind.rank == 1 and ind.objectives is not None]
        allv = [ind.objectives for ind in self.population if ind.objectives is not None]
        self.history["pareto_fronts"].append(np.array(pf, dtype=np.float64) if pf else np.zeros((0, 2)))
        self.history["all_solutions"].append(np.array(allv, dtype=np.float64) if allv else np.zeros((0, 2)))

    def run(self, env, nb_jobs: int, verbose: bool = True, n_jobs=-1):
        self.initialize_population(nb_jobs)
        self._evaluate_population(env, n_jobs=n_jobs)
        self.non_dominated_sort()
        self.calculate_crowding_distance()
        self.save_pareto_front()

        for gen in range(1, self.num_generations + 1):
            parents = self.population.copy()
            self.population = self.create_offspring()
            self._evaluate_population(env, n_jobs=n_jobs)
            self.population = parents + self.population
            self.non_dominated_sort()
            self.calculate_crowding_distance()

            next_pop = []
            r = 1
            while len(next_pop) < self.pop_size:
                front = [ind for ind in self.population if ind.rank == r]
                if not front:
                    break
                remain = self.pop_size - len(next_pop)
                if len(front) <= remain:
                    next_pop.extend(front)
                else:
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    next_pop.extend(front[:remain])
                r += 1
            self.population = next_pop
            self.save_pareto_front()
            if verbose and (gen % 10 == 0 or gen == self.num_generations):
                print(f"世代 {gen} - PF size: {len([i for i in self.population if i.rank == 1])}")

        return self.get_final_pareto_front()

    def get_final_pareto_front(self):
        pf = [ind for ind in self.population if ind.rank == 1 and ind.objectives is not None]
        return {
            "objectives": np.array([ind.objectives for ind in pf], dtype=np.float64) if pf else np.zeros((0, 2)),
            "chromosomes": [ind.chromosome for ind in pf],
        }

    def calc_hypervolume(self):
        objectives = np.array([ind.objectives for ind in self.population if ind.objectives is not None], dtype=np.float64)
        if len(objectives) == 0:
            return 0.0
        return hypervolume([200000, 20], objectives)
