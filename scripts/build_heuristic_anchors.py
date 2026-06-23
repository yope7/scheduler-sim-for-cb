#!/usr/bin/env python3
"""道B: 弱アンカー(ヒューリスティック)生成。NSGA-II探索なしでアンカー集合を作る。

WaitTimeThreshold(現ジョブのオンプレ予測待ち>=閾値ならクラウド)を閾値スイープで実行し、
各閾値の (行動列=遺伝子, 達成値=(cost, avg_wait)) を集めて nsga2 npz と同形式で保存する。
→ AnchorSet.from_npz でそのまま読める。探索器に依存しないので未知ワークロードにも即適用可能。

これは「アンカーが自明な最適解でない」設定を作るためのもの。マッチしたNSGAアンカーでは
eval が縮退する(純followで100%)が、弱いアンカーなら方策が flip で価値を足す余地と必要が生まれる。

usage:
  CFG=experiments/distributed_pcn/job_trace_256_pcn.yml NJ=256 JOB_SEED=0 \
  OUT=results/eval_pf/heur_anchors_256_s0.npz \
  PYTHONPATH=. .venv/bin/python scripts/build_heuristic_anchors.py
"""
import os
import sys

for _a in sys.argv[1:]:
    if "=" in _a:
        k, v = _a.split("=", 1)
        os.environ[k] = v

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

import numpy as np

from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import get_non_dominated_inds_minimize

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_256_pcn.yml")
NJ = int(os.environ.get("NJ", "256"))
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
OUT = os.environ.get("OUT", f"results/eval_pf/heur_anchors_{NJ}_s{JOB_SEED}.npz")
# 閾値スイープ: 0(待ち発生で即クラウド=高コスト低待ち) 〜 大(全オンプレ=低コスト高待ち)
THRS = os.environ.get("THRS", "0,30,60,120,250,500,1000,2000,4000,8000,16000,32000,64000,999999999")
thresholds = [float(x) for x in THRS.split(",") if x.strip()]


def _predicted_wait_action(env, threshold):
    j = int(env.index_next_job)
    if j >= len(env.jobs):
        return 0
    raw_job = env.jobs[j]
    job = env._to_queue_job(raw_job)
    arrival = int(raw_job[0])
    _, onprem_start = env._find_event_allocation(job, False, arrival)
    predicted_wait = int(onprem_start) - arrival
    return 1 if predicted_wait >= float(threshold) else 0


def rollout_threshold(env, threshold):
    """閾値ヒューリスティックで1エピソード回し、(遺伝子, cost, avg_wait) を返す。
    遺伝子は scheduled 時のみ前進(nsga2_agent._rollout / AnchorSet と同規約)。"""
    env.reset()
    done = False
    gene = []
    while not done:
        try:
            a = _predicted_wait_action(env, threshold)
        except Exception:
            a = 0
        _, _, scheduled, _, done = env.step(a)
        if scheduled:
            gene.append(int(a))
        if done:
            env.finalize_window_history()
    cost, _, avg_wait = env.calc_objective_values(calc_makespan=False)
    # 遺伝子長を NJ に揃える(末尾不足は0=オンプレで埋め)
    if len(gene) < NJ:
        gene = gene + [0] * (NJ - len(gene))
    return np.array(gene[:NJ], dtype=np.int8), float(cost), float(avg_wait)


def main():
    env = create_eval_env(load_config(CFG), job_seed=JOB_SEED, n_jobs=NJ)
    genes, pf = [], []
    for th in thresholds:
        g, c, w = rollout_threshold(env, th)
        genes.append(g)
        pf.append([c, w])
        print(f"  th={th:>10.0f} → cost={c:>14.0f} wait={w:>10.1f}")
    genes = np.array(genes, dtype=np.int8)
    pf = np.array(pf, dtype=np.float64)
    # 非劣解のみ残す(支配される閾値点はアンカーとして無価値)
    nd = get_non_dominated_inds_minimize(pf)
    genes, pf = genes[nd], pf[nd]
    order = np.argsort(pf[:, 0])
    genes, pf = genes[order], pf[order]
    np.savez(OUT, pf=pf, chromosomes=genes,
             meta=np.array([("source", "waittime_threshold"), ("nj", NJ),
                            ("job_seed", JOB_SEED), ("n_anchors", len(pf))], dtype=object))
    print(f"saved {OUT}: {len(pf)} 弱アンカー(非劣) "
          f"cost[{pf[:,0].min():.0f},{pf[:,0].max():.0f}] wait[{pf[:,1].min():.1f},{pf[:,1].max():.1f}]")


if __name__ == "__main__":
    main()
