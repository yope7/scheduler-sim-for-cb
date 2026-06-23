#!/usr/bin/env python3
"""DQN 重みスイープによる PF 構築（PCN と同一設定: event native env / trace256 / 同一NSGA真PF基準）。

スカラー化MORL: 各重み w_cost∈[0,1] で reward = -(wait/W)*w_wt - (cost/C)*w_cost を最大化する
DQNを1つ学習し、greedy方策の達成(cost, avg_wait)を1点として集める。重みスイープでPFを描く。

最適化(遅い既存実装の改善):
  - env を bitmap → **event native**(PCNと同じ高速env)
  - update を毎step → **K step ごと**(GD回数 1/K)
  - 重みスイープを **ProcessPoolで並列**(CPU worker, OMP=1)
  - スカラー化を **正規化**(cost~5.5e8 >> wait~1.5e5 で生だとcost支配→極端領域に偏る)
    NORM=0 で生スカラー化(極端偏りの確認用)も可。

usage:
  N_WEIGHTS=21 N_EPISODES=250 NPROC=16 NORM=1 OUT=results/eval_pf/dqn_sweep_256.npz \
  PYTHONPATH=. .venv/bin/python scripts/dqn_weight_sweep.py
"""
import os
import sys

for _a in sys.argv[1:]:
    if "=" in _a:
        k, v = _a.split("=", 1)
        os.environ[k] = v

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
# worker内のスレッド過剰割当を防ぐ(並列はプロセス側で取る)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_256_pcn.yml")
NJ = int(os.environ.get("NJ", "256"))
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))
N_WEIGHTS = int(os.environ.get("N_WEIGHTS", "21"))
N_EPISODES = int(os.environ.get("N_EPISODES", "250"))
NPROC = int(os.environ.get("NPROC", str(min(16, (os.cpu_count() or 4) - 2))))
NORM = os.environ.get("NORM", "1") == "1"
ALLOW_DEFER = os.environ.get("SCHEDULER_ALLOW_DEFER", "0") == "1"
UPDATE_EVERY = int(os.environ.get("DQN_UPDATE_EVERY", "4"))
OUT = os.environ.get("OUT", f"results/eval_pf/dqn_sweep_{NJ}{'_defer' if ALLOW_DEFER else ''}.npz")

# 正規化スケール(端点から固定; 行動依存しない定数)
C_SCALE = float(os.environ.get("DQN_C_SCALE", "556116624"))      # 全クラウドの総コスト
W_SCALE = float(os.environ.get("DQN_W_SCALE", str(153813.1 * NJ)))  # 全オンプレの総待ち


def _train_one_weight(args):
    """1つの重みで DQN を学習し greedy 達成(cost, avg_wait)を返す。worker プロセスで実行。"""
    w_idx, w_cost, seed = args
    import torch as th
    import torch.nn as nn
    from collections import deque
    import random
    from scripts.pcn_replay_snapshot import create_eval_env, load_config

    th.set_num_threads(1)
    rng = random.Random(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=JOB_SEED, n_jobs=NJ)
    sdim = env.observation_space.shape[0]
    adim = env.action_space.n
    w_wt = 1.0 - w_cost

    def scal(r0, r1):
        # env報酬は既に [-wait, -cost](負)。重み付けして合成するだけ(再否定しない)。
        # 正規化版は各成分をスケールで割る → 重み w_cost が公平にトレードオフを張る。
        if NORM:
            return (r0 / W_SCALE) * w_wt + (r1 / C_SCALE) * w_cost
        return r0 * w_wt + r1 * w_cost

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, adim))
        def forward(s, x):
            return s.f(x)

    pol, tgt = Net(), Net()
    tgt.load_state_dict(pol.state_dict())
    opt = th.optim.Adam(pol.parameters(), lr=1e-3)
    mem = deque(maxlen=50000)
    batch, gamma, target_every = 512, 0.99, 500
    eps, eps_min, eps_decay = 1.0, 0.05, 0.995
    gstep = 0

    def update():
        if len(mem) < batch:
            return
        b = rng.sample(mem, batch)
        s = th.as_tensor(np.array([x[0] for x in b]), dtype=th.float32)
        a = th.as_tensor([x[1] for x in b], dtype=th.long)
        r = th.as_tensor([x[2] for x in b], dtype=th.float32)
        ns = th.as_tensor(np.array([x[3] for x in b]), dtype=th.float32)
        d = th.as_tensor([x[4] for x in b], dtype=th.float32)
        q = pol(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with th.no_grad():
            nq = tgt(ns).max(1)[0]
            tq = r + (1 - d) * gamma * nq
        loss = nn.functional.mse_loss(q, tq)
        opt.zero_grad(); loss.backward(); opt.step()

    for ep in range(N_EPISODES):
        s = env.reset(); done = False; step = 0
        while not done:
            if rng.random() < eps:
                a = rng.randrange(adim)
            else:
                with th.no_grad():
                    a = int(pol(th.as_tensor(s, dtype=th.float32).unsqueeze(0)).argmax(1).item())
            ns, reward, scheduled, wt_step, done = env.step(a)
            rsc = scal(float(reward[0]), float(reward[1]))
            mem.append((np.asarray(s, dtype=np.float32), a, rsc, np.asarray(ns, dtype=np.float32), float(done)))
            s = ns; gstep += 1; step += 1
            if gstep % UPDATE_EVERY == 0:
                update()
            if gstep % target_every == 0:
                tgt.load_state_dict(pol.state_dict())
        eps = max(eps_min, eps * eps_decay)

    # greedy 評価
    s = env.reset(); done = False
    while not done:
        with th.no_grad():
            a = int(pol(th.as_tensor(s, dtype=th.float32).unsqueeze(0)).argmax(1).item())
        s, _, _, _, done = env.step(a)
    cost, _, avg_wait = env.calc_objective_values(calc_makespan=False)
    return (w_idx, float(w_cost), float(cost), float(avg_wait))


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    weights = np.linspace(0.0, 1.0, N_WEIGHTS)
    tasks = [(i, float(w), 1000 + i) for i, w in enumerate(weights)]
    print(f"[DQN_SWEEP] CFG={CFG} NJ={NJ} weights={N_WEIGHTS} eps/w={N_EPISODES} "
          f"NORM={NORM} defer={ALLOW_DEFER} nproc={NPROC} update_every={UPDATE_EVERY}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        futs = {ex.submit(_train_one_weight, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                print(f"  w_idx={r[0]:2d} w_cost={r[1]:.3f} → cost={r[2]:14.0f} avg_wait={r[3]:10.1f}", flush=True)
            except Exception as e:
                print(f"  w {futs[fut]} FAILED: {e}", flush=True)
    results.sort()
    pts = np.array([[r[2], r[3]] for r in results], dtype=np.float64)  # (cost, avg_wait)
    wcs = np.array([r[1] for r in results], dtype=np.float64)
    np.savez(OUT, points=pts, weight_cost=wcs,
             meta=np.array([("nj", NJ), ("n_weights", N_WEIGHTS), ("n_episodes", N_EPISODES),
                            ("norm", int(NORM)), ("defer", int(ALLOW_DEFER))], dtype=object))
    print(f"saved {OUT}: {len(pts)}点 cost[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
          f"wait[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]")
    # 極端偏りの簡易診断: cost軸で点がどれだけ散っているか
    if len(pts) > 2:
        cspan = (pts[:, 0].max() - pts[:, 0].min())
        uniq_frac = len(np.unique(np.round(pts[:, 0] / max(cspan, 1), 2))) / len(pts)
        print(f"[診断] コスト軸の点の散らばり(uniq率)={uniq_frac:.2f} (低いほど極端領域に密集=スカラー化の偏り)")


if __name__ == "__main__":
    main()
