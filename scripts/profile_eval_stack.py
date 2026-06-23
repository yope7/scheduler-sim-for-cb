#!/usr/bin/env python3
"""単一プロセスで eval(greedy rollout) の関数スタック時間を測る。
Ray を使わない=py-spy / cProfile が素直に効く代表ワークロード(env.step + NN 推論の経路)。

二役:
  MODE=cprofile (既定): cProfile で関数別 cumulative 時間を集計し pstats を保存+上位を表示。
                        → plot_eval_stack.py が pstats から関数時間バー/アイシクルPNGを描く。
  MODE=raw            : cProfile を付けずに同じ rollout をひたすら回す(py-spy record 用)。

usage:
  CKPT=.. CFG=.. NJ=512 NROLL=20 MODE=cprofile PSTATS=/tmp/eval_stack.pstats \
    PYTHONPATH=. .venv/bin/python scripts/profile_eval_stack.py
  # flamegraph:
  py-spy record -o flame.svg -- env CKPT=.. CFG=.. NJ=512 NROLL=100000 MODE=raw \
    PYTHONPATH=. .venv/bin/python scripts/profile_eval_stack.py
"""
import os
import time

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch as th

from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import objectives_to_command

CKPT = os.environ["CKPT"]
CFG = os.environ["CFG"]
NJ = int(os.environ.get("NJ", "512"))
NROLL = int(os.environ.get("NROLL", "20"))
MODE = os.environ.get("MODE", "cprofile")
PSTATS = os.environ.get("PSTATS", "/tmp/eval_stack.pstats")
SEED = int(os.environ.get("SEED", "0"))


def build():
    th.set_num_threads(1)
    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=SEED, n_jobs=NJ)
    state = th.load(CKPT, map_location="cpu", weights_only=False)
    use_enh = (state.get("model_type", "") == "EnhancedPCNModel")
    ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
             scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32), learning_rate=1e-3,
             batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
             use_enhanced_model=use_enh)
    tg = ag.network if ag.use_enhanced_model else ag.model
    tg.load_state_dict(state.get("model_state_dict", state), strict=False)
    tg.eval()
    return env, ag


def rp_range(env):
    """全クラウド〜全オンプレ掃引でコスト範囲を得る(command 目標を作るため)。"""
    lo = hi = None
    for p in (0.0, 1.0):
        env.reset(); tc = 0.0; done = False; st = 0; rng = np.random.default_rng(7)
        while not done and st < NJ + 5:
            a = 1 if rng.random() < p else 0
            r = env.step(a); tc += -float(r[1][1]); done = r[-1]; st += 1
        lo = tc if lo is None else min(lo, tc); hi = tc if hi is None else max(hi, tc)
    return lo, hi


def workload(env, ag, nroll):
    """nroll 個の greedy rollout をコスト範囲に渡って回す(eval と同じ経路)。"""
    mx = np.full(2, np.inf, dtype=np.float32)
    lo, hi = rp_range(env)
    cmds = [objectives_to_command(float(c), 0.0, NJ).astype(np.float32)
            for c in np.linspace(lo, hi, max(2, min(nroll, 40)))]
    n = 0
    while n < nroll:
        dr = cmds[n % len(cmds)]
        ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=True)
        n += 1
    return n


def main():
    env, ag = build()
    # ウォームアップ(JIT/キャッシュ確定)
    workload(env, ag, 2)
    if MODE == "raw":
        print(f"[raw] looping greedy rollouts NJ={NJ} (py-spy record me) ...", flush=True)
        workload(env, ag, NROLL)
        return
    import cProfile
    import pstats
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    n = workload(env, ag, NROLL)
    pr.disable()
    dt = time.perf_counter() - t0
    pr.dump_stats(PSTATS)
    st = pstats.Stats(pr)
    total = st.total_tt
    print(f"\n=== eval stack profile: NJ={NJ} rollouts={n} wall={dt:.2f}s "
          f"({dt / n * 1000:.1f} ms/rollout, {dt / n / NJ * 1000:.3f} ms/step) ===")
    print(f"pstats -> {PSTATS}   (tottime sum={total:.2f}s)")
    print("\ntop 20 by cumulative time:")
    st.sort_stats("cumulative").print_stats(20)
    print("\ntop 15 by total(self) time:")
    st.sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
