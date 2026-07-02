#!/usr/bin/env python3
"""週次トレースへのローリング窓適用評価。

256ジョブ窓で学習した方策を、一週間分(数万ジョブ)のトレースに1エピソードで適用する。
命令(desired_return)は「窓スケール」のまま、WINDOW ジョブ配置するごとにリセットする
(=実運用のストリーミング適用と同型)。資源の占有状態はエピソードを通して連続。
学習には一切触れない読み取り専用の評価。

usage: CKPT=.. CFG=.. NJ=16000 WINDOW=256 FRACS=0.2,0.5,0.8 OUT=/tmp/weekly.json \
       PYTHONPATH=. .venv/bin/python scripts/eval_weekly_rolling.py
"""
import os, time, json

os.environ.update({
    "DISTRIBUTED_PCN_USE_EVENT_OBS": "1",
    "DISTRIBUTED_PCN_USE_EVENT_NATIVE": "1",
    "SCHEDULER_LEARNER_BITMAP": "0",
    "SCHEDULER_OBS_URGENCY": os.environ.get("OBS_URGENCY", "0"),
    "SCHEDULER_OBS_OCCUPANCY": "1",
    "SCHEDULER_ALLOW_DEFER": "1",
    "SCHEDULER_DEFER_OFFSET": "1",
    "PCN_FILM": "1",
    "PCN_FOURIER_CMD": "1",
    "PCN_FOURIER_BANDS": os.environ.get("PCN_FOURIER_BANDS", "4"),
})
import numpy as np
import torch as th

from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import objectives_to_command

CKPT = os.environ["CKPT"]
CFG = os.environ["CFG"]
NJ = int(os.environ.get("NJ", "16000"))
WINDOW = int(os.environ.get("WINDOW", "256"))
FRACS = [float(x) for x in os.environ.get("FRACS", "0.2,0.5,0.8").split(",")]
OUT = os.environ.get("OUT", "/tmp/weekly_rolling.json")

env = create_eval_env(load_config(CFG), job_seed=0, n_jobs=NJ)
st = th.load(CKPT, map_location="cpu", weights_only=False)
ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / WINDOW], dtype=np.float32), learning_rate=1e-3,
         batch_size=512, hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
         project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False)
ag.model.load_state_dict(st.get("model_state_dict", st), strict=False)
ag.model.eval()
th.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


# 窓の実スケール(学習時の WORKLOAD_CALIB と同じ意味): frac を実値の (cost, wait) 目標に変換する。
# frac=0 → 全オンプレ端(cost 0, wait 大) / frac=1 → 全クラウド端(cost 大, wait 小)。
# objectives_to_command は実スケールの目的値を受け取る(割合ではない)。
COST_CL = float(os.environ.get("WIN_COST_CLOUD", "4.0e6"))   # 窓 all-cloud の cost
WAIT_OP = float(os.environ.get("WIN_WAIT_ONPREM", "7393.0"))  # 窓 all-onprem の avg_wait
WAIT_CL = float(os.environ.get("WIN_WAIT_CLOUD", "717.0"))    # 窓 all-cloud の avg_wait


def rolling_pass(frac: float):
    """窓スケール命令 frac で週全体を1パス。窓ごとに命令をリセット。"""
    cost_t = float(frac) * COST_CL
    wait_t = WAIT_OP + float(frac) * (WAIT_CL - WAIT_OP)  # frac↑でwait目標↓
    win_cmd = objectives_to_command(cost_t, wait_t, WINDOW).astype(np.float32)
    obs = env.reset()
    done = False
    dr = win_cmd.copy()
    hz = np.float32(WINDOW)
    placed_in_window = 0
    n_windows = 1
    t0 = time.perf_counter()
    with th.no_grad():
        while not done:
            policy_obs = ag._obs_for_policy(env, obs)
            action = ag._act(policy_obs, dr, hz, True)  # eval_mode=True (greedy)
            obs, reward, scheduled, _wt, done = env.step(action)
            dr = (dr - reward).astype(np.float32, copy=False)
            hz = np.float32(max(float(hz) - 1.0, 1.0))
            if scheduled:
                placed_in_window += 1
                if placed_in_window >= WINDOW and not done:
                    dr = win_cmd.copy()   # 窓命令リセット(ストリーミング適用と同型)
                    hz = np.float32(WINDOW)
                    placed_in_window = 0
                    n_windows += 1
    cost, _mk, avgwait = env.calc_objective_values()
    dt = time.perf_counter() - t0
    return dict(frac=frac, cost=float(cost), avg_wait=float(avgwait),
                n_windows=n_windows, pass_sec=round(dt, 1))


results = []
for frac in FRACS:
    r = rolling_pass(frac)
    results.append(r)
    print(f"frac={r['frac']:.2f} cost={r['cost']:.4e} avg_wait={r['avg_wait']:.1f} "
          f"windows={r['n_windows']} pass={r['pass_sec']}s", flush=True)

with open(OUT, "w") as f:
    json.dump(dict(ckpt=CKPT, cfg=CFG, nj=NJ, window=WINDOW, results=results), f, indent=1)
print(f"saved {OUT}")
