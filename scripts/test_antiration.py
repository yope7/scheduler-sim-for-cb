#!/usr/bin/env python3
"""rationing仮説の eval-only 検証（再学習なし）。
高cost指令で greedy rollout を2通り走らせ達成costを比較:
  (A) 通常: desired_return を各stepで decrement(残予算が減る)
  (B) anti-ration: cost成分の desired_return を target一定に保つ(予算を減らさない)
(B)で達成costが(A)より大きく伸び expensive端に届けば → rationingが飽和の原因 & 手当ての方向が正しい。
アーキ/workload は ckpt から自動(cmd_follow_check と同方式, trace top1024)。

usage: CKPT=.. NJ=256 PYTHONPATH=. uv run python scripts/test_antiration.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch as th

CKPT = os.environ["CKPT"]
NJ = int(os.environ.get("NJ", "256"))
CFG = os.environ.get("CFG", "config/config.yml")
TRACE_PATH = os.environ.get("TRACE_PATH", "job_trace/FY2024/scacctreq_202412_top1024_jobs.csv")

_raw = th.load(CKPT, map_location="cpu", weights_only=False)
_sd = _raw.get("model_state_dict", _raw)
_obs = int(_sd["s_emb.0.weight"].shape[1])
os.environ.update({
    "DISTRIBUTED_PCN_USE_EVENT_OBS": "1",
    "SCHEDULER_OBS_OCCUPANCY": "1" if _obs >= 221 else "0",
    "SCHEDULER_OBS_URGENCY": "1" if _obs >= 222 else "0",
    "PCN_FILM": "1" if any(k.startswith("film_gamma") for k in _sd) else "0",
    "PCN_FOURIER_CMD": "1" if "fourier_freqs" in _sd else "0",
})
if "fourier_freqs" in _sd:
    os.environ["PCN_FOURIER_BANDS"] = str(int(_sd["fourier_freqs"].shape[0]))

import yaml
from scripts.pcn_replay_snapshot import load_config
from src.agents.pcn_agent import PCN
from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.pf_command_eval import objectives_to_command


def build_env():
    cfg = yaml.safe_load(yaml.dump(load_config(CFG)))
    cfg["param_env"]["n_jobs"] = NJ
    cfg.setdefault("param_job", {})
    cfg["param_job"]["job_type"] = 2
    cfg["param_job"]["job_trace_path"] = TRACE_PATH
    cfg["param_job"]["job_trace_n_jobs"] = NJ
    cfg["param_job"]["job_trace_exclude_largest_outlier"] = True
    jg = JobGenerator(0, 2, cfg["param_env"]["n_window"], cfg["param_env"]["n_on_premise_node"],
                      cfg["param_env"]["n_cloud_node"], cfg, NJ, 0.2, 0)
    js = jg.generate_jobs_set()
    return SchedulingEnvEventNative(
        np.inf, cfg["param_env"]["n_window"], cfg["param_env"]["n_on_premise_node"],
        cfg["param_env"]["n_cloud_node"], cfg["param_env"]["n_job_queue_obs"],
        cfg["param_env"]["n_job_queue_bck"], cfg["param_agent"]["weight_wt"],
        cfg["param_agent"]["weight_cost"], cfg["param_env"]["penalty_not_allocate"],
        cfg["param_env"]["penalty_invalid_action"], js, None, flag=0)


env = build_env()
ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / NJ], dtype=np.float32), learning_rate=1e-3,
         batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
         use_enhanced_model=False)
m = ag.model
m.load_state_dict(_sd, strict=False)
m.eval()
scale = m.desired_return_scale.detach().cpu().numpy()
center = m.desired_return_center.detach().cpu().numpy()
cost_scale = float(scale[1])


def rollout(dr0, hold_cost):
    """greedy rollout。hold_cost=True なら cost成分(dr[1])を毎step target一定に保つ。"""
    obs = env.reset(); done = False
    dr = np.array(dr0, dtype=np.float32); dh = np.float32(NJ)
    dr_cost_target = float(dr0[1])
    n_giant_cloud = 0; n_giant = 0
    while not done:
        po = ag._obs_for_policy(env, obs)
        with th.no_grad():
            out = m(th.tensor(po[None], dtype=th.float32),
                    th.tensor(dr[None], dtype=th.float32),
                    th.tensor([[dh]], dtype=th.float32))
            logp = out[0] if isinstance(out, tuple) else out
            a = int(np.argmax(th.exp(logp)[0].numpy()))
        n_obs, reward, scheduled, wt, done = env.step(a)
        dr = dr - np.array(reward, dtype=np.float32)
        if hold_cost:
            dr[1] = dr_cost_target  # cost予算を減らさない(anti-ration)
        if scheduled:
            dh = np.float32(max(dh - 1, 1.0))
        obs = n_obs
    cost, _, wait = env.calc_objective_values()
    return float(cost), float(wait)


# 最大cost指令(全クラウド端): 高cost / 低wait を狙う
target_cost = cost_scale          # ≈ 5.56e8
target_wait = float(scale[0]) / NJ * 0.02
dr0 = objectives_to_command(target_cost, target_wait, NJ).astype(np.float32)
print(f"[setup] cost_scale={cost_scale:.3e}  target_cost(指令)={target_cost:.3e}")
cA, wA = rollout(dr0, hold_cost=False)
cB, wB = rollout(dr0, hold_cost=True)
print(f"(A) 通常decrement   : 達成cost={cA:.3e}  wait={wA:.0f}")
print(f"(B) anti-ration(一定): 達成cost={cB:.3e}  wait={wB:.0f}")
print(f"→ cost改善: {cA:.3e} → {cB:.3e}  (×{cB/max(cA,1):.2f})  {'rationingが原因=手当て方向◎' if cB>cA*1.15 else '差小=rationingでない'}")

if os.environ.get("UNIFORM", "0") == "1":
    # 全uniform指令で A/B の達成PFを比較(高い端が伸びるか, n_pf)
    from src.agents.pcn_agent import get_non_dominated_inds_minimize
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    G = 12
    r1 = np.linspace(0.0, cost_scale * 1.05, G)
    r0 = np.linspace(0.0, -float(scale[0]) * 1.05, G)
    achA, achB = [], []
    for w in r0:
        for c in r1:
            dr = np.array([float(w), float(-c)], dtype=np.float32)
            achA.append(rollout(dr, False)); achB.append(rollout(dr, True))
    achA = np.array(achA); achB = np.array(achB)
    def npf(a):
        nd = get_non_dominated_inds_minimize(a); return len(nd), a[nd] if len(nd) else a
    nA, pfA = npf(achA); nB, pfB = npf(achB)
    print(f"[UNIFORM] 通常 n_pf={nA} maxCost={achA[:,0].max():.2e} | anti-ration n_pf={nB} maxCost={achB[:,0].max():.2e}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(achA[:, 0], achA[:, 1], s=14, c="gray", alpha=0.5, label=f"通常 (n_pf={nA}, max={achA[:,0].max()/1e8:.2f}億)")
    ax.scatter(achB[:, 0], achB[:, 1], s=14, c="red", alpha=0.6, label=f"anti-ration (n_pf={nB}, max={achB[:,0].max()/1e8:.2f}億)")
    pa = pfA[np.argsort(pfA[:, 0])]; pb = pfB[np.argsort(pfB[:, 0])]
    ax.plot(pa[:, 0], pa[:, 1], "-", c="black", lw=1); ax.plot(pb[:, 0], pb[:, 1], "-", c="darkred", lw=1.5)
    ax.set_xlabel("Achieved Cost"); ax.set_ylabel("Achieved Wait"); ax.legend()
    ax.set_title(os.environ.get("TITLE", "anti-ration uniform PF"))
    out = os.environ.get("OUT", "docs/figures/antiration_uniform.png")
    fig.tight_layout(); fig.savefig(out, dpi=95); print("[SAVED]", out)
