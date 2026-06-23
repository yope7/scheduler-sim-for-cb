#!/usr/bin/env python3
"""早食い(front-loading)が緊急度で解消するかを OFF/ON で可視化（cloud率 vs episode progress）。
   同じ指令を OFF/ON 両方に与え、各時点の「クラウド率(瞬間の移動平均)」を episode progress に対して描く。
   OFF: 序盤に高い→右下がり(序盤でクラウドを使い切る=早食い)。
   ON : 序盤から低く平坦(序盤に集中しない=早食い解消)。
   ※ cumulative の最終値正規化は total_cloud 差で誤解を生むため、瞬間クラウド率を使う。
   usage: OFF_CKPT=.. ON_CKPT=.. ON_SNAP=.. CFG=.. [NJ=24] OUT=.. PYTHONPATH=. PCN_FILM=.. python ...
"""
import os
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import (create_eval_env, load_config, eval_n_jobs,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import dedupe_pf, objectives_to_command

CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_1024_pcn.yml")
OFF_CKPT = os.environ["OFF_CKPT"]; ON_CKPT = os.environ["ON_CKPT"]; ON_SNAP = os.environ["ON_SNAP"]
NJ = int(os.environ.get("NJ", "1024"))
OUT = os.environ.get("OUT", "docs/figures/pf_urgency_timing1024.png")

config = load_config(CFG)
snap = load_learner_replay_snapshot(ON_SNAP)
arch = dedupe_pf(archive_pf_from_snapshot(snap, NJ)); order = np.argsort(arch[:, 1])
picks = {"mid": arch[order[len(order) // 2]]}
max_return = np.full(2, np.inf, dtype=np.float32)


def load_agent(ckpt, env):
    state = th.load(ckpt, map_location="cpu", weights_only=False)
    ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
             scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32), learning_rate=1e-3,
             batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False)
    tg = ag.model; tg.load_state_dict(state.get("model_state_dict", state), strict=False); tg.eval()
    return ag


os.environ["SCHEDULER_OBS_URGENCY"] = "0"; env_off = create_eval_env(config, job_seed=0, n_jobs=NJ); ag_off = load_agent(OFF_CKPT, env_off)
os.environ["SCHEDULER_OBS_URGENCY"] = "1"; env_on = create_eval_env(config, job_seed=0, n_jobs=NJ); ag_on = load_agent(ON_CKPT, env_on)

colors = {"low_wait(high cost)": "#1f77b4", "mid": "#ff7f0e", "high_wait(low cost)": "#2ca02c"}
fig, ax = plt.subplots(figsize=(10, 6.2))


def cloud_curve(ag, env, dr):
    out = ag._run_episode(env, dr, np.float32(NJ), max_return, eval_mode=True)
    acts = np.array([int(t.action) for t in out[0]]); cloud = (acts == 1).astype(float)
    n = len(cloud); third = max(1, n // 3)
    w = max(3, n // 8)
    ma = np.convolve(cloud, np.ones(w) / w, mode="same")  # 瞬間クラウド率(移動平均)
    x = np.linspace(0, 1, n)
    return x, ma, cloud[:third].mean(), cloud[-third:].mean(), cloud.mean()


for lbl, (ct, wt) in picks.items():
    dr = objectives_to_command(float(ct), float(wt), NJ).astype(np.float32)
    x, ma, er, lr, tot = cloud_curve(ag_off, env_off, dr)
    ax.plot(x, ma, "-", color=colors[lbl], lw=2.4, label=f"OFF {lbl}: early={er:.2f} late={lr:.2f}")
    print(f"OFF [{lbl}] early={er:.2f} late={lr:.2f} total={tot:.2f}")
    x, ma, er, lr, tot = cloud_curve(ag_on, env_on, dr)
    ax.plot(x, ma, "--", color=colors[lbl], lw=2.4, label=f"ON  {lbl}: early={er:.2f} late={lr:.2f}")
    print(f"ON  [{lbl}] early={er:.2f} late={lr:.2f} total={tot:.2f}")

ax.axhline(0, color="#cccccc", lw=0.8)
ax.set_xlabel("episode progress (job index / total)")
ax.set_ylabel("cloud rate at that point (sliding mean)")
ax.set_title("Front-loading: cloud rate over the episode (1024-job trace, mid command)\n"
             "OFF (solid) = high early then drops = spends cloud early (greedy)   |   ON (dashed) = flat & low = no early binge")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right"); ax.set_ylim(-0.05, 1.05)
fig.tight_layout(); fig.savefig(OUT, dpi=115, bbox_inches="tight"); print("saved", OUT)
