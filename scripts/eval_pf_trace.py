#!/usr/bin/env python3
"""1つの checkpoint で command を cost 全域に振り、達成 (cost, wait) を「Eval PF」として描く。
   崩壊カーブ(F/span)が要約する中身を、ユーザが見たい PF の形で可視化する。
   commanded target と achieved を線で結び、追従(対角に乗る)か無視(1点に潰れる)かを示す。
   usage: CKPT=<model.pth> CFG=<cfg.yml> NJ=24 NCMD=24 OBS_URGENCY=1 OUT=eval_pf.png python scripts/eval_pf_trace.py"""
import os, glob, re
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
if os.environ.get("OBS_URGENCY", "1") == "1":
    os.environ["SCHEDULER_OBS_URGENCY"] = "1"
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import (create_eval_env, load_config,
    load_learner_replay_snapshot, archive_pf_from_snapshot, episode_objectives_from_snapshot)
from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.utils.pf_command_eval import dedupe_pf, stratified_sample_pf, objectives_to_command

CKPT = os.environ["CKPT"]
CFG = os.environ["CFG"]
NJ = int(os.environ.get("NJ", "24"))
NCMD = int(os.environ.get("NCMD", "24"))
OUT = os.environ.get("OUT", "eval_pf.png")
LABEL = os.environ.get("LABEL", os.path.basename(os.path.dirname(os.path.dirname(CKPT))))
EXEC = os.environ.get("EXEC", "")  # snapshot 親dir（無ければ CKPT から推定）

config = load_config(CFG)
if not EXEC:
    EXEC = re.sub(r"/iteration_\d+/.*$", "", CKPT)
snap = load_learner_replay_snapshot(glob.glob(EXEC + "/learner_replay_snapshot.pkl.gz")[0])
try:
    explo = episode_objectives_from_snapshot(snap, NJ)  # 探索で到達した全エピソードの (cost, wait)
except Exception:
    explo = np.empty((0, 2))
arch = dedupe_pf(archive_pf_from_snapshot(snap, NJ))
nd = get_non_dominated_inds_minimize(arch); pf = arch[nd] if len(nd) else arch
pf = pf[np.argsort(pf[:, 0])]
if os.environ.get("DENSE", "0") == "1":
    # アーカイブPFを cost で等間隔に補間し、密な指令を作る（方策の分解能を引き出す）
    cg = np.linspace(float(pf[:, 0].min()), float(pf[:, 0].max()), NCMD)
    wg = np.interp(cg, pf[:, 0], pf[:, 1])
    targets = list(zip(cg.tolist(), wg.tolist()))
else:
    targets = stratified_sample_pf(arch, NCMD, rng=np.random.default_rng(0), include_extremes=True)

DEVICE = os.environ.get("DEVICE", "cpu")
JOB_SEED = int(os.environ.get("JOB_SEED", "0"))  # 0=学習と同一(in-sample), >0=未知ジョブ(汎化テスト)
env = create_eval_env(config, job_seed=JOB_SEED, n_jobs=NJ)
state = th.load(CKPT, map_location="cpu", weights_only=False)
ag = PCN(env, device=DEVICE, state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32), learning_rate=1e-3,
         batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
         use_enhanced_model=(state.get("model_type", "") == "EnhancedPCNModel"))
tg = ag.network if ag.use_enhanced_model else ag.model
tg.load_state_dict(state.get("model_state_dict", state), strict=False); tg.eval()

mx = np.full(2, np.inf, dtype=np.float32)
cmd, ach = [], []
for ct, wt in targets:
    dr = objectives_to_command(float(ct), float(wt), NJ).astype(np.float32)
    r = ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=True)
    cmd.append([float(ct), float(wt)]); ach.append([float(r[5][0]), float(r[5][1])])
cmd = np.array(cmd); ach = np.array(ach)

# achieved の非支配front
ndA = get_non_dominated_inds_minimize(ach); apf = ach[ndA]; apf = apf[np.argsort(apf[:, 0])]
F = float(np.corrcoef(cmd[:, 0], ach[:, 0])[0, 1]) if cmd[:, 0].std() > 0 and ach[:, 0].std() > 0 else 0.0
span = (ach[:, 0].max() - ach[:, 0].min()) / max(1e-9, cmd[:, 0].max() - cmd[:, 0].min())
np.savez(OUT.replace(".png", ".npz"), commanded=cmd, achieved=ach, archive_pf=pf, explored=explo)

fig, ax = plt.subplots(figsize=(11, 7))
if len(explo):
    ax.scatter(explo[:, 0], explo[:, 1], s=9, c="#9ecae1", alpha=0.35, edgecolor="none",
               label=f"explored during training ({len(explo)})", zorder=0)
ax.plot(pf[:, 0], pf[:, 1], "-D", color="#555555", ms=4, lw=1.3, label=f"discovered PF ({len(pf)})", zorder=2)
for (cc, cw), (ac, aw) in zip(cmd, ach):
    ax.plot([cc, ac], [cw, aw], "-", color="#f0b8b8", lw=0.8, zorder=1)
ax.scatter(cmd[:, 0], cmd[:, 1], s=45, marker="x", c="#1a73e8", label=f"commanded target ({len(cmd)})", zorder=4)
ax.scatter(ach[:, 0], ach[:, 1], s=42, c="#d62728", edgecolor="k", lw=0.4, label="achieved", zorder=5)
ax.plot(apf[:, 0], apf[:, 1], "-", color="#d62728", lw=1.4, alpha=0.6, zorder=3, label=f"achieved PF ({len(apf)})")
ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time"); ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper right")
ax.set_title(f"Eval PF (command sweep) — {LABEL}\nF=corr(cmd,achieved cost)={F:+.2f}  span={span:.2f}  "
             f"(short pink lines = command followed; long = ignored)")
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved {OUT}  F={F:+.3f} span={span:.3f} achieved_cost[{ach[:,0].min():.0f},{ach[:,0].max():.0f}] "
      f"commanded_cost[{cmd[:,0].min():.0f},{cmd[:,0].max():.0f}]")
