#!/usr/bin/env python3
"""崩壊カーブ測定: run の各 checkpoint で「方策がコマンドに従うか」を測る。
   F = corr(commanded_cost, achieved_cost)、span = 到達cost範囲/指令cost範囲。
   高い=コマンド追従(健全)、低い=command無視(崩壊)。n_pf/acc は使わない。
   usage: EXEC=<run exec dir> CFG=<config.yml> NJOBS=24 OBS_URGENCY=1 OUT=collapse.png python scripts/collapse_curve.py"""
import os, glob, re
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
if os.environ.get("OBS_URGENCY", "1") == "1":
    os.environ["SCHEDULER_OBS_URGENCY"] = "1"
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.pcn_replay_snapshot import (create_eval_env, load_config,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import dedupe_pf, stratified_sample_pf, objectives_to_command

EXEC = os.environ["EXEC"]
CFG = os.environ["CFG"]
NJ = int(os.environ.get("NJOBS", "24"))
N_CMD = int(os.environ.get("N_CMD", "14"))
OUT = os.environ.get("OUT", "collapse_curve.png")
LABEL = os.environ.get("LABEL", os.path.basename(EXEC.rstrip("/")))

config = load_config(CFG)
snap = load_learner_replay_snapshot(glob.glob(EXEC + "/learner_replay_snapshot.pkl.gz")[0])
arch = dedupe_pf(archive_pf_from_snapshot(snap, NJ))
targets = stratified_sample_pf(arch, N_CMD, rng=np.random.default_rng(0), include_extremes=True)
env = create_eval_env(config, job_seed=0, n_jobs=NJ)
tc = np.array([t[0] for t in targets], dtype=np.float64)  # commanded cost
tw = np.array([t[1] for t in targets], dtype=np.float64)

def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0

def eval_ckpt(ckpt):
    state = th.load(ckpt, map_location="cpu", weights_only=False)
    ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
             scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32), learning_rate=1e-3,
             batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
             use_enhanced_model=(state.get("model_type", "") == "EnhancedPCNModel"))
    tg = ag.network if ag.use_enhanced_model else ag.model
    tg.load_state_dict(state.get("model_state_dict", state), strict=False); tg.eval()
    mx = np.full(2, np.inf, dtype=np.float32); ach = []
    for ct, wt in targets:
        dr = objectives_to_command(float(ct), float(wt), NJ).astype(np.float32)
        r = ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=True)
        ach.append([float(r[5][0]), float(r[5][1])])
    return np.array(ach)

rows = []
for ckpt in sorted(glob.glob(EXEC + "/iteration_*/model_iter_*.pth")):
    it = int(re.search(r'model_iter_(\d+)', ckpt).group(1))
    ach = eval_ckpt(ckpt)
    ac, aw = ach[:, 0], ach[:, 1]
    F = corr(tc, ac)
    span = (ac.max() - ac.min()) / max(1e-9, tc.max() - tc.min())
    nuniq = len(np.unique(np.round(ach, 1), axis=0))
    rows.append((it, F, span, nuniq))
    print(f"iter{it:3d} F_cost={F:+.3f} span={span:.3f} unique={nuniq}/{len(targets)}", flush=True)

a = np.array(rows, dtype=np.float64)
np.savez(OUT.replace(".png", ".npz"), curve=a, targets=np.column_stack([tc, tw]))
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.plot(a[:, 0], a[:, 1], "-o", color="#1a73e8", ms=6, label="F = corr(commanded cost, achieved cost)")
ax.plot(a[:, 0], a[:, 2], "-s", color="#2ca02c", ms=6, label="span = achieved / commanded cost range")
ax.axhline(0.8, color="#888", ls="--", lw=1, label="healthy threshold (0.8)")
ax.set_xlabel("training iteration"); ax.set_ylabel("command-following fidelity"); ax.set_ylim(-0.25, 1.1)
ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
ax.set_title(f"Collapse curve — {LABEL} (n_jobs={NJ})\nhigh = follows the cost command; drop = conditioning collapse")
fig.tight_layout(); fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"saved {OUT}  iters={len(a)}  F[start={a[0,1]:+.2f} end={a[-1,1]:+.2f}]  span[start={a[0,2]:.2f} end={a[-1,2]:.2f}]")
