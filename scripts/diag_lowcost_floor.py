#!/usr/bin/env python3
"""低コスト floor の原因切り分け: cost指令を低く固定し wait指令を振って、達成costが動くか見る。
   wait指令で達成costが下がる → wait軸conditioningは生きている（command-balance問題）。
   不変 → 極値で wait軸conditioningが死んでいる（conditioning強度問題、LOW_BAND系が必要）。
   usage: CKPT=<pth> EXEC=<dir> CFG=<yml> NJ=24 python scripts/diag_lowcost_floor.py"""
import os, glob, re
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
if os.environ.get("OBS_URGENCY", "1") == "1":
    os.environ["SCHEDULER_OBS_URGENCY"] = "1"
import numpy as np, torch as th
from scripts.pcn_replay_snapshot import (create_eval_env, load_config,
    load_learner_replay_snapshot, archive_pf_from_snapshot)
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import dedupe_pf, objectives_to_command

CKPT = os.environ["CKPT"]; CFG = os.environ["CFG"]; NJ = int(os.environ.get("NJ", "24"))
EXEC = os.environ.get("EXEC", re.sub(r"/iteration_\d+/.*$", "", CKPT))
config = load_config(CFG)
snap = load_learner_replay_snapshot(glob.glob(EXEC + "/learner_replay_snapshot.pkl.gz")[0])
arch = dedupe_pf(archive_pf_from_snapshot(snap, NJ))
cost_lo, cost_hi = float(arch[:, 0].min()), float(arch[:, 0].max())
wait_lo, wait_hi = float(arch[:, 1].min()), float(arch[:, 1].max())
print(f"archive: cost[{cost_lo:.0f},{cost_hi:.0f}] wait[{wait_lo:.1f},{wait_hi:.1f}]")

env = create_eval_env(config, job_seed=0, n_jobs=NJ)
state = th.load(CKPT, map_location="cpu", weights_only=False)
ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32), learning_rate=1e-3,
         batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
         use_enhanced_model=(state.get("model_type", "") == "EnhancedPCNModel"))
tg = ag.network if ag.use_enhanced_model else ag.model
tg.load_state_dict(state.get("model_state_dict", state), strict=False); tg.eval()
mx = np.full(2, np.inf, dtype=np.float32)

def run(cc, cw):
    dr = objectives_to_command(float(cc), float(cw), NJ).astype(np.float32)
    r = ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=True)
    return float(r[5][0]), float(r[5][1])

print("\n[A] cost指令=0 固定, wait指令を lo→hi にスイープ（達成costが動くか?）")
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    cw = wait_lo + frac * (wait_hi - wait_lo)
    ac, aw = run(cost_lo, cw)
    print(f"  cmd(cost=0, wait={cw:6.1f}) -> achieved cost={ac:8.0f} wait={aw:6.1f}")

print("\n[B] wait指令=hi 固定, cost指令を 0→max にスイープ（参考: cost軸は動くか?）")
for frac in [0.0, 0.1, 0.25, 0.5, 1.0]:
    cc = cost_lo + frac * (cost_hi - cost_lo)
    ac, aw = run(cc, wait_hi)
    print(f"  cmd(cost={cc:8.0f}, wait={wait_hi:.1f}) -> achieved cost={ac:8.0f} wait={aw:6.1f}")
