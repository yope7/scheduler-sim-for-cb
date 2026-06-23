"""Uniform full-range command eval for 1024 runs (forces correct n_jobs; snapshot meta n_jobs is buggy).
Sweeps a grid of (cost,wait) commands incl. the calibration extremes, collects achieved PF, scores n/span/HV.
Usage: python scripts/eval1024_uniform.py <ckpt> <snap> <out_npz> [grid]"""
import os, sys
import numpy as np
import torch as th

from scripts.eval_uniform_command_pf import load_config, create_eval_env, eval_n_jobs, load_learner_replay_snapshot
from src.agents.pcn_agent import PCN
from src.agents.pcn_agent import get_non_dominated_inds_minimize

ckpt, snap_path, out_npz = sys.argv[1], sys.argv[2], sys.argv[3]
grid = int(sys.argv[4]) if len(sys.argv) > 4 else 10
config = load_config(os.environ.get("DISTRIBUTED_PCN_CONFIG"))
n_jobs = int(config.get("param_env", {}).get("n_jobs") or eval_n_jobs(config))
env = create_eval_env(config, job_seed=0, n_jobs=n_jobs)
state = th.load(ckpt, map_location="cpu", weights_only=False)
agent = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
            scaling_factor=np.array([1.0, 1.0, 1.0/max(1, n_jobs)], dtype=np.float32),
            learning_rate=1e-3, batch_size=512, hidden_dim=512, project_name="t",
            experiment_name="PCN", log=False, use_enhanced_model=False)
agent.model.load_state_dict(state.get("model_state_dict", state), strict=False)
agent.model.eval()
sc = agent.model.desired_return_scale.detach().cpu().numpy()
wait_scale, cost_scale = float(sc[0]), float(sc[1])   # total_wait, cost
# command grid spanning [0, scale]*extend in both objectives (probes all directions incl. extremes)
ext = 1.15
costs = np.linspace(0.0, cost_scale*ext, grid)
waits = np.linspace(0.0, wait_scale*ext, grid)   # total wait
achieved = []
for c in costs:
    for w in waits:
        dr = np.array([-w, -c], dtype=np.float32)   # r0=-total_wait, r1=-cost
        out = agent._run_episode(env, dr.copy(), np.float32(n_jobs), max_return=None, eval_mode=True)
        val = out[-1]   # (cost, avg_wait)
        achieved.append([float(val[0]), float(val[1])*n_jobs])  # store [cost, total_wait]
pts = np.asarray(achieved, dtype=np.float64)
nd = get_non_dominated_inds_minimize(pts)
pf = pts[nd]
np.savez(out_npz, achieved=pts, pareto_front=pf)
uniq = np.unique(np.round(pf, 1), axis=0)
print(f"n_jobs={n_jobs} grid={grid} ({grid*grid} cmds)  achieved cost=[{pts[:,0].min():.3g},{pts[:,0].max():.3g}] "
      f"wait=[{pts[:,1].min():.3g},{pts[:,1].max():.3g}]")
print(f"PF: n={len(uniq)} cost_span={pf[:,0].max()-pf[:,0].min():.3g} cost=[{pf[:,0].min():.3g},{pf[:,0].max():.3g}] "
      f"(full cost=[0,{cost_scale:.3g}])  wait=[{pf[:,1].min():.3g},{pf[:,1].max():.3g}]")
print(f"saved {out_npz}")
