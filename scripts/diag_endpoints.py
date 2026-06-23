"""Fast diagnostic: can the trained policy reach the cost endpoints when commanded?
Commands the policy at the true calibration extremes + extrapolated extremes; reports achieved (cost, wait)."""
import os, sys
import numpy as np
import torch as th

from scripts.eval_uniform_command_pf import (
    load_config, create_eval_env, eval_n_jobs,
    load_learner_replay_snapshot,
)
from src.agents.pcn_agent import PCN

ckpt = sys.argv[1]
snap_path = sys.argv[2]
cfg = os.environ.get("DISTRIBUTED_PCN_CONFIG")
config = load_config(cfg)
snap = load_learner_replay_snapshot(snap_path)
# NOTE: snapshot metadata n_jobs is buggy (records 24 for 1024 runs). Use config n_jobs.
n_jobs = int(config.get("param_env", {}).get("n_jobs") or eval_n_jobs(config))
print(f"[diag] snapshot meta n_jobs={snap.get('metadata',{}).get('n_jobs')} -> using config n_jobs={n_jobs}")
env = create_eval_env(config, job_seed=0, n_jobs=n_jobs)
state = th.load(ckpt, map_location="cpu", weights_only=False)
agent = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
            scaling_factor=np.array([1.0, 1.0, 1.0/max(1, n_jobs)], dtype=np.float32),
            learning_rate=1e-3, batch_size=512, hidden_dim=512, project_name="t",
            experiment_name="PCN", log=False, use_enhanced_model=False)
agent.model.load_state_dict(state.get("model_state_dict", state), strict=False)
agent.model.eval()
sc = agent.model.desired_return_scale.detach().cpu().numpy()
print(f"n_jobs={n_jobs} return_scale(wait,cost)={sc.tolist()}")

# calibration endpoints
COST_CL, WAIT_OP, WAIT_CL = 1.83e9, 1.64e6, 1.95e5
def cmd(cost, avg_wait):
    return np.array([-avg_wait*n_jobs, -cost], dtype=np.float32)
tests = [
    ("all-onprem (cost=0)",            cmd(0.0,        WAIT_OP)),
    ("all-onprem x1.5 extrap",         cmd(0.0,        WAIT_OP*1.5)),
    ("all-cloud (cost=max)",           cmd(COST_CL,    WAIT_CL)),
    ("all-cloud x1.5 extrap (cost)",   cmd(COST_CL*1.5, WAIT_CL)),
    ("mid",                            cmd(COST_CL*0.5, (WAIT_OP+WAIT_CL)/2)),
]
print("cmd_label                     -> achieved cost / avg_wait / action1_frac")
for label, dr in tests:
    hz = float(n_jobs)
    out = agent._run_episode(env, dr.copy(), np.float32(hz), max_return=None, eval_mode=True)
    val = out[-1]  # (cost, avgwait)
    # action distribution
    a1 = None
    try:
        trans = out[0]
        acts = [t.action for t in trans]
        a1 = float(np.mean(acts))
    except Exception:
        pass
    print(f"{label:30s} -> cost={float(val[0]):.3g}  wait={float(val[1]):.4g}  a1={a1}")
