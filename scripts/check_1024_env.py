"""Fast standalone 1024-trace env sanity check (no Ray). Catches data/dim bugs quickly."""
import sys, time
import numpy as np
import yaml

from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.workload_calibration import calibrate_from_config

CFG = "experiments/distributed_pcn/job_trace_1024_pcn.yml"
config = yaml.safe_load(open(CFG))
nj = int(config["param_env"]["n_jobs"])
print(f"[check] n_jobs={nj}")

# 1) calibration (all-OP / all-CL) — exercises the IndexError fix
t0 = time.time()
cal = calibrate_from_config(config)
print(f"[check] calibration OK in {time.time()-t0:.1f}s: "
      f"cost(op/cl)={cal.cost_all_onprem:.0f}/{cal.cost_all_cloud:.0f} "
      f"wait(op/cl)={cal.wait_all_onprem:.1f}/{cal.wait_all_cloud:.1f}")

# 2) build jobs and run a random-action episode — exercises the pt>=1 clamp
pe, pa = config["param_env"], config["param_agent"]
jg = JobGenerator(0, config["param_simulation"]["nb_steps"], pe["n_window"],
                  pe["n_on_premise_node"], pe["n_cloud_node"], config, nj, 0.2, 1)
jobs_set = jg.generate_jobs_set()

def run_ep(label, policy):
    env = SchedulingEnvEventNative(np.inf, pe["n_window"], pe["n_on_premise_node"],
        pe["n_cloud_node"], pe["n_job_queue_obs"], pe["n_job_queue_bck"],
        pa["weight_wt"], pa["weight_cost"], pe["penalty_not_allocate"],
        pe["penalty_invalid_action"], jobs_set, None, 0)
    obs = env.reset()
    obs_dim = np.asarray(obs).shape
    done = False; steps = 0
    rng = np.random.default_rng(0)
    while not done:
        a = policy(rng)
        _, _, _, _, done = env.step(a)
        steps += 1
        if steps > nj * 200:
            print(f"[check] {label}: step guard hit"); break
    env.finalize_window_history(build_maps=False)
    cost, _, avg_wt = env.calc_objective_values()
    return steps, obs_dim, cost, avg_wt

for label, pol in [("all_cloud", lambda r: 1), ("all_onprem", lambda r: 0),
                    ("random", lambda r: int(r.integers(0, 2)))]:
    t0 = time.time()
    steps, obs_dim, cost, wt = run_ep(label, pol)
    print(f"[check] {label}: steps={steps} obs_dim={obs_dim} cost={cost:.0f} wait={wt:.1f} ({time.time()-t0:.1f}s)")

print("[check] ALL OK")
