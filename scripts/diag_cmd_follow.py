#!/usr/bin/env python3
"""指令cost→達成cost の追従曲線を出す診断。rich_eval_cell と同じ greedy uniform-command eval で、
各指令の (commanded_cost, achieved_cost, achieved_wait) を取り出して JSON 保存。
y=x 対角に乗れば追従、上に外れれば「指令costを超えるオーバーシュート(崩壊)」。
usage: CKPT=.. NJ=256 CFG=experiments/distributed_pcn/job_trace_256_pcn.yml JOB_TYPE=2 \
       OUT=/tmp/cf_base2.json PYTHONPATH=. uv run python scripts/diag_cmd_follow.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PCN_FAST_ENV", "1"); os.environ.setdefault("PCN_FAST_ENV_SWEEP", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np
import torch as th
th.set_num_threads(1)

CKPT = os.environ["CKPT"]; NJ = int(os.environ.get("NJ", "256"))
CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_256_pcn.yml")
TRACE = os.environ.get("TRACE_PATH", "job_trace/FY2024/scacctreq_202412_top1024_jobs.csv")
JOB_TYPE = int(os.environ.get("JOB_TYPE", "2"))
OUT = os.environ.get("OUT", "/tmp/cf.json")

_raw = th.load(CKPT, map_location="cpu", weights_only=False); _sd = _raw.get("model_state_dict", _raw)
_obs = int(_sd["s_emb.0.weight"].shape[1])
_nact = int(_sd["fc.2.weight"].shape[0]) if "fc.2.weight" in _sd else 2
if _nact >= 3:
    os.environ["SCHEDULER_ALLOW_DEFER"] = "1"; os.environ["SCHEDULER_DEFER_OFFSET"] = "1"
os.environ.update({"DISTRIBUTED_PCN_USE_EVENT_OBS": "1",
                   "SCHEDULER_OBS_OCCUPANCY": "1" if _obs >= 221 else "0",
                   "SCHEDULER_OBS_URGENCY": "1" if _obs >= 222 else "0",
                   "PCN_FILM": "1" if any(k.startswith("film_gamma") for k in _sd) else "0",
                   "PCN_FOURIER_CMD": "1" if "fourier_freqs" in _sd else "0"})
if "fourier_freqs" in _sd:
    os.environ["PCN_FOURIER_BANDS"] = str(int(_sd["fourier_freqs"].shape[0]))

import yaml
from scripts.pcn_replay_snapshot import load_config
from src.agents.pcn_agent import PCN
from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.pf_eval_gap import _uniform_grid_commands, _run_uniform_grid_commands


def build_env():
    cfg = yaml.safe_load(yaml.dump(load_config(CFG))); cfg["param_env"]["n_jobs"] = NJ
    cfg.setdefault("param_job", {})
    if JOB_TYPE == 2:
        cfg["param_job"].update({"job_type": 2, "job_trace_path": TRACE, "job_trace_n_jobs": NJ,
                                 "job_trace_exclude_largest_outlier": True})
    else:
        cfg["param_job"]["job_type"] = JOB_TYPE
    jg = JobGenerator(0, JOB_TYPE, cfg["param_env"]["n_window"], cfg["param_env"]["n_on_premise_node"],
                      cfg["param_env"]["n_cloud_node"], cfg, NJ, 0.2, 0)
    js = jg.generate_jobs_set()
    return SchedulingEnvEventNative(np.inf, cfg["param_env"]["n_window"], cfg["param_env"]["n_on_premise_node"],
        cfg["param_env"]["n_cloud_node"], cfg["param_env"]["n_job_queue_obs"], cfg["param_env"]["n_job_queue_bck"],
        cfg["param_agent"]["weight_wt"], cfg["param_agent"]["weight_cost"], cfg["param_env"]["penalty_not_allocate"],
        cfg["param_env"]["penalty_invalid_action"], js, None, flag=0)


env = build_env()
ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / NJ], dtype=np.float32), learning_rate=1e-3, batch_size=512,
         hidden_dim=512, project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False)
ag.model.load_state_dict(_sd, strict=False); ag.model.eval()

commands, ref_pts, expl, _ = _uniform_grid_commands(ag, NJ, 12, 1.10)
ach = _run_uniform_grid_commands(ag, env, commands, actors=None)
cmd_cost = np.array([-c[0][1] for c in commands])     # 指令cost(正の大きさ)
ach_cost = ach[:, 0]; ach_wait = ach[:, 1]
order = np.argsort(cmd_cost)
out = {"ckpt": CKPT,
       "cmd_cost": [round(float(cmd_cost[i]), 1) for i in order],
       "ach_cost": [round(float(ach_cost[i]), 1) for i in order],
       "ach_wait": [round(float(ach_wait[i]), 1) for i in order]}
json.dump(out, open(OUT, "w"))
# 追従指標(参考)
cs = float(ag.model.desired_return_scale.detach().cpu().numpy()[1])
diff = (ach_cost - cmd_cost) / cs
print(f"[diag] {OUT}  cmd_dist={float((diff**2).mean()):.4f}  over={float((np.maximum(diff,0)**2).mean()):.4f}")
