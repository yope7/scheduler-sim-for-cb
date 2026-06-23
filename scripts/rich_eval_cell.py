#!/usr/bin/env python3
"""1セルのckptをリッチに再評価し HV/Spacing/cmd距離 を厳密算出(6点フロント問題の解消)。
uniform_command_eval_pf(多数指令格子)で達成PFを得て、論文式18 Spacing と HV を計算。
アーキ(FILM/Fourier/occupancy)は ckpt から自動検出、workload は trace256。
出力: 1行 JSON を stdout。
usage: CKPT=.. PYTHONPATH=. uv run python scripts/rich_eval_cell.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 高速env(配置探索の超線形コスト対策, ビット一致)。BLASも単一スレッド化(並列時のover-subscription回避)。
os.environ.setdefault("PCN_FAST_ENV", "1")
os.environ.setdefault("PCN_FAST_ENV_SWEEP", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, glob
import numpy as np
import torch as th
th.set_num_threads(1)

CKPT = os.environ["CKPT"]
NJ = int(os.environ.get("NJ", "256"))
CFG = os.environ.get("CFG", "config/config.yml")
TRACE = os.environ.get("TRACE_PATH", "job_trace/FY2024/scacctreq_202412_top1024_jobs.csv")
JOB_TYPE = int(os.environ.get("JOB_TYPE", "2"))  # 1=合成(JobSimulator) / 2=trace
# HV参照点(nadir)=達成可能最大より安全に大きく(論文 r_m>=max)。全クラウド端 cost≈5.56億・全オンプレ端 wait≈1.54e5
# を厳密に内側に含めるため、それぞれ余裕を持たせる。Spacing正規化のcost_scaleは別(5.56e8固定)。
COST_REF = float(os.environ.get("HV_COST_REF", "6.5e8"))
WAIT_REF = float(os.environ.get("HV_WAIT_REF", "1.7e5"))
COST_SCALE = float(os.environ.get("COST_SCALE", "5.56e8"))  # Spacing正規化スケール(合成は別値)

_raw = th.load(CKPT, map_location="cpu", weights_only=False); _sd = _raw.get("model_state_dict", _raw)
_obs = int(_sd["s_emb.0.weight"].shape[1])
# 行動数を出力層から検出: 3なら defer(後回し)込み→ env も3択にする
_nact = int(_sd["fc.2.weight"].shape[0]) if "fc.2.weight" in _sd else 2
if _nact >= 3:
    os.environ["SCHEDULER_ALLOW_DEFER"] = "1"
    os.environ["SCHEDULER_DEFER_OFFSET"] = "1"  # ユーザー設計: deferは1つ後ろのみ
os.environ.update({"DISTRIBUTED_PCN_USE_EVENT_OBS": "1",
                   "SCHEDULER_OBS_OCCUPANCY": "1" if _obs >= 221 else "0",
                   "SCHEDULER_OBS_URGENCY": "1" if _obs >= 222 else "0",
                   "PCN_FILM": "1" if any(k.startswith("film_gamma") for k in _sd) else "0",
                   "PCN_FOURIER_CMD": "1" if "fourier_freqs" in _sd else "0"})
if "fourier_freqs" in _sd:
    os.environ["PCN_FOURIER_BANDS"] = str(int(_sd["fourier_freqs"].shape[0]))

import yaml
from scripts.pcn_replay_snapshot import load_config
from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.pf_eval_gap import _uniform_grid_commands, _run_uniform_grid_commands
from src.utils.pf_command_eval import objectives_to_command


def build_env():
    cfg = yaml.safe_load(yaml.dump(load_config(CFG))); cfg["param_env"]["n_jobs"] = NJ
    cfg.setdefault("param_job", {})
    if JOB_TYPE == 2:
        cfg["param_job"].update(
            {"job_type": 2, "job_trace_path": TRACE, "job_trace_n_jobs": NJ, "job_trace_exclude_largest_outlier": True})
    else:
        cfg["param_job"]["job_type"] = JOB_TYPE   # 合成(job_type=1): seed=0でJobGeneratorが固定生成(学習と同一)
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
nd = get_non_dominated_inds_minimize(ach); pf = ach[nd] if len(nd) else ach


def hv2d(f, ref=(COST_REF, WAIT_REF)):
    f = np.asarray(f, float); f = f[(f[:, 0] < ref[0]) & (f[:, 1] < ref[1])]
    if not len(f): return 0.0
    f = f[np.argsort(-f[:, 0])]; hv = 0.0; prev = ref[0]
    for c, w in f.tolist(): hv += (prev - c) * (ref[1] - w); prev = c
    return hv / (ref[0] * ref[1])  # 参照ボックス面積で正規化[0,1]


def spacing(f):
    f = np.unique(np.asarray(f, float), axis=0)
    if len(f) < 3: return None
    fn = np.column_stack([f[:, 0] / COST_SCALE, f[:, 1] / WAIT_REF])
    d = [min(abs(fn[i, 0] - fn[j, 0]) + abs(fn[i, 1] - fn[j, 1]) for j in range(len(fn)) if j != i) for i in range(len(fn))]
    d = np.array(d); return float(np.sqrt(np.sum((d - d.mean()) ** 2) / (len(d) - 1)))


# cmd距離(指令→達成 正規化MSE, cost成分): 両側(超過も不足も罰する)=正しい追従指標
scale = ag.model.desired_return_scale.detach().cpu().numpy(); cs = float(scale[1])
cmd_cost = np.array([-c[0][1] for c in commands])
diff = (ach[:, 0] - cmd_cost) / cs           # 両側(符号付き)
cmd_dist = float((diff ** 2).mean())          # 両側MSE=追従距離
cmd_over = float((np.maximum(diff, 0) ** 2).mean())   # 参考: 超過側のみ
print(json.dumps({"n_pf": int(len(pf)), "hv": round(hv2d(pf), 4),
                  "spacing": (round(spacing(pf), 4) if spacing(pf) is not None else None),
                  "cmd_dist": round(cmd_dist, 4), "cmd_over": round(cmd_over, 4),
                  "maxcost_oku": round(float(ach[:, 0].max() / 1e8), 2),
                  "hv_ref": [COST_REF, WAIT_REF],
                  "pf": [[round(float(p[0]), 1), round(float(p[1]), 2)] for p in pf.tolist()]}))
