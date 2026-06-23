#!/usr/bin/env python3
"""指令→達成の追従診断（ロバスト版・自己検証付き）。

設計（[[pcn-eval-scale-mismatch]]の教訓）:
- ckpt のアーキ(FILM/Fourier/occupancy obs)を**重みから自動検出**し env var を構築前に設定。
- workload を JOB_TYPE で synth(1)/trace(2) 切替（trace は TRACE_PATH/TRACE_NJOBS/TRACE_EXCLUDE）。
  → eval env を学習 workload に一致させる（不一致だと指令が範囲外で誤結果）。
- 達成は**権威ある `uniform_command_eval_pf`（学習中図と同じ関数）**で計算。
- EXPECT_NPF を渡すと、保存済み n_pf を**再現できるか自己検証**（不一致なら警告）。
  → 一致して初めて、図(commanded→achieved 対角線)を信頼してよい。

usage 例（trace の qd256, n_pf=18 を再現検証）:
  CKPT=.../model_iter_100.pth CFG=config/config.yml NJ=256 SEED=0 \
  JOB_TYPE=2 TRACE_PATH=job_trace/FY2024/scacctreq_202412_top1024_jobs.csv \
  TRACE_NJOBS=256 TRACE_EXCLUDE=1 EXPECT_NPF=18 \
  OUT=docs/figures/cf_qd256.png TITLE=qd256 PYTHONPATH=. uv run python scripts/cmd_follow_check.py
"""
import os

import numpy as np
import torch as th

CKPT = os.environ["CKPT"]
CFG = os.environ.get("CFG", "config/config.yml")
NJ = int(os.environ.get("NJ", "256"))
SEED = int(os.environ.get("SEED", "0"))
GRID = int(os.environ.get("GRID", "12"))
OUT = os.environ.get("OUT", "cmd_follow.png")
TITLE = os.environ.get("TITLE", "cmd-follow")
EXPECT_NPF = os.environ.get("EXPECT_NPF", "")
JOB_TYPE = int(os.environ.get("JOB_TYPE", "1"))

# --- 1) ckpt のアーキを重みから自動検出し、構築前に env var を設定 -------------
_raw = th.load(CKPT, map_location="cpu", weights_only=False)
_sd = _raw.get("model_state_dict", _raw)
_obs_expect = int(_sd["s_emb.0.weight"].shape[1])
_has_film = any(k.startswith("film_gamma") for k in _sd)
_has_fourier = "fourier_freqs" in _sd
_bands = int(_sd["fourier_freqs"].shape[0]) if _has_fourier else 0

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ["SCHEDULER_OBS_URGENCY"] = "0"
os.environ["SCHEDULER_OBS_OCCUPANCY"] = "1" if _obs_expect == 221 else "0"
os.environ["PCN_FILM"] = "1" if _has_film else "0"
os.environ["PCN_FOURIER_CMD"] = "1" if _has_fourier else "0"
if _has_fourier:
    os.environ["PCN_FOURIER_BANDS"] = str(_bands)
print(
    f"[ARCH] obs_expect={_obs_expect} occ={os.environ['SCHEDULER_OBS_OCCUPANCY']} "
    f"film={_has_film} fourier={_has_fourier} bands={_bands}"
)

# --- 2) env var 設定後に import（FILM/Fourier はモジュール/構築時に読まれる）-----
import yaml  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scripts.pcn_replay_snapshot import load_config  # noqa: E402
from src.agents.pcn_agent import PCN  # noqa: E402
from src.envs.scheduling_variants.event_c_env import SchedulingEnvEventObs  # noqa: E402
from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative  # noqa: E402
from src.utils.job_gen.job_generator import JobGenerator  # noqa: E402
from src.utils.pf_eval_gap import _uniform_grid_commands, _run_uniform_grid_commands  # noqa: E402
from src.agents.pcn_agent import get_non_dominated_inds_minimize  # noqa: E402


def build_env(cfg, job_seed, nj):
    """学習 workload に一致した eval env を構築（synth=1 / trace=2）。"""
    cfg = yaml.safe_load(yaml.dump(cfg))
    cfg["param_env"]["n_jobs"] = nj
    cfg.setdefault("param_job", {})
    cfg["param_job"]["job_type"] = JOB_TYPE
    if JOB_TYPE == 2:
        cfg["param_job"]["job_trace_path"] = os.environ["TRACE_PATH"]
        cfg["param_job"]["job_trace_n_jobs"] = int(os.environ.get("TRACE_NJOBS", str(nj)))
        cfg["param_job"]["job_trace_exclude_largest_outlier"] = (
            os.environ.get("TRACE_EXCLUDE", "0") == "1"
        )
    jg = JobGenerator(
        job_seed, JOB_TYPE, cfg["param_env"]["n_window"],
        cfg["param_env"]["n_on_premise_node"], cfg["param_env"]["n_cloud_node"],
        cfg, nj, 0.2, 0,
    )
    jobs_set = jg.generate_jobs_set()
    use_native = os.environ.get("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1") == "1"
    env_cls = SchedulingEnvEventNative if use_native else SchedulingEnvEventObs
    return env_cls(
        np.inf, cfg["param_env"]["n_window"], cfg["param_env"]["n_on_premise_node"],
        cfg["param_env"]["n_cloud_node"], cfg["param_env"]["n_job_queue_obs"],
        cfg["param_env"]["n_job_queue_bck"], cfg["param_agent"]["weight_wt"],
        cfg["param_agent"]["weight_cost"], cfg["param_env"]["penalty_not_allocate"],
        cfg["param_env"]["penalty_invalid_action"], jobs_set, None, flag=0,
    )


cfg = load_config(CFG)
env = build_env(cfg, SEED, NJ)
obs_dim = env.observation_space.shape[0]
assert obs_dim == _obs_expect, f"obs_dim {obs_dim} != ckpt expect {_obs_expect}"

ag = PCN(
    env, device="cpu", state_dim=obs_dim,
    scaling_factor=np.array([1.0, 1.0, 1.0 / max(1, NJ)], dtype=np.float32),
    learning_rate=1e-3, batch_size=512,
    hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
    project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False,
)
model = ag.model
missing, unexpected = model.load_state_dict(_sd, strict=False)
model.eval()
scale = model.desired_return_scale.detach().cpu().numpy()
print(f"[LOAD] obs_dim={obs_dim} missing={len(missing)} unexpected={len(unexpected)} scale(wait*nj,cost)={scale}")

# --- 3) 権威ある uniform grid 指令で達成を計算（学習中図と同じ経路）-------------
commands, ref_pts, exploration, _r0 = _uniform_grid_commands(ag, NJ, GRID, 1.10)
achieved = _run_uniform_grid_commands(ag, env, commands, actors=None)  # (M,2) cost,wait
cmd_cost = np.array([-c[0][1] for c in commands])  # desired_return=[−wait*nj,−cost] → −[1]=cost
nd = get_non_dominated_inds_minimize(achieved)
eval_pf = achieved[nd] if len(nd) else achieved
n_pf = int(len(eval_pf))
print(f"[NPF] uniform-command eval n_pf={n_pf}  (commands={len(commands)})")

if EXPECT_NPF:
    exp = int(EXPECT_NPF)
    ok = abs(n_pf - exp) <= max(3, int(0.15 * exp))
    print(f"[VERIFY] expected n_pf≈{exp} → {'MATCH ✓ ツール信頼可' if ok else 'MISMATCH ✗ ツール不忠実'}")

# --- 4) 図: 達成雲 + Eval PF + commanded→achieved cost 対角線 -------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
if exploration.size:
    ax.scatter(exploration[:, 0], exploration[:, 1], s=8, c="cyan", alpha=0.3, label=f"Exploration ({len(exploration)})")
if ref_pts.size:
    ax.plot(np.sort(ref_pts[:, 0]), ref_pts[np.argsort(ref_pts[:, 0]), 1], "-", c="green", lw=1.0, alpha=0.5, label=f"Archive PF ({len(ref_pts)})")
ax.scatter(achieved[:, 0], achieved[:, 1], s=18, c="purple", alpha=0.5, label=f"Achieved uniform ({len(achieved)})")
ev = eval_pf[np.argsort(eval_pf[:, 0])]
ax.plot(ev[:, 0], ev[:, 1], "-o", c="red", ms=4, lw=1.2, label=f"Eval PF ({n_pf})")
ax.set_xlabel("Cost")
ax.set_ylabel("Avg Wait")
ax.legend(fontsize=8)
ax.set_title(f"{TITLE}: uniform-command PF")
ax2 = axes[1]
lim = max(float(cmd_cost.max()), float(achieved[:, 0].max()))
ax2.plot([0, lim], [0, lim], "--", c="gray", label="perfect follow")
ax2.scatter(cmd_cost, achieved[:, 0], s=16, c="red", alpha=0.5)
ax2.set_xlabel("Commanded cost")
ax2.set_ylabel("Achieved cost")
ax2.set_title(f"cost following  (n_pf={n_pf})")
ax2.legend()
fig.tight_layout()
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=90)
print("[SAVED]", OUT)
