#!/usr/bin/env python3
"""崩壊seed vs 良好seed の機序を実測比較（再学習なし）。
記録済み仮説([[pcn-learning-destroys-conditioning]]): 崩壊=重みノルム膨張→logit飽和→指令無視。
2信号を測る:
  (1) 重みノルム: fc/c_emb/s_emb 各層の L2。
  (2) 指令→P(cloud)応答: snapshot の実観測に対し cost指令を 0→cost_max で掃引し、
      P(cloud) が単調に上がるか(=条件付け生存) / 平坦・逆(=崩壊)。span と corr。
アーキ(FILM/Fourier/occupancy)は ckpt から自動検出。env は PCN 構築の obs_dim 用のみ(forward は実観測)。

usage: CKPT=... SNAP=... NJ=512 TAG=L06_1 PYTHONPATH=. uv run python scripts/diag_collapse_mechanism.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 全CPU強制（forward を直接呼ぶのでデバイス統一）
import numpy as np
import torch as th

CKPT = os.environ["CKPT"]
SNAP = os.environ["SNAP"]
CFG = os.environ.get("CFG", "config/config.yml")
NJ = int(os.environ.get("NJ", "512"))
TAG = os.environ.get("TAG", "x")
KCMD = int(os.environ.get("KCMD", "9"))
NOBS = int(os.environ.get("NOBS", "60"))

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

from scripts.pcn_replay_snapshot import create_eval_env, load_config, load_learner_replay_snapshot  # noqa: E402
from src.agents.pcn_agent import PCN  # noqa: E402
from src.utils.pf_command_eval import objectives_to_command  # noqa: E402

env = create_eval_env(load_config(CFG), job_seed=0, n_jobs=NJ)
obs_dim = env.observation_space.shape[0]
assert obs_dim == _obs_expect, f"obs {obs_dim}!={_obs_expect}"
ag = PCN(env, device="cpu", state_dim=obs_dim,
         scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32),
         learning_rate=1e-3, batch_size=512, hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
         project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False)
model = ag.model
model.load_state_dict(_sd, strict=False)
model.eval()
scale = model.desired_return_scale.detach().cpu().numpy()
center = model.desired_return_center.detach().cpu().numpy()
cost_scale = float(scale[1])

# (1) 重みノルム
norms = {}
for name in ["s_emb.0.weight", "c_emb.0.weight", "fc.0.weight", "fc.2.weight"]:
    if name in _sd:
        norms[name] = float(_sd[name].norm().item())
if _has_film:
    norms["film_gamma.weight"] = float(_sd["film_gamma.weight"].norm().item())
    norms["film_beta.weight"] = float(_sd["film_beta.weight"].norm().item())

# (2) 指令→P(cloud) 応答（snapshot 実観測）
snap = load_learner_replay_snapshot(SNAP)
eps = snap["episodes"]
obs_list = []
rng = np.random.default_rng(0)
# 各エピソードの中間ステップから観測をサンプル（決定が cloud/onprem 両方ありうる領域）
ep_idx = rng.choice(len(eps), size=min(NOBS, len(eps)), replace=False)
for ei in ep_idx:
    ep = eps[int(ei)]
    if not ep:
        continue
    t = ep[rng.integers(0, len(ep))]
    o = getattr(t, "observation", None)
    if o is not None:
        oa = np.asarray(o, dtype=np.float32).ravel()
        if oa.shape[0] == obs_dim:
            obs_list.append(oa)
obs_arr = np.asarray(obs_list[:NOBS], dtype=np.float32)

# cost 指令を 0→cost_max で掃引（wait 指令は中位固定）。各観測で P(cloud)。
cmd_costs = np.linspace(0.0, cost_scale, KCMD)
wait_fix = float(scale[0]) / max(1, NJ) * 0.5  # avg_wait 中位
pcloud = np.zeros((KCMD, len(obs_arr)), dtype=np.float64)
ot = th.tensor(obs_arr, dtype=th.float32)
ht = th.full((len(obs_arr), 1), float(NJ), dtype=th.float32)
with th.no_grad():
    for i, cc in enumerate(cmd_costs):
        dr = objectives_to_command(float(cc), wait_fix, NJ).astype(np.float32)
        rt = th.tensor(np.tile(dr, (len(obs_arr), 1)), dtype=th.float32)
        out = model(ot, rt, ht)
        logp = out[0] if isinstance(out, tuple) else out
        probs = th.exp(logp).cpu().numpy()  # (B,2)
        pcloud[i] = probs[:, 1]

mean_pc = pcloud.mean(axis=1)  # P(cloud) per command level (averaged over states)
span = float(mean_pc[-1] - mean_pc[0])  # 高cost指令 − 低cost指令
# 単調性: command-level と mean P(cloud) の相関
corr = float(np.corrcoef(cmd_costs, mean_pc)[0, 1]) if np.std(mean_pc) > 1e-9 else 0.0

print(f"### {TAG}  arch(film={_has_film},fourier={_has_fourier},obs={obs_dim})  cost_scale={cost_scale:.3g}")
print(f"  [norms] " + "  ".join(f"{k.split('.')[0]}={v:.1f}" for k, v in norms.items()))
print(f"  [cond ] P(cloud): cheapCmd={mean_pc[0]:.3f} -> expensiveCmd={mean_pc[-1]:.3f}  span={span:+.3f}  corr={corr:+.3f}")
print(f"  [curve] " + " ".join(f"{p:.2f}" for p in mean_pc))
import json
os.makedirs("results/diag", exist_ok=True)
json.dump(dict(tag=TAG, norms=norms, mean_pcloud=mean_pc.tolist(), span=span, corr=corr,
               cost_scale=cost_scale, film=_has_film, fourier=_has_fourier),
          open(f"results/diag/mech_{TAG}.json", "w"), indent=1)
