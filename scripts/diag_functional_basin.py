#!/usr/bin/env python3
"""崩壊/良好seedが「同じ関数(basin)」に収束するかを置換不変な機能フィンガープリントで検証。
同一の (obs × cost指令) プローブ群に各ネットの P(cloud) 出力を取り、ネット間の機能類似度行列を作る。
良好seed同士が高類似(共通basin)で崩壊seedが離れるか → 重み「方向/構造」レベルの弁別。
生重みcosineは隠れユニット置換対称性で無意味なので使わない（関数で比較）。

usage: PYTHONPATH=. uv run python scripts/diag_functional_basin.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import glob
import numpy as np
import torch as th

# 比較対象（trace128, 同一レシピ・同一instance。W0/W1=cmd-track小摂動で同workload）。
RUNS = [
    ("W0_1", "ctbase128_1", 17, "collapse"), ("W0_2", "ctbase128_2", 94, "good"),
    ("W0_3", "ctbase128_3", 31, "mid"), ("W0_4", "ctbase128_4", 16, "collapse"),
    ("W0_5", "ctbase128_5", 27, "mid"),
    ("W1_1", "ctw10128_1", 72, "good"), ("W1_3", "ctw10128_3", 48, "good"),
    ("W1_4", "ctw10128_4", 13, "collapse"), ("W1_2", "ctw10128_2", 27, "mid"),
    ("W1_5", "ctw10128_5", 38, "mid"),
]
CFG = "config/config.yml"
NJ = 128
KCMD = 8
NOBS = 64


def ckpt_snap(run):
    sub = sorted(glob.glob(f"experiments/distributed_pcn/run_synth128_{run}/2026*"))[-1]
    ck = sorted(glob.glob(sub + "/iteration_100/model_iter_*.pth"))[-1]
    return ck, sub + "/learner_replay_snapshot.pkl.gz"


# arch 検出（全runは同レシピ=221/FILM/Fourier想定だが ckpt から確定）
ck0, snap0 = ckpt_snap(RUNS[0][1])
_sd0 = th.load(ck0, map_location="cpu", weights_only=False)
_sd0 = _sd0.get("model_state_dict", _sd0)
_obs = int(_sd0["s_emb.0.weight"].shape[1])
# obs = 220(base) + occupancy(+1, obs>=221) + urgency(+1, obs==222)
os.environ.update({
    "DISTRIBUTED_PCN_USE_EVENT_OBS": "1",
    "SCHEDULER_OBS_OCCUPANCY": "1" if _obs >= 221 else "0",
    "SCHEDULER_OBS_URGENCY": "1" if _obs >= 222 else "0",
    "PCN_FILM": "1" if any(k.startswith("film_gamma") for k in _sd0) else "0",
    "PCN_FOURIER_CMD": "1" if "fourier_freqs" in _sd0 else "0",
})
if "fourier_freqs" in _sd0:
    os.environ["PCN_FOURIER_BANDS"] = str(int(_sd0["fourier_freqs"].shape[0]))

from scripts.pcn_replay_snapshot import create_eval_env, load_config, load_learner_replay_snapshot  # noqa
from src.agents.pcn_agent import PCN  # noqa
from src.utils.pf_command_eval import objectives_to_command  # noqa

env = create_eval_env(load_config(CFG), job_seed=0, n_jobs=NJ)
obs_dim = env.observation_space.shape[0]

# 共有プローブ obs（1つの snapshot から、全ネットに同じ obs を当てる）
snap = load_learner_replay_snapshot(ckpt_snap(RUNS[1][1])[1])
rng = np.random.default_rng(0)
obs_list = []
eps = [e for e in snap["episodes"] if e]
for ei in rng.choice(len(eps), size=min(NOBS * 2, len(eps)), replace=False):
    ep = eps[int(ei)]
    o = getattr(ep[rng.integers(0, len(ep))], "observation", None)
    if o is not None:
        oa = np.asarray(o, dtype=np.float32).ravel()
        if oa.shape[0] == obs_dim:
            obs_list.append(oa)
    if len(obs_list) >= NOBS:
        break
obs_arr = np.asarray(obs_list[:NOBS], dtype=np.float32)


def signature(ck):
    sd = th.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    ag = PCN(env, device="cpu", state_dim=obs_dim,
             scaling_factor=np.array([1., 1., 1. / NJ], dtype=np.float32),
             learning_rate=1e-3, batch_size=512, hidden_dim=512,
             project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False)
    m = ag.model
    m.load_state_dict(sd, strict=False)
    m.eval()
    scale = m.desired_return_scale.detach().cpu().numpy()
    cs = float(scale[1]); wf = float(scale[0]) / NJ * 0.5
    ot = th.tensor(obs_arr, dtype=th.float32)
    ht = th.full((len(obs_arr), 1), float(NJ), dtype=th.float32)
    sig = []
    with th.no_grad():
        for cc in np.linspace(0, cs, KCMD):
            dr = objectives_to_command(float(cc), wf, NJ).astype(np.float32)
            rt = th.tensor(np.tile(dr, (len(obs_arr), 1)), dtype=th.float32)
            out = m(ot, rt, ht)
            logp = out[0] if isinstance(out, tuple) else out
            sig.append(th.exp(logp)[:, 1].cpu().numpy())  # P(cloud)
    return np.concatenate(sig)  # (KCMD*NOBS,)


sigs, labels, npfs, cats = [], [], [], []
for lab, run, npf, cat in RUNS:
    ck, _ = ckpt_snap(run)
    sigs.append(signature(ck))
    labels.append(lab); npfs.append(npf); cats.append(cat)
S = np.array(sigs)
# 相関行列（機能類似度）
C = np.corrcoef(S)
print("=== 機能類似度 (P_cloud signature correlation) ===")
print("        " + " ".join(f"{l:>6}" for l in labels))
for i, l in enumerate(labels):
    print(f"{l:>6} " + " ".join(f"{C[i,j]:6.2f}" for j in range(len(labels))))

def grpmean(cond_i, cond_j):
    vals = [C[i, j] for i in range(len(labels)) for j in range(len(labels))
            if i < j and cats[i] in cond_i and cats[j] in cond_j]
    return np.mean(vals) if vals else float("nan"), len(vals)

gg = grpmean({"good"}, {"good"}); cc = grpmean({"collapse"}, {"collapse"})
gc = grpmean({"good"}, {"collapse"})
print(f"\n良好-良好 mean corr = {gg[0]:.3f} (n={gg[1]})")
print(f"崩壊-崩壊 mean corr = {cc[0]:.3f} (n={cc[1]})")
print(f"良好-崩壊 mean corr = {gc[0]:.3f} (n={gc[1]})")
print("→ 良好同士が崩壊より高類似なら『良好basin』が存在。差なしなら方向も弁別子でない。")
# n_pf と「他の良好seedへの平均類似度」の相関
good_idx = [i for i, c in enumerate(cats) if c == "good"]
aff = [np.mean([C[i, j] for j in good_idx if j != i]) for i in range(len(labels))]
print(f"\ncorr(n_pf, 良好seed群への機能類似度) = {np.corrcoef(npfs, aff)[0,1]:+.3f}")
