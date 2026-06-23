#!/usr/bin/env python3
"""素のPCN(お手本なし)の内部を1ステップずつ全部出力する診断。結論を出さず生データを並べる。

各指令(cost,wait)について greedy エピソードを回し、各ステップで記録:
  - desired_return(raw 両成分 / 正規化後) ＝ 残り予算がどう減るか
  - P(cloud)/P(onprem) ＝ NN softmax 出力(=方策が指令に応答しているか)
  - 選んだ行動・scheduled・現ジョブの占有量パーセンタイル(巨大ジョブか)
  - 報酬(raw, 符号確認用)
最後に 達成(cost,wait) vs 指令ターゲット、NSGA真PFとの待ち比。

usage:
  CKPT=... CFG=experiments/distributed_pcn/job_trace_256_pcn.yml NJ=256 \
  PYTHONPATH=. .venv/bin/python scripts/diagnose_pcn_internal.py
出力: results/diag/pcn_internal_<tag>.json (全生データ) + コンソール要約
"""
import os, sys, json
for _a in sys.argv[1:]:
    if "=" in _a:
        k, v = _a.split("=", 1); os.environ[k] = v
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")  # dens3v2 は urgency obs (221次元)
# Fourier/FILM はモジュールimport時にグローバル読み込みされるので import より前に設定する
for _v, _d in (("PCN_FILM", "1"), ("PCN_FOURIER_CMD", "1"), ("PCN_FOURIER_BANDS", "4")):
    os.environ.setdefault(_v, _d)

import numpy as np, torch as th
import torch.nn.functional as F
from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import PCN, get_non_dominated_inds_minimize
from src.utils.pf_command_eval import objectives_to_command

CKPT = os.environ["CKPT"]
CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_256_pcn.yml")
NJ = int(os.environ.get("NJ", "256"))
TAG = os.environ.get("TAG", "dens3v2")
os.makedirs("results/diag", exist_ok=True)

cfg = load_config(CFG)
env = create_eval_env(cfg, job_seed=0, n_jobs=NJ)
state = th.load(CKPT, map_location="cpu", weights_only=False)
ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
         scaling_factor=np.array([1., 1., 1. / NJ], dtype=np.float32),
         learning_rate=1e-3, batch_size=2048, hidden_dim=512,
         project_name="t", experiment_name="diag", log=False, use_enhanced_model=False)
ag.model.load_state_dict(state["model_state_dict"], strict=False)
ag.model = ag.model.cpu()  # ckpt の buffer が cuda の場合があるので明示的に cpu へ
ag.return_norm_center = ag.model.desired_return_center.detach().cpu().numpy().copy()
ag.return_norm_scale = ag.model.desired_return_scale.detach().cpu().numpy().copy()
print(f"[diag] ckpt={CKPT}")
print(f"[diag] return_norm center={ag.return_norm_center} scale={ag.return_norm_scale}")

# 占有量順位(巨大ジョブ判定)
jobs = np.asarray([env._to_queue_job(env.jobs[i]) if False else env.jobs[i] for i in range(len(env.jobs))]) if hasattr(env, "jobs") else None

def occ_rank_of(env, j):
    try:
        jb = env.jobs
        occ_all = np.sort(np.asarray(jb, dtype=np.float64)[:, 1] * np.asarray(jb, dtype=np.float64)[:, 2])
        raw = jb[j]; occ = float(raw[1]) * float(raw[2])
        return float(np.searchsorted(occ_all, occ, side="right")) / len(occ_all)
    except Exception:
        return float("nan")

nsga = np.load(CKPT.rsplit("/run_synth", 1)[0].replace("experiments/distributed_pcn", "results/eval_pf") if False else f"results/eval_pf/nsga2_trace{NJ}_s0.npz", allow_pickle=True)["pf"]
ns = nsga[np.argsort(nsga[:, 0])]

# 指令: 真PFから cheap/knee/mid/expensive/extreme の5点
sel = [int(x) for x in np.linspace(0, len(ns) - 1, 5).round()]
labels = ["全オンプレ寄り(端)", "膝", "中域", "高コスト", "全クラウド寄り(端)"]

def run_with_internals(dr0, dh0):
    obs = env.reset(); done = False
    dr = np.array(dr0, dtype=np.float32); dh = np.float32(dh0)
    rows = []
    while not done:
        po = ag._obs_for_policy(env, obs)
        ot = th.tensor(po[None], dtype=th.float32)
        rt = th.tensor(dr[None], dtype=th.float32)
        ht = th.tensor([[dh]], dtype=th.float32)
        with th.no_grad():
            out = ag.model(ot, rt, ht)
            logp = out[0] if isinstance(out, tuple) else out
            probs = th.exp(logp.detach()[0]).numpy()  # [P(onprem), P(cloud)]
        a = int(np.argmax(probs))
        # 正規化後のdr(条件入力)
        drn = (dr - ag.return_norm_center) / ag.return_norm_scale
        j = int(getattr(env, "index_next_job", -1))
        rk = occ_rank_of(env, j) if j >= 0 and j < len(env.jobs) else float("nan")
        n_obs, reward, scheduled, wt, done = env.step(a)
        rows.append(dict(step=len(rows), dr_wait=float(dr[0]), dr_cost=float(dr[1]),
                         drn_wait=float(drn[0]), drn_cost=float(drn[1]),
                         p_onprem=float(probs[0]), p_cloud=float(probs[1]), action=a,
                         scheduled=bool(scheduled), occ_rank=rk,
                         rew_wait=float(reward[0]), rew_cost=float(reward[1])))
        dr = (dr - np.array(reward, dtype=np.float32))
        if scheduled: dh = np.float32(max(dh - 1, 1.0))
        obs = n_obs
    cost, _, wait = env.calc_objective_values()
    return rows, float(cost), float(wait)

result = {"ckpt": CKPT, "norm_center": ag.return_norm_center.tolist(), "norm_scale": ag.return_norm_scale.tolist(), "commands": []}
print(f"\n{'指令':<20}{'目標(cost,wait)':>22}{'達成(cost,wait)':>22}{'cloud率':>8}{'巨大cloud率':>10}")
for k, lab in zip(sel, labels):
    tc, tw = float(ns[k, 0]), float(ns[k, 1])
    dr0 = objectives_to_command(tc, tw, NJ); dh0 = float(NJ)
    rows, ac, aw = run_with_internals(dr0, dh0)
    sched = [r for r in rows if r["scheduled"]]
    cloud_rate = np.mean([r["action"] for r in sched]) if sched else 0
    giants = [r for r in sched if r["occ_rank"] >= 0.9]
    giant_cloud = np.mean([r["action"] for r in giants]) if giants else float("nan")
    print(f"{lab:<18}({tc:>10.0f},{tw:>8.0f}) ({ac:>10.0f},{aw:>8.0f}){cloud_rate:>8.2f}{giant_cloud:>10.2f}")
    result["commands"].append(dict(label=lab, target=[tc, tw], achieved=[ac, aw],
                                   cloud_rate=float(cloud_rate), giant_cloud_rate=float(giant_cloud) if not np.isnan(giant_cloud) else None,
                                   n_giants=len(giants), rows=rows))

out = f"results/diag/pcn_internal_{TAG}.json"
json.dump(result, open(out, "w"))
print(f"\n[diag] 全生データ保存: {out}")
# P(cloud) が dr_cost(残りコスト予算)とどう相関するか(条件応答性)の要約
print("\n=== 条件応答性: 残りコスト予算 vs P(cloud) の相関(指令ごと) ===")
for c in result["commands"]:
    sched = [r for r in c["rows"] if r["scheduled"]]
    if len(sched) > 5:
        drc = np.array([r["drn_cost"] for r in sched]); pc = np.array([r["p_cloud"] for r in sched])
        cor = np.corrcoef(drc, pc)[0, 1] if np.std(drc) > 1e-9 and np.std(pc) > 1e-9 else float("nan")
        print(f"  {c['label']:<18} corr(残りcost正規化, P(cloud))={cor:+.2f}  P(cloud)平均={pc.mean():.2f} std={pc.std():.2f}")
