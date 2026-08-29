#!/usr/bin/env python3
"""PCN の「どのジョブをクラウドへ送ったか」を多族(family_sweep)と突き合わせる。

多族掃引では burst族(混雑時間帯のジョブから逃がす)が単独で圧勝し、big族(大きい順)は
pダイヤルより悪かった。PCN の行動がどちらに近いかが分かれば、次の改善の直接の手がかりになる。

指令格子ごとに greedy エピソードを1本走らせ、クラウドへ送った集合 S を取り出して:
  - 同じ送信件数 |S| の族集合との Jaccard 係数(1に近いほどその族と同じジョブを選んでいる)
  - S に入ったジョブの「到着密集度」「占有量」の平均パーセンタイル
    (全ジョブを0..1に順位正規化。0.5=無選択、1に近い=その属性が大きいものを選んでいる)
を出す。条件付けフラグは呼び出し側(eval_jscale_c3.sh と同じ env)で与えること。

usage:
  CKPT=... CFG=... NJ=20000 NCMD=12 OUT=results/eval_pf/pcn_actions.json \
  PYTHONPATH=. .venv/bin/python scripts/analyze_pcn_actions.py
"""
import json
import os

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
if os.environ.get("OBS_URGENCY", "1") == "1":
    os.environ["SCHEDULER_OBS_URGENCY"] = "1"
# SCHEDULER_OBS_OCCUPANCY は明示的に "0"/"1" を書き込む(未設定のままにしない)。
# 下で import する scripts.family_sweep がモジュール読み込み時に
# os.environ.setdefault("SCHEDULER_OBS_OCCUPANCY", "1") を実行するため、ここで
# setdefault のまま(未設定)にしておくと import 後に黙って occupancy 観測が1次元
# 混入し、学習時(run_j20000_c3.sh は SCHEDULER_OBS_OCCUPANCY を設定しない=OFF)と
# 観測次元がずれて checkpoint load が shape mismatch になる(224→225)。
os.environ["SCHEDULER_OBS_OCCUPANCY"] = "1" if os.environ.get("OBS_OCCUPANCY", "0") == "1" else "0"

import numpy as np
import torch as th

from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import objectives_to_command
from scripts.family_sweep import build_order

CKPT = os.environ["CKPT"]
CFG = os.environ["CFG"]
NJ = int(os.environ.get("NJ", "20000"))
NCMD = int(os.environ.get("NCMD", "12"))
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "results/eval_pf/pcn_actions.json")
FAMS = os.environ.get("FAMS", "burst,big,small,short,long,fcfs,random").split(",")
DEVICE = os.environ.get("DEVICE", "cuda" if th.cuda.is_available() else "cpu")


def main():
    cfg = load_config(CFG)
    cfg["param_job"]["job_trace_n_jobs"] = NJ
    cfg["param_env"]["n_jobs"] = NJ
    env = create_eval_env(cfg, job_seed=SEED, n_jobs=NJ)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64)[:NJ]

    st = th.load(CKPT, map_location="cpu", weights_only=False)
    use_enh = (st.get("model_type", "") == "EnhancedPCNModel")
    ag = PCN(env, device=DEVICE, state_dim=env.observation_space.shape[0],
             scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32),
             learning_rate=1e-3, batch_size=512,
             hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
             project_name="t", experiment_name="PCN", log=False, use_enhanced_model=use_enh)
    tg = ag.network if ag.use_enhanced_model else ag.model
    tg.load_state_dict(st.get("model_state_dict", st), strict=False)
    tg.eval()
    mx = np.full(2, np.inf, dtype=np.float32)

    # 族ごとの並び + 属性の順位(0..1)
    orders = {f: build_order(f, jobs) for f in FAMS}
    occ = jobs[:, 1] * jobs[:, 2]
    s = np.sort(jobs[:, 0])
    lo = np.searchsorted(s, jobs[:, 0] - 1800.0, side="left")
    hi = np.searchsorted(s, jobs[:, 0] + 1800.0, side="right")
    dens = (hi - lo).astype(np.float64)

    def pct(v):
        r = np.argsort(np.argsort(v))
        return r / max(len(v) - 1, 1)

    occ_p, dens_p = pct(occ), pct(dens)

    # pダイヤル掃引で指令格子(cost,wait)を作る(eval_b2_compare と同じ規約)
    rp = []
    for p in np.linspace(0, 1, 20):
        rng = np.random.default_rng(1000)
        env.reset()
        tw = tc = 0.0
        n = 0
        done = False
        k = 0
        while not done and k < NJ + 5:
            r = env.step(1 if rng.random() < p else 0)
            tw += -float(r[1][0])
            tc += -float(r[1][1])
            n += 1 if r[2] else 0
            done = r[-1]
            k += 1
        rp.append([tc, tw / max(1, n)])
    rp = np.array(rp)
    cg = np.linspace(rp[:, 0].min(), rp[:, 0].max(), NCMD)
    wg = np.interp(cg, np.sort(rp[:, 0]), rp[np.argsort(rp[:, 0]), 1])

    rows = []
    for cc, ww in zip(cg, wg):
        dr = objectives_to_command(float(cc), float(ww), NJ).astype(np.float32)
        r = ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=True)
        acts = np.array([t.action for t in r[0]], dtype=np.int64)[:NJ]
        cost, wait = float(r[5][0]), float(r[5][1])
        sel = np.where(acts == 1)[0]
        k = len(sel)
        row = dict(cmd_cost=float(cc), cmd_wait=float(ww), cost=cost, wait=wait,
                   n_cloud=int(k), frac=float(k / NJ))
        if k:
            row["occ_pct"] = float(occ_p[sel].mean())
            row["dens_pct"] = float(dens_p[sel].mean())
            ssel = set(sel.tolist())
            for f, o in orders.items():
                fs = set(o[:k].tolist())
                inter = len(ssel & fs)
                row[f"jac_{f}"] = float(inter / max(len(ssel | fs), 1))
        rows.append(row)
        print(f"cmd_cost={cc:.3e} 送信{k:6d}件({k/NJ:5.1%}) cost={cost:.3e} wait={wait:8.2f} "
              + (f"占有順位={row['occ_pct']:.3f} 密集順位={row['dens_pct']:.3f} "
                 + " ".join(f"{f}={row[f'jac_{f}']:.3f}" for f in orders) if k else "(全オンプレ)"),
              flush=True)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(ckpt=CKPT, cfg=CFG, nj=NJ, rows=rows), f, indent=1)
    print(f"saved {OUT}")

    ok = [r for r in rows if r.get("n_cloud", 0) > 0]
    if ok:
        print("\n=== 平均(送信ありの指令のみ) ===")
        for f in orders:
            print(f"  Jaccard {f:<7} = {np.mean([r[f'jac_{f}'] for r in ok]):.3f}")
        print(f"  占有量の平均順位   = {np.mean([r['occ_pct'] for r in ok]):.3f} (0.5=無選択)")
        print(f"  到着密集の平均順位 = {np.mean([r['dens_pct'] for r in ok]):.3f} (0.5=無選択)")


if __name__ == "__main__":
    main()
