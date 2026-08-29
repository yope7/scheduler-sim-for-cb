#!/usr/bin/env python3
"""rationing 機序の実測: 高cost命令のとき、エピソード中に「残りcost予算」が減り
cloud投入率が落ちるか(=rationingの署名)を step ごとに追う。decrement(既定) vs hold を比較。

_run_episode(pcn_agent:4174) を再現し、per-step で
  - 残り desired_return[1] (cost予算)
  - action (0/1/2, cloud率)
を記録。hold=True なら cost予算を初期値に固定(=PCN_COST_HOLD相当)。

使い方(学習一致フラグ必須):
  PCN_FILM=1 PCN_FOURIER_CMD=1 PCN_FOURIER_BANDS=4 SCHEDULER_OBS_OCCUPANCY=1 OBS_OCCUPANCY=1 \
  SCHEDULER_ALLOW_DEFER=1 SCHEDULER_DEFER_OFFSET=1 DISTRIBUTED_PCN_USE_EVENT_OBS=1 PYTHONPATH=. \
  uv run python scripts/diag_rationing.py --run-dir <exec> --config <cfg> --out docs/figures/rationing.png
"""
from __future__ import annotations
import argparse, os, glob
from pathlib import Path
import numpy as np
import torch as th

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
from src.agents.pcn_agent import PCN
from scripts.pcn_replay_snapshot import (
    load_learner_replay_snapshot, create_eval_env, load_config, eval_n_jobs,
    archive_pf_from_snapshot,
)


def build_agent(run_dir, config_path):
    ck = sorted(glob.glob(run_dir + "/iteration_*/model_iter_*.pth"))[-1]
    snap_path = run_dir + "/learner_replay_snapshot.pkl.gz"
    snap = load_learner_replay_snapshot(snap_path)
    n_jobs = int(snap.get("metadata", {}).get("n_jobs", eval_n_jobs(load_config(config_path))))
    config = load_config(config_path)
    env = create_eval_env(config, job_seed=0, n_jobs=n_jobs)
    state = th.load(ck, map_location="cpu", weights_only=False)
    mt = state.get("model_type", "DiscreteActionsDefaultModel")
    h = 1.0 / max(1, n_jobs)
    agent = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
                scaling_factor=np.array([1.0, 1.0, h], dtype=np.float32), learning_rate=1e-3,
                batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False,
                use_enhanced_model=(mt == "EnhancedPCNModel"))
    tgt = agent.network if agent.use_enhanced_model else agent.model
    tgt.load_state_dict(state.get("model_state_dict", state), strict=False)
    tgt.eval()
    apf = archive_pf_from_snapshot(snap, n_jobs)
    cost_max = float(apf[:, 0].max()) if apf.size else 9e8
    wait_max = float(apf[:, 1].max()) if apf.size else 1.5e5
    return agent, env, n_jobs, cost_max, wait_max, Path(ck).parent.name


def trace_episode(agent, env, dr0, hz0, hold):
    """_run_episode を再現し per-step (残りcost予算dr[1], そのstepで発生したcost=-reward[1]) を返す。"""
    dr = np.asarray(dr0, dtype=np.float32).copy()
    hz = np.float32(hz0)
    cost_hold_target = float(dr[1]) if hold else None
    obs = env.reset()
    done = False
    budget, cost_step = [], []
    while not done:
        pobs = agent._obs_for_policy(env, obs)
        action = agent._act(pobs, dr, hz, True)
        n_obs, reward, scheduled, wt, done = env.step(int(action))
        budget.append(float(dr[1]))            # そのstep開始時の残りcost予算(負, 0に近い=枯渇)
        cost_step.append(-float(np.asarray(reward, dtype=np.float32)[1]))  # そのstepで発生したcost
        obs = n_obs
        dr = (dr - np.asarray(reward, dtype=np.float32)).astype(np.float32)
        if hold:
            dr[1] = cost_hold_target
        if scheduled:
            hz = np.float32(max(hz - 1, 1.0))
    return np.array(budget), np.array(cost_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="docs/figures/rationing.png")
    args = ap.parse_args()
    agent, env, nj, cost_max, wait_max, tag = build_agent(args.run_dir, args.config)
    print(f"[diag] model={tag} n_jobs={nj} cost_max={cost_max:.2e}")

    # 高cost命令(cost端 r1=-cost_max, wait も端寄り)
    dr_high = np.array([-wait_max * nj, -cost_max], dtype=np.float32)
    hz = float(nj)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.analyze_pf_retention import setup_jp_font
    setup_jp_font()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for hold, c, lab in [(False, "#dc2626", "decrement(既定=rationing)"), (True, "#16a34a", "hold(COST_HOLD)")]:
        b, cs = trace_episode(agent, env, dr_high, hz, hold)
        x = np.arange(len(b))
        cum = np.cumsum(cs)
        a1.plot(x, cum / cost_max, "-", c=c, lw=1.8, label=lab)  # 累積cost(正規化)
        # 残り予算 vs step当たりcost の相関(rationing=予算枯渇でcost投入↓)
        rem = -b  # 残り予算(正=まだ予算)。0に近い=枯渇
        corr = float(np.corrcoef(rem, cs)[0, 1]) if len(cs) > 3 and cs.std() > 0 else float("nan")
        win = max(3, len(cs) // 15)
        roll = np.convolve(cs, np.ones(win) / win, mode="same")
        a2.plot(x, roll / cost_max, "-", c=c, lw=1.8, label=f"{lab} (corr予算-cost={corr:+.2f})")
        print(f"  hold={hold}: 最終累積cost={cum[-1]:.2e} ({cum[-1]/cost_max:.2f}×命令) "
              f"前半cost={cs[:len(cs)//2].sum():.2e} 後半cost={cs[len(cs)//2:].sum():.2e} corr(残予算,cost)={corr:+.2f}")
    a1.axhline(1.0, ls=":", c="gray", alpha=0.6)
    a1.set_xlabel("step(ジョブ順)"); a1.set_ylabel("累積cost(正規化, 1.0=命令通り)")
    a1.set_title("累積cost到達: hold(緑)がdecrement(赤)より高く伸びれば rationing 確認"); a1.legend(fontsize=9); a1.grid(alpha=0.3)
    a2.set_xlabel("step(ジョブ順)"); a2.set_ylabel("step当たりcost(移動平均, 正規化)")
    a2.set_title("cost投入の推移: decrement(赤)が後半に落ちる=rationing"); a2.legend(fontsize=9); a2.grid(alpha=0.3)
    fig.suptitle(f"rationing機序の実測 ({tag}, 高cost命令)", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[diag] -> {args.out}")


if __name__ == "__main__":
    main()
