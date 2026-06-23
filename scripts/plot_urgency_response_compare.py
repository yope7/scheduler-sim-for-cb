#!/usr/bin/env python3
"""早食い解消の直接証拠: 「方策は“混むジョブ(オンプレ予測待ちが長い)”をクラウドに回すか?」を OFF/ON で比較。
   各ステップで先頭ジョブのオンプレ予測待ち(=緊急度, _find_event_allocation の dry-run)と方策の行動(0/1)を収集。
   表示: (1) 生データ点 全ジョブ(薄く, 縦jitter) + (2) 予測待ちでソートした移動平均(滑らかな曲線)。
   予測待ちの分布が二極化(空いてる多数+混む少数)なので、固定ビンより移動平均の方が素直に見える。
   OFF: 右下がり(混むジョブをオンプレ=反緊急=早食い) / ON: 右下がりが緩和(浪費抑制)。
   corr は horizon scaling 非依存 → clobber bug の影響なし。rollout 結果は npz にキャッシュ。
   usage: OFF_CKPT=.. ON_CKPT=.. SNAP=.. CFG=.. [NJ=..] [WIN=81] [CACHE=urgency_resp.npz] OUT=.. PYTHONPATH=. python ...
"""
import os
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
import numpy as np, torch as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.environ.get("OUT", "docs/figures/pf_urgency_response1024.png")
WIN = int(os.environ.get("WIN", "81"))
CACHE = os.environ.get("CACHE", "urgency_resp.npz")


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def moving_avg(pw, act, w=WIN):
    idx = np.argsort(pw); ps = pw[idx].astype(float); a = act[idx].astype(float)
    w = min(w, len(a) if len(a) % 2 == 1 else len(a) - 1); w = max(w, 3)
    k = np.ones(w) / w
    ma = np.convolve(a, k, mode="valid")
    pc = ps[w // 2: w // 2 + len(ma)]
    return pc, ma


if os.path.isfile(CACHE):
    d = np.load(CACHE); pw_off, act_off, pw_on, act_on = d["pw_off"], d["act_off"], d["pw_on"], d["act_on"]
    print("loaded cache", CACHE)
else:
    from scripts.pcn_replay_snapshot import (create_eval_env, load_config,
        load_learner_replay_snapshot, archive_pf_from_snapshot)
    from src.agents.pcn_agent import PCN
    from src.utils.pf_command_eval import dedupe_pf, objectives_to_command
    CFG = os.environ.get("CFG", "experiments/distributed_pcn/job_trace_1024_pcn.yml")
    OFF_CKPT = os.environ["OFF_CKPT"]; ON_CKPT = os.environ["ON_CKPT"]; SNAP = os.environ["SNAP"]
    snap = load_learner_replay_snapshot(SNAP)
    n_jobs = int(os.environ.get("NJ", str(snap.get("metadata", {}).get("n_jobs", 1024))))
    arch = dedupe_pf(archive_pf_from_snapshot(snap, n_jobs)); order = np.argsort(arch[:, 1])
    ct, wt = arch[order[len(order) // 2]]
    config = load_config(CFG)

    def rollout(ckpt, urg):
        os.environ["SCHEDULER_OBS_URGENCY"] = "1" if urg else "0"
        env = create_eval_env(config, job_seed=0, n_jobs=n_jobs)
        st = th.load(ckpt, map_location="cpu", weights_only=False)
        ag = PCN(env, device="cpu", state_dim=env.observation_space.shape[0],
                 scaling_factor=np.array([1., 1., 1. / max(1, n_jobs)], dtype=np.float32), learning_rate=1e-3,
                 batch_size=512, hidden_dim=512, project_name="t", experiment_name="PCN", log=False)
        tg = ag.model; tg.load_state_dict(st.get("model_state_dict", st), strict=False); tg.eval()
        dr = objectives_to_command(float(ct), float(wt), n_jobs).astype(np.float32); hz = np.float32(n_jobs)
        obs = env.reset(); done = False; desired = dr.copy(); horizon = hz; rows = []
        while not done:
            j = int(env.index_next_job)
            if j >= len(env.jobs): break
            raw = env.jobs[j]; job = env._to_queue_job(raw); arr = int(raw[0])
            _, onp = env._find_event_allocation(job, False, arr); pw = max(0, int(onp) - arr)
            pobs = ag._obs_for_policy(env, obs)
            action = int(ag._act(pobs, desired, horizon, eval_mode=True))
            obs, reward, scheduled, wts, done = env.step(action)
            desired = desired - np.array(reward, dtype=np.float32)
            if scheduled: horizon = np.float32(max(float(horizon) - 1, 1.0))
            rows.append((pw, action))
        a = np.array(rows, dtype=np.float64); return a[:, 0], a[:, 1]

    pw_off, act_off = rollout(OFF_CKPT, False)
    pw_on, act_on = rollout(ON_CKPT, True)
    np.savez(CACHE, pw_off=pw_off, act_off=act_off, pw_on=pw_on, act_on=act_on); print("saved cache", CACHE)

c_off, c_on = corr(act_off, pw_off), corr(act_on, pw_on)
rng = np.random.RandomState(0)
fig, ax = plt.subplots(figsize=(9.5, 6.2))
if os.environ.get("XAXIS", "pw") == "rank":
    def _to_rank(x):
        r = np.empty(len(x), float); r[np.argsort(x)] = np.arange(len(x)) / max(1, len(x) - 1) * 100.0; return r
    xoff = _to_rank(pw_off); xon = _to_rank(pw_on)
else:
    xoff = np.maximum(pw_off, 0.3); xon = np.maximum(pw_on, 0.3)
# (1) 生データ点（全ジョブ）薄く + 縦jitter
ax.scatter(xoff, act_off + rng.uniform(-0.04, 0.04, len(act_off)), s=12, c="#d62728", alpha=0.10, edgecolor="none", zorder=1)
ax.scatter(xon, act_on + rng.uniform(-0.04, 0.04, len(act_on)), s=12, c="#1a73e8", alpha=0.10, edgecolor="none", zorder=1)
# (2) 予測待ちでソートした移動平均（滑らかな曲線）
pco, mao = moving_avg(xoff, act_off); pcn, man = moving_avg(xon, act_on)
ax.plot(pco, mao, "-", color="#d62728", lw=3, zorder=4, label=f"urgency OFF  corr={c_off:+.2f}  (anti-urgency = front-loading)")
ax.plot(pcn, man, "--", color="#1a73e8", lw=3, zorder=5, label=f"urgency ON   corr={c_on:+.2f}  (waste suppressed)")
if os.environ.get("XAXIS", "pw") == "rank":
    ax.set_xscale("linear")
    ax.set_xlabel("jobs sorted by congestion (predicted on-prem wait)   0% = idle ... 100% = most congested")
else:
    ax.set_xscale(os.environ.get("XSCALE", "symlog"))
    ax.set_xlabel("predicted on-prem wait of each job  (= urgency)")
ax.set_ylabel("cloud rate  (faint dots = every job;  curve = sliding mean)")
ax.set_title("Does the policy send the CONGESTED jobs to cloud?  (1024 trace, mid target)\n"
             "ideal = up-right (congested→cloud).  OFF=down-right (anti-urgency).  ON=flattened (waste suppressed, not yet up-right)")
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper right"); ax.set_ylim(-0.1, 1.12)
fig.tight_layout(); fig.savefig(OUT, dpi=115, bbox_inches="tight")
print(f"OFF corr={c_off:+.3f}  ON corr={c_on:+.3f}  n={len(pw_off)}  (smoothed win={WIN})")
print("saved", OUT)
