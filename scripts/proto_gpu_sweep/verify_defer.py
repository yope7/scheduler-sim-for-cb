"""defer(3行動) JAX 工場 と torch env(event_native, SCHEDULER_ALLOW_DEFER=1) の一致検証。

[A] 同一ジョブ集合(weekA win256)・同一の固定行動列(defer 含む 100種)で、
    達成(cost, wait合計)・エピソード長・per-step 報酬が一致するか(cost/len は厳密、
    wait は rtol 1e-5 だが整数演算なので厳密一致想定)。
[B] ランダム初期化の 3 アクション Fourier モデル(FILM=0/FOURIER=1/bands4/FC_DEPTH2/
    OBS_LOG=1)で greedy rollout の step 行動一致率 >99% を確認。

usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/verify_defer.py
"""
from __future__ import annotations

import os
import sys

# --- モジュールレベルで env フラグを読む import 群より前に、検証構成を固定する ---
os.environ["SCHEDULER_ALLOW_DEFER"] = "1"
os.environ.setdefault("SCHEDULER_DEFER_MAX", "3")
os.environ.setdefault("SCHEDULER_DEFER_OFFSET", "1")
os.environ["SCHEDULER_OBS_URGENCY"] = "1"
os.environ["DISTRIBUTED_PCN_USE_EVENT_OBS"] = "1"
os.environ["PCN_FILM"] = "0"
os.environ["PCN_FOURIER_CMD"] = "1"
os.environ["PCN_FOURIER_BANDS"] = "4"
os.environ["PCN_FC_DEPTH"] = "2"
os.environ["PCN_OBS_LOG"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_MAIN = "/home/noguchi/scheduler-sim-for-cb"
sys.path.insert(0, _MAIN)
sys.path.insert(0, os.path.join(_MAIN, "scripts", "proto_gpu_sweep"))

import numpy as np
import yaml

CFG = os.path.join(_MAIN, "experiments/distributed_pcn/job_trace_weekA_win256_pcn.yml")
DEFER_MAX = int(os.environ["SCHEDULER_DEFER_MAX"])
DEFER_OFFSET = int(os.environ["SCHEDULER_DEFER_OFFSET"])


def make_env():
    from scripts.pcn_replay_snapshot import create_eval_env

    cfg = yaml.safe_load(open(CFG))
    return create_eval_env(cfg, job_seed=0, n_jobs=256)


def torch_rollout_fixed(env, acts):
    """固定行動列で done まで回し (cost, wait合計, 長さ, per-step rewards) を返す。"""
    env.reset()
    rews = []
    i = 0
    done = False
    while not done:
        _, r, sch, wt, done = env.step(int(acts[i]))
        rews.append(np.asarray(r, dtype=np.float64))
        i += 1
    cost, _, avg_wait = env.calc_objective_values()
    return float(cost), float(avg_wait) * env.total_jobs_count, i, np.array(rews)


def part_a():
    print("=== [A] 固定行動列 100 種の達成一致 ===", flush=True)
    env = make_env()
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    T = len(jobs)
    t_scan = T * (1 + DEFER_MAX)
    rng = np.random.default_rng(20260803)
    fa = np.zeros((100, t_scan), dtype=np.int32)
    fa[:30] = rng.integers(0, 3, size=(30, t_scan))                     # 3値一様
    p2 = rng.random((30, t_scan))
    fa[30:60] = np.where(p2 < 0.5, 2, rng.integers(0, 2, size=(30, t_scan)))  # defer 多め
    fa[60:80] = rng.integers(0, 2, size=(20, t_scan))                   # 0/1 のみ
    fa[80:90] = rng.integers(1, 3, size=(10, t_scan))                   # 1/2 のみ
    fa[90:95] = 2                                                       # all-defer(毎回cap踏み)
    fa[95:97] = 0
    fa[97:99] = 1
    fa[99] = np.tile([2, 2, 2, 2, 0, 1], t_scan // 6 + 1)[:t_scan]

    from factory_defer import run_fused_defer

    res = run_fused_defer(
        jobs, fixed_actions=fa, n_on=env.n_on_premise_node, n_cl=env.n_cloud_node,
        n_window=env.n_window, defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    print(f"  JAX fixed rollout wall={res['stats']['wall']:.1f}s overflow={res['overflow']}")

    n_bad = 0
    max_rel_wait = 0.0
    for b in range(100):
        cost_t, wsum_t, L_t, rews_t = torch_rollout_fixed(env, fa[b])
        cost_j = float(res["achieved"][b, 0])
        wsum_j = float(res["achieved"][b, 2]) * T
        L_j = int(res["lengths"][b])
        rews_j = res["rewards"][b, :L_j]
        ok_cost = cost_t == cost_j
        ok_len = L_t == L_j
        ok_wait = np.isclose(wsum_t, wsum_j, rtol=1e-5, atol=1e-6)
        ok_rews = (L_t == L_j) and np.allclose(rews_t, rews_j, rtol=1e-5, atol=1e-6)
        if wsum_t > 0:
            max_rel_wait = max(max_rel_wait, abs(wsum_t - wsum_j) / wsum_t)
        if not (ok_cost and ok_len and ok_wait and ok_rews):
            n_bad += 1
            if n_bad <= 5:
                print(f"  MISMATCH b={b}: cost {cost_t} vs {cost_j} | wait {wsum_t} vs "
                      f"{wsum_j} | len {L_t} vs {L_j} | rews_ok={ok_rews}")
    print(f"  [A] 一致 {100 - n_bad}/100  (max wait rel err = {max_rel_wait:.2e})")
    return n_bad == 0


def part_b():
    print("=== [B] ランダム初期化 3 アクションモデルの greedy 行動一致 ===", flush=True)
    import torch as th

    from src.agents.pcn_agent import DiscreteActionsDefaultModel

    env = make_env()
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    T = len(jobs)

    th.manual_seed(20260803)
    model = DiscreteActionsDefaultModel(
        state_dim=221, action_dim=3, reward_dim=2,
        scaling_factor=np.array([1.0, 1.0, 1.0 / T], dtype=np.float32), hidden_dim=512)
    # 素のランダム初期化は logits が定数項支配で単一行動に倒れ、行動多様性の無い弱い
    # 検証になる。最終層を再初期化した上で、実 obs 1 点での logits が拮抗するよう bias を
    # 調整する(LogSoftmax は shift 不変なので出力差=logits差)。以降は状態・指令の揺らぎで
    # argmax が割れ、0/1/2(defer 含む)が実際に出る方策になる。
    with th.no_grad():
        model.s_emb[0].weight.normal_(0.0, 0.30)   # 状態感度を上げ step 間で行動が割れるように
        model.c_emb[0].weight.normal_(0.0, 0.30)   # 指令感度(指令間でも行動が変わる)
        model.fc[0].weight.normal_(0.0, 0.10)
        model.fc[2].weight.normal_(0.0, 0.30)
        model.fc[2].bias.zero_()
    model.eval()
    obs0 = env.reset()
    with th.no_grad():
        sc0 = model(th.tensor(np.array([obs0])).float(),
                    th.tensor(np.array([[-60000.0, -40000000.0]])).float(),
                    th.tensor([[float(T - 2)]]).float())[0]
        model.fc[2].bias.copy_(-(sc0 - sc0.mean()))

    # 達成レンジの端(all-onprem / all-cloud)から指令グリッドと正規化を作る
    _, wsum_on, _, _ = torch_rollout_fixed(env, np.zeros(T, dtype=np.int64))
    cost_cl, wsum_cl, _, _ = torch_rollout_fixed(env, np.ones(T, dtype=np.int64))
    r_on = np.array([-wsum_on, 0.0])
    r_cl = np.array([-wsum_cl, -cost_cl])
    alphas = np.linspace(0.0, 1.0, 20)
    cmds = [((1 - a) * r_on + a * r_cl).astype(np.float32) for a in alphas]
    center = (r_on + r_cl) / 2.0
    scale = np.maximum(np.abs(r_cl - r_on) / 2.0, 1.0)
    model.set_desired_return_normalization(center.astype(np.float32),
                                           scale.astype(np.float32))

    sd = model.state_dict()
    import jax.numpy as jnp
    params = {
        "s_w": jnp.asarray(sd["s_emb.0.weight"].numpy()), "s_b": jnp.asarray(sd["s_emb.0.bias"].numpy()),
        "c_w": jnp.asarray(sd["c_emb.0.weight"].numpy()), "c_b": jnp.asarray(sd["c_emb.0.bias"].numpy()),
        "f0_w": jnp.asarray(sd["fc.0.weight"].numpy()), "f0_b": jnp.asarray(sd["fc.0.bias"].numpy()),
        "f2_w": jnp.asarray(sd["fc.2.weight"].numpy()), "f2_b": jnp.asarray(sd["fc.2.bias"].numpy()),
        "center": jnp.asarray(sd["desired_return_center"].numpy()),
        "scale": jnp.asarray(sd["desired_return_scale"].numpy()),
        "sf": jnp.asarray(sd["scaling_factor"].numpy()),
    }

    # torch greedy rollout（CPU Actor / eval と同じ dr/hz 更新則）
    hz0 = float(max(1.0, T - 2))
    torch_actions = []
    torch_achv = []
    for dr_cmd in cmds:
        obs = env.reset()
        dr = np.array(dr_cmd, dtype=np.float32)
        hz = np.float32(hz0)
        acts = []
        done = False
        while not done:
            with th.no_grad():
                sc = model(th.tensor(np.array([obs])).float(),
                           th.tensor(np.array([dr])).float(),
                           th.tensor([[hz]]).float())
            a = int(th.argmax(sc[0]).item())
            obs, r, sch, wt, done = env.step(a)
            acts.append(a)
            dr = dr - np.asarray(r, dtype=np.float32)
            if sch:
                hz = np.float32(max(hz - 1, 1.0))
        cost, _, avg_wait = env.calc_objective_values()
        torch_actions.append(acts)
        torch_achv.append((float(cost), float(avg_wait)))

    from factory_defer import run_fused_defer

    res = run_fused_defer(
        jobs, [(c, hz0) for c in cmds], params=params, greedy=True,
        n_on=env.n_on_premise_node, n_cl=env.n_cloud_node, n_window=env.n_window,
        defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    print(f"  JAX greedy rollout wall={res['stats']['wall']:.1f}s overflow={res['overflow']}")

    tot = 0
    match = 0
    n_exact = 0
    for k in range(len(cmds)):
        at = np.array(torch_actions[k])
        L_j = int(res["lengths"][k])
        aj = res["actions"][k, :L_j].astype(np.int64)
        L = min(len(at), L_j)
        m = int((at[:L] == aj[:L]).sum())
        tot += max(len(at), L_j)
        match += m
        cj, wj = float(res["achieved"][k, 0]), float(res["achieved"][k, 2])
        ct, wt_ = torch_achv[k]
        exact = (len(at) == L_j) and (m == L) and cj == ct
        n_exact += int(exact)
        uq, ct_a = np.unique(at, return_counts=True)
        dist = "/".join(f"{int(u)}:{int(c)}" for u, c in zip(uq, ct_a))
        print(f"  cmd{k:02d}: len {len(at)}/{L_j} act一致 {m}/{L} 行動分布[{dist}] "
              f"cost {ct:.0f}/{cj:.0f} wait {wt_:.2f}/{wj:.2f}{' EXACT' if exact else ''}")
    rate = match / max(1, tot)
    print(f"  [B] greedy 行動一致率 = {rate * 100:.3f}%  (完全一致エピソード {n_exact}/{len(cmds)})")
    return rate > 0.99


if __name__ == "__main__":
    ok_a = part_a()
    ok_b = part_b()
    print(f"\nRESULT: A={'PASS' if ok_a else 'FAIL'} B={'PASS' if ok_b else 'FAIL'}")
    sys.exit(0 if (ok_a and ok_b) else 1)
