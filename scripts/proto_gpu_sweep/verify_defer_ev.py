"""イベント駆動 defer カーネル(factory_defer_ev) と torch env の一致・速度・メモリ検証。

[A] 小規模(weekA win256 cap, 1520/65536): 固定行動列 100 種(defer 含む)で
    torch env ⇔ 新カーネル: 達成 cost/wait・エピソード長・per-step 報酬の一致
    (整数演算なので厳密一致想定)。旧 dense カーネル(factory_defer)とも 3 点照合。
[B] 小規模 greedy: ランダム初期化 3 行動 Fourier モデルで行動一致率 >99%。
[C] 大容量(weekAfull win4096, 22700/90800): OOM なしで固定行動列 30 種の
    torch env ⇔ 新カーネル一致(本丸)。
[D] 速度: 4096×正容量 B=64 の eps/s(sample/bernoulli)と 256cap の新旧比較。

usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      scripts/proto_gpu_sweep/verify_defer_ev.py [a|b|c|d ...]
"""
from __future__ import annotations

import os
import sys
import time

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

CFG_S = os.path.join(_MAIN, "experiments/distributed_pcn/job_trace_weekA_win256_cap_pcn.yml")
CFG_L = os.path.join(_MAIN, "experiments/distributed_pcn/job_trace_weekAfull_win4096_pcn.yml")
DEFER_MAX = int(os.environ["SCHEDULER_DEFER_MAX"])
DEFER_OFFSET = int(os.environ["SCHEDULER_DEFER_OFFSET"])


def make_env(cfg_path, n_jobs):
    from scripts.pcn_replay_snapshot import create_eval_env

    cfg = yaml.safe_load(open(cfg_path))
    return create_eval_env(cfg, job_seed=0, n_jobs=n_jobs)


def torch_rollout_fixed(env, acts):
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


def make_fixed_actions(n_seq, t_scan, seed):
    rng = np.random.default_rng(seed)
    fa = np.zeros((n_seq, t_scan), dtype=np.int32)
    n1 = int(n_seq * 0.3)
    n2 = int(n_seq * 0.6)
    n3 = int(n_seq * 0.8)
    n4 = int(n_seq * 0.9)
    fa[:n1] = rng.integers(0, 3, size=(n1, t_scan))
    p2 = rng.random((n2 - n1, t_scan))
    fa[n1:n2] = np.where(p2 < 0.5, 2, rng.integers(0, 2, size=(n2 - n1, t_scan)))
    fa[n2:n3] = rng.integers(0, 2, size=(n3 - n2, t_scan))
    fa[n3:n4] = rng.integers(1, 3, size=(n4 - n3, t_scan))
    for k in range(n4, n_seq):
        pat = [[2, 2, 2, 2, 0, 1], [0], [1], [2]][k % 4]
        fa[k] = np.tile(pat, t_scan // len(pat) + 1)[:t_scan]
    return fa


def _compare_fixed(env, jobs, res, fa, label):
    T = len(jobs)
    n_bad = 0
    max_rel_wait = 0.0
    for b in range(fa.shape[0]):
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
    print(f"  [{label}] 一致 {fa.shape[0] - n_bad}/{fa.shape[0]}"
          f"  (max wait rel err = {max_rel_wait:.2e})")
    return n_bad == 0


def part_a():
    print("=== [A] 小規模(1520/65536) 固定行動列 100 種: torch ⇔ ev ⇔ dense ===",
          flush=True)
    env = make_env(CFG_S, 256)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    T = len(jobs)
    t_scan = T * (1 + DEFER_MAX)
    fa = make_fixed_actions(100, t_scan, 20260804)

    from factory_defer_ev import run_fused_defer_ev

    res = run_fused_defer_ev(
        jobs, fixed_actions=fa, n_on=env.n_on_premise_node, n_cl=env.n_cloud_node,
        n_window=env.n_window, defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    print(f"  ev fixed rollout wall={res['stats']['wall']:.1f}s "
          f"overflow={res['overflow']} shapes={res['stats']}")
    ok_t = _compare_fixed(env, jobs, res, fa, "A: torch⇔ev")

    # 旧 dense カーネルとの 3 点照合(cost / wait / lengths + per-step 報酬)
    from factory_defer import run_fused_defer

    res_d = run_fused_defer(
        jobs, fixed_actions=fa, n_on=env.n_on_premise_node, n_cl=env.n_cloud_node,
        n_window=env.n_window, defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    ok_d = (
        np.array_equal(res["achieved"][:, 0], res_d["achieved"][:, 0])
        and np.allclose(res["achieved"][:, 2], res_d["achieved"][:, 2], rtol=1e-12)
        and np.array_equal(res["lengths"], res_d["lengths"])
        and np.allclose(res["rewards"], res_d["rewards"], rtol=1e-12, atol=0)
    )
    print(f"  [A] ev⇔dense 3点照合(cost/wait/len + rewards): {'PASS' if ok_d else 'FAIL'}")
    return ok_t and ok_d and not res["overflow"]


def part_b():
    print("=== [B] 小規模 greedy 行動一致(ランダム初期化 3 行動モデル) ===", flush=True)
    import torch as th

    from src.agents.pcn_agent import DiscreteActionsDefaultModel

    env = make_env(CFG_S, 256)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    T = len(jobs)

    th.manual_seed(20260803)
    model = DiscreteActionsDefaultModel(
        state_dim=221, action_dim=3, reward_dim=2,
        scaling_factor=np.array([1.0, 1.0, 1.0 / T], dtype=np.float32), hidden_dim=512)
    with th.no_grad():
        model.s_emb[0].weight.normal_(0.0, 0.30)
        model.c_emb[0].weight.normal_(0.0, 0.30)
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
        "fc_w": (jnp.asarray(sd["fc.0.weight"].numpy()),
                 jnp.asarray(sd["fc.2.weight"].numpy())),
        "fc_b": (jnp.asarray(sd["fc.0.bias"].numpy()),
                 jnp.asarray(sd["fc.2.bias"].numpy())),
        "s_w": jnp.asarray(sd["s_emb.0.weight"].numpy()),
        "s_b": jnp.asarray(sd["s_emb.0.bias"].numpy()),
        "c_w": jnp.asarray(sd["c_emb.0.weight"].numpy()),
        "c_b": jnp.asarray(sd["c_emb.0.bias"].numpy()),
        "center": jnp.asarray(sd["desired_return_center"].numpy()),
        "scale": jnp.asarray(sd["desired_return_scale"].numpy()),
        "sf": jnp.asarray(sd["scaling_factor"].numpy()),
    }
    if "fourier_freqs" in sd:
        params["fourier"] = jnp.asarray(sd["fourier_freqs"].numpy())

    hz0 = float(max(1.0, T - 2))
    torch_actions = []
    torch_achv = []
    torch_traj = []          # (obs, dr, hz) 列 — 教師強制一致率の材料
    for dr_cmd in cmds:
        obs = env.reset()
        dr = np.array(dr_cmd, dtype=np.float32)
        hz = np.float32(hz0)
        acts = []
        obss, drs, hzs = [], [], []
        done = False
        while not done:
            with th.no_grad():
                sc = model(th.tensor(np.array([obs])).float(),
                           th.tensor(np.array([dr])).float(),
                           th.tensor([[hz]]).float())
            a = int(th.argmax(sc[0]).item())
            obss.append(np.asarray(obs, dtype=np.float32).copy())
            drs.append(dr.copy())
            hzs.append(float(hz))
            obs, r, sch, wt, done = env.step(a)
            acts.append(a)
            dr = dr - np.asarray(r, dtype=np.float32)
            if sch:
                hz = np.float32(max(hz - 1, 1.0))
        cost, _, avg_wait = env.calc_objective_values()
        torch_actions.append(acts)
        torch_achv.append((float(cost), float(avg_wait)))
        torch_traj.append((np.stack(obss), np.stack(drs),
                           np.array(hzs, np.float32)))

    from factory_defer_ev import run_fused_defer_ev

    res = run_fused_defer_ev(
        jobs, [(c, hz0) for c in cmds], params=params, greedy=True,
        n_on=env.n_on_premise_node, n_cl=env.n_cloud_node, n_window=env.n_window,
        defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    print(f"  ev greedy rollout wall={res['stats']['wall']:.1f}s overflow={res['overflow']}")

    from factory_fused import pcn_logits
    import jax.numpy as _jnp

    # obs の bit 一致検査: torch の行動列を fixed として ev に再生し、torch が見た obs 列と
    # 比較(方策 forward の fp 差と独立に env/obs の厳密性を確認する)。
    t_scan = T * (1 + DEFER_MAX)
    fa = np.zeros((len(cmds), t_scan), dtype=np.int32)
    for k, acts in enumerate(torch_actions):
        fa[k, :len(acts)] = np.asarray(acts, dtype=np.int32)
    res_fx = run_fused_defer_ev(
        jobs, fixed_actions=fa, n_on=env.n_on_premise_node, n_cl=env.n_cloud_node,
        n_window=env.n_window, defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    obs_bit_ok = True
    obs_maxdiff = 0.0
    for k in range(len(cmds)):
        L = len(torch_actions[k])
        d = np.abs(res_fx["obs"][k, :L] - torch_traj[k][0])
        obs_maxdiff = max(obs_maxdiff, float(d.max()) if d.size else 0.0)
        obs_bit_ok = obs_bit_ok and np.array_equal(res_fx["obs"][k, :L], torch_traj[k][0])
    print(f"  obs bit一致(torch行動列の fixed 再生): {obs_bit_ok} "
          f"(max|diff|={obs_maxdiff:.3e})")

    tot = 0
    match = 0
    n_exact = 0
    tf_tot = 0
    tf_match = 0
    for k in range(len(cmds)):
        at = np.array(torch_actions[k])
        L_j = int(res["lengths"][k])
        aj = res["actions"][k, :L_j].astype(np.int64)
        L = min(len(at), L_j)
        m = int((at[:L] == aj[:L]).sum())
        tot += max(len(at), L_j)
        match += m
        # 教師強制: torch 軌道上の (obs,dr,hz) で jax argmax(自由走行の分岐カスケードを排除
        # した per-step 判定一致。分岐は logit 同点割れ=GEMM 順序 fp 差が原因のため、
        # env/obs の正しさはこちらで判定するのが適切。obs 一致は part A + 下の bit 検査)
        ob_t, dr_t, hz_t = torch_traj[k]
        lg = np.asarray(pcn_logits(params, _jnp.asarray(ob_t), _jnp.asarray(dr_t),
                                   _jnp.asarray(hz_t)))
        tf = int((np.argmax(lg, axis=1) == at).sum())
        tf_match += tf
        tf_tot += len(at)
        cj = float(res["achieved"][k, 0])
        ct, wt_ = torch_achv[k]
        exact = (len(at) == L_j) and (m == L) and cj == ct
        n_exact += int(exact)
        print(f"  cmd{k:02d}: len {len(at)}/{L_j} act一致 {m}/{L} 教師強制 {tf}/{len(at)} "
              f"cost {ct:.0f}/{cj:.0f}{' EXACT' if exact else ''}")
    rate = match / max(1, tot)
    tf_rate = tf_match / max(1, tf_tot)
    print(f"  [B] greedy 行動一致率: 自由走行 = {rate * 100:.3f}% / "
          f"教師強制 = {tf_rate * 100:.3f}%  (完全一致エピソード {n_exact}/{len(cmds)})")
    # 判定は教師強制(分岐後カスケードを除いた per-step 判定一致)。自由走行は参考値
    # (同点割れ 1 回で以降全ステップが不一致扱いになるため)。
    return tf_rate > 0.99


def part_c():
    print("=== [C] 大容量(4096×22700/90800) 固定行動列 30 種: torch ⇔ ev(本丸) ===",
          flush=True)
    env = make_env(CFG_L, 4096)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    T = len(jobs)
    t_scan = T * (1 + DEFER_MAX)
    fa = make_fixed_actions(30, t_scan, 20260805)

    from factory_defer_ev import run_fused_defer_ev

    t0 = time.time()
    res = run_fused_defer_ev(
        jobs, fixed_actions=fa, n_on=env.n_on_premise_node, n_cl=env.n_cloud_node,
        n_window=env.n_window, defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    print(f"  ev fixed rollout(B=30, t_scan={t_scan}) wall={res['stats']['wall']:.1f}s "
          f"overflow={res['overflow']} shapes={res['stats']}", flush=True)
    try:
        import jax
        ms = jax.local_devices()[0].memory_stats()
        print(f"  GPU peak_bytes_in_use = {ms.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB",
              flush=True)
    except Exception:
        pass
    ok = _compare_fixed(env, jobs, res, fa, "C: torch⇔ev @4096")
    print(f"  (torch 30 本 rollout 込み wall={time.time() - t0:.0f}s)")
    return ok and not res["overflow"]


def part_d():
    print("=== [D] 速度: 4096×正容量 と 256cap 新旧比較 ===", flush=True)
    import jax

    from factory_defer import run_fused_defer
    from factory_defer_ev import run_fused_defer_ev

    # ---- 256cap: 旧 dense vs 新 ev (bernoulli B=64, t_scan=T) ----
    env = make_env(CFG_S, 256)
    env.reset()
    jobs_s = np.asarray(env.jobs, dtype=np.float64).copy()
    kw_s = dict(probs=[0.5] * 64, n_on=env.n_on_premise_node,
                n_cl=env.n_cloud_node, n_window=env.n_window,
                defer_max=DEFER_MAX, defer_offset=DEFER_OFFSET)
    for name, fn in (("dense", run_fused_defer), ("ev", run_fused_defer_ev)):
        fn(jobs_s, seed=1, **kw_s)                      # warmup(コンパイル)
        t0 = time.time()
        r = fn(jobs_s, seed=2, **kw_s)
        dt = time.time() - t0
        print(f"  256cap  {name:5s} B=64 bernoulli: {dt:.2f}s = {64 / dt:.1f} eps/s "
              f"(overflow={r.get('overflow')})", flush=True)

    # ---- 4096×正容量: ev のみ(旧 dense は OOM)。bernoulli(t_scan=T) と greedy(4T) ----
    env = make_env(CFG_L, 4096)
    env.reset()
    jobs_l = np.asarray(env.jobs, dtype=np.float64).copy()
    T = len(jobs_l)
    kw_l = dict(n_on=env.n_on_premise_node, n_cl=env.n_cloud_node,
                n_window=env.n_window, defer_max=DEFER_MAX,
                defer_offset=DEFER_OFFSET)
    B = 64
    run_fused_defer_ev(jobs_l, probs=[0.5] * B, seed=1, **kw_l)   # warmup
    t0 = time.time()
    r = run_fused_defer_ev(jobs_l, probs=[0.5] * B, seed=2, **kw_l)
    dt = time.time() - t0
    print(f"  4096cap ev B={B} bernoulli(t_scan={T}): {dt:.1f}s = {B / dt:.2f} eps/s "
          f"(overflow={r['overflow']})", flush=True)
    ms = jax.local_devices()[0].memory_stats()
    print(f"  GPU peak_bytes_in_use = {ms.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB")

    # sample モード(方策付き; ランダム重みで形だけ) t_scan = T*(1+defer_max)
    rng = np.random.default_rng(0)
    import jax.numpy as jnp
    D = 221
    params = {
        "fc_w": (jnp.asarray(rng.normal(0, 0.05, (512, 512)).astype(np.float32)),
                 jnp.asarray(rng.normal(0, 0.05, (3, 512)).astype(np.float32))),
        "fc_b": (jnp.zeros(512, jnp.float32), jnp.zeros(3, jnp.float32)),
        "s_w": jnp.asarray(rng.normal(0, 0.05, (512, D)).astype(np.float32)),
        "s_b": jnp.zeros(512, jnp.float32),
        "c_w": jnp.asarray(rng.normal(0, 0.05, (512, 27)).astype(np.float32)),
        "c_b": jnp.zeros(512, jnp.float32),
        "center": jnp.zeros(2, jnp.float32), "scale": jnp.ones(2, jnp.float32),
        "sf": jnp.asarray(np.array([1, 1, 1.0 / T], np.float32)),
        "fourier": jnp.asarray(np.array([1.0, 2.0, 4.0, 8.0], np.float32)),
    }
    cmds = [(np.array([-1e4, -1e7], np.float32), float(T - 2))] * B
    run_fused_defer_ev(jobs_l, cmds, params=params, seed=1, **kw_l)   # warmup
    t0 = time.time()
    r = run_fused_defer_ev(jobs_l, cmds, params=params, seed=2, **kw_l)
    dt = time.time() - t0
    print(f"  4096cap ev B={B} sample(t_scan={T * (1 + DEFER_MAX)}): {dt:.1f}s = "
          f"{B / dt:.2f} eps/s (overflow={r['overflow']})", flush=True)
    ms = jax.local_devices()[0].memory_stats()
    print(f"  GPU peak_bytes_in_use = {ms.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB")
    return True


if __name__ == "__main__":
    parts = sys.argv[1:] or ["a", "b", "c", "d"]
    results = {}
    for p in parts:
        results[p] = {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[p]()
    print("\nRESULT: " + " ".join(f"{p.upper()}={'PASS' if ok else 'FAIL'}"
                                  for p, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)
