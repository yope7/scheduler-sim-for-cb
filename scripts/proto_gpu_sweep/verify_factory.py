"""Phase 3 検証: rollout データ工場（factory_jax）の合否判定。

eval（Phase 2）と基準が違う: 工場データは探索用なのでビット一致は要求しない。
  [0] obs フル GPU 化 vs C 実装の差分定量（タイ順起因の行% を1回計測して報告）
      同一 jobs × 参照行動列で obs 列を JAX 再構築し、Phase 2 検証済みの
      C 経路 obs（refeval npz; ビット一致確認済み）と uint32 比較。
      不一致行が「同一 start のタイ並べ替え」だけかを行集合一致で確認する。
  [a] 形式: アダプタ出力を本体 Learner（Ray 実体）の ReplayBuffer→learn() に流し、
      例外なく 1 update 回ること（use_training_cache=True の実運用経路）。
  [b] 妥当性: 同一重み・同一指令で CPU Actor サンプリング（pcn_agent._run_episode
      eval_mode=False, mx=inf = distributed Actor と同一演算列）N_EPS 本 vs 工場 N_EPS 本の
      達成 (cost, avg_wait) 分布比較（平均・分位点・KS 検定）。
  [c] 決定性: 同一シードで工場を 2 回 → 全出力ビット同一（チャンク分割不変も確認）。

usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/verify_factory.py
  env: NJ=128 SEED=1 N_EPS=1000 STEPS=0abc CKPT=<pth> SMOKE_DEVICE=cpu
"""
from __future__ import annotations

import glob
import os
import sys

# --- 学習時フラグと完全一致（verify_batch_eval.py と同一; src import より前）---
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ["SCHEDULER_OBS_URGENCY"] = "1"
os.environ["SCHEDULER_OBS_OCCUPANCY"] = "0"
os.environ["PCN_FILM"] = "0"
os.environ["PCN_FOURIER_CMD"] = "0"
os.environ["PCN_OBS_LOG"] = "1"
os.environ.setdefault("PCN_HIDDEN_DIM", "512")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

MAIN_REPO = "/home/noguchi/scheduler-sim-for-cb"
NJ = int(os.environ.get("NJ", "128"))
SEED = int(os.environ.get("SEED", "1"))
N_EPS = int(os.environ.get("N_EPS", "1000"))
STEPS = os.environ.get("STEPS", "0abc")
SMOKE_DEVICE = os.environ.get("SMOKE_DEVICE", "cpu")
CFG = os.environ.get("CFG", f"{MAIN_REPO}/experiments/distributed_pcn/job_synthetic_pcn.yml")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def default_ckpt(nj: int) -> str:
    pat = (
        f"{MAIN_REPO}/experiments/distributed_pcn/"
        f"run_synth{nj}_nded{nj}_r1/*/iteration_100/model_iter_100.pth"
    )
    hits = sorted(glob.glob(pat))
    assert hits, pat
    return hits[-1]


CKPT = os.environ.get("CKPT", default_ckpt(NJ))


def load_ref(nj: int) -> dict:
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"refeval_nj{nj}_seed{SEED}_ncmd40_*.npz")))
    assert hits, f"run verify_batch_eval.py first (NJ={nj})"
    return {k: v for k, v in np.load(hits[-1]).items()}


# ---------------------------------------------------------------------------
# [0] obs フル GPU 化 vs C 実装（タイ順差分の定量）
# ---------------------------------------------------------------------------
def step0_obs_diff() -> bool:
    import jax.numpy as jnp

    from batch_env_jax import make_initial_state
    from batch_eval_jax import precompute_job_queue
    from factory_jax import OBS_DIM, _alloc_jit_cached, _apply_jit_cached, _obs_jit_cached

    ref = load_ref(NJ)
    jobs = np.asarray(ref["jobs"], dtype=np.float64)
    ref_actions = ref["actions"]                        # (NCMD,T) 参照行動列
    ref_obs = ref["obs"]                                # (NCMD,T,221) C経路（bit一致検証済み）
    n_on, n_cl, n_window = int(ref["n_on"]), int(ref["n_cl"]), int(ref["n_window"])
    NCMD, T = ref_actions.shape
    norm_time = float(max(1, n_window))
    norm_nodes = float(max(n_on, n_cl))
    seg_len = 64

    state = make_initial_state(np.broadcast_to(jobs, (NCMD, T, 8)), n_on=n_on, n_cl=n_cl)
    h_max = int(jobs[:, 2].max())
    apply_jit = _apply_jit_cached(n_on, n_cl, h_max)
    obs_jit = _obs_jit_cached(n_window, norm_time, norm_nodes)
    jq = np.zeros((T + 1, 40), dtype=np.float64)
    jq[:T] = precompute_job_queue(jobs)
    jq_dev = jnp.asarray(jq)
    arr_arr = jobs[:, 0].astype(np.int64)

    obs_gpu = np.zeros((NCMD, T, OBS_DIM), dtype=np.float32)
    for t in range(T):
        e_view = min(T, ((t // seg_len) + 1) * seg_len)
        s_on, nd_on, s_cl, nd_cl = _alloc_jit_cached(h_max, "v1", e_view)(state)
        pw = jnp.maximum(0, s_on - int(arr_arr[t]))
        obs_gpu[:, t] = np.asarray(
            obs_jit(state.ev_obs[:, :e_view], jnp.int32(t), state.time, jq_dev[t], pw)
        )
        state, _ = apply_jit(
            state, jnp.asarray(ref_actions[:, t], dtype=jnp.int32),
            s_on, nd_on, s_cl, nd_cl,
        )

    # --- 差分定量（uint32 = ビット比較; 値は全て有限なので NaN の罠なし）---
    ev_ref = ref_obs[:, :, :180].reshape(NCMD, T, 30, 6).view(np.uint32)
    ev_gpu = obs_gpu[:, :, :180].reshape(NCMD, T, 30, 6).view(np.uint32)
    tail_ref = ref_obs[:, :, 180:].view(np.uint32)
    tail_gpu = obs_gpu[:, :, 180:].view(np.uint32)

    row_diff = np.any(ev_ref != ev_gpu, axis=3)          # (NCMD,T,30)
    n_rows = row_diff.size
    n_row_diff = int(row_diff.sum())
    step_diff = row_diff.any(axis=2)                     # (NCMD,T)
    n_step_diff = int(step_diff.sum())
    tail_ok = np.array_equal(tail_ref, tail_gpu)

    # 不一致 step の行集合一致（=並び替えのみ）を確認: 各 (i,t) の 30x6 を辞書式ソートして比較
    perm_only = True
    worst = None
    ii, tt = np.nonzero(step_diff)
    for i, t in zip(ii[:20000], tt[:20000]):
        a = ev_ref[i, t]
        b = ev_gpu[i, t]
        a_s = a[np.lexsort(a.T[::-1])]
        b_s = b[np.lexsort(b.T[::-1])]
        if not np.array_equal(a_s, b_s):
            perm_only = False
            worst = (int(i), int(t))
            break

    print(
        f"[0] obs GPU化 vs C実装 (NJ={NJ}, {NCMD}cmd×{T}step):\n"
        f"    イベント行 不一致 = {n_row_diff}/{n_rows} ({n_row_diff / n_rows:.4%})\n"
        f"    step単位   不一致 = {n_step_diff}/{NCMD * T} ({n_step_diff / (NCMD * T):.4%})\n"
        f"    行集合(値の多重集合)一致 = {'YES(並び替えのみ=タイ順起因)' if perm_only else f'NO first={worst}'}\n"
        f"    job_queue+urgency 41次元 = {'ビット一致' if tail_ok else '不一致!'}"
    )
    return tail_ok and perm_only


# ---------------------------------------------------------------------------
# 工場エピソード生成の共通部
# ---------------------------------------------------------------------------
def make_factory_episodes(jobs, ref, n_cmd_eps, n_rand_eps, policy, seed0=1234):
    """指令付き + ランダムの両モードでエピソードを作る（スモーク/決定性用）。"""
    from factory_jax import episodes_to_transitions, run_factory

    n_on, n_cl, n_window = int(ref["n_on"]), int(ref["n_cl"]), int(ref["n_window"])
    cmds40 = ref["commands"]
    commands = [
        (cmds40[i % len(cmds40)], float(NJ)) for i in range(n_cmd_eps)
    ]
    res_c = run_factory(
        jobs, n_cmd_eps, commands=commands, policy=policy, seed0=seed0,
        episode_id0=0, n_on=n_on, n_cl=n_cl, n_window=n_window,
    )
    probs = list(np.linspace(0.0, 1.0, n_rand_eps))
    res_r = run_factory(
        jobs, n_rand_eps, random_probs=probs, seed0=seed0,
        episode_id0=n_cmd_eps, n_on=n_on, n_cl=n_cl, n_window=n_window,
    )
    return episodes_to_transitions(res_c) + episodes_to_transitions(res_r), (res_c, res_r)


# ---------------------------------------------------------------------------
# [a] 形式スモーク: 本体 Learner が食べて 1 update 回るか（実運用の Ray 経路）
# ---------------------------------------------------------------------------
def step_a_smoke() -> bool:
    import ray
    import torch as th

    import src.distributed.distributed_pcn as dp
    from factory_jax import SamplingPolicy
    from scripts.pcn_replay_snapshot import create_eval_env, load_config

    cfg = load_config(CFG)
    cfg["param_env"]["n_jobs"] = NJ
    env = create_eval_env(cfg, job_seed=0, n_jobs=NJ)    # 学習と同じ job_seed=0
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    ref = load_ref(NJ)

    policy = SamplingPolicy(
        CKPT, NJ, device="cuda" if th.cuda.is_available() else "cpu"
    )
    episodes, _ = make_factory_episodes(jobs, ref, n_cmd_eps=8, n_rand_eps=8, policy=policy)
    n_tr = sum(len(e) for e in episodes)
    print(f"[a] 工場エピソード {len(episodes)} 本 ({n_tr} transitions) を Learner に投入")

    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True,
             logging_level="ERROR")
    try:
        buffer = dp.ReplayBuffer.remote(max_size=10000)  # 既に @ray.remote 済み(:721)
        LearnerActor = ray.remote(dp.Learner)
        learner = LearnerActor.remote(cfg, buffer, device=SMOKE_DEVICE)
        added = ray.get(buffer.add_batch.remote(episodes))
        assert added == len(episodes), f"added {added} != {len(episodes)}"
        loss = ray.get(
            learner.learn.remote(batch_size=2048, n_updates=1, use_training_cache=True)
        )
        n_replay = ray.get(learner.get_buffer_size.remote()) \
            if hasattr(dp.Learner, "get_buffer_size") else "?"
        ok = np.isfinite(loss)
        print(f"[a] learn(n_updates=1, use_training_cache=True) → loss={loss:.4f} "
              f"(finite={ok}) learner_replay={n_replay}")
        # 2回目: 指令選択（_choose_commands = replay の非支配集合から）も回ることを確認
        dr, hz = ray.get(learner._choose_commands.remote(16))
        print(f"[a] _choose_commands → dr={np.asarray(dr)}, hz={float(hz):.1f}")
        return bool(ok)
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# [b] 分布比較: CPU Actor サンプリング vs 工場（同一重み・同一指令）
# ---------------------------------------------------------------------------
def step_b_distribution() -> bool:
    import time as _time

    import torch as th

    from factory_jax import SamplingPolicy, run_factory
    from scripts.pcn_replay_snapshot import create_eval_env, load_config
    from src.agents.pcn_agent import PCN

    ref = load_ref(NJ)
    jobs = np.asarray(ref["jobs"], dtype=np.float64)
    cmds40 = ref["commands"]
    dr_mid = cmds40[len(cmds40) // 2].astype(np.float32)  # 中央帯の指令
    hz0 = float(NJ)

    # --- 参照: CPU Actor サンプリング（distributed Actor と同一演算列; mx=inf）---
    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=SEED, n_jobs=NJ)
    env.reset()
    assert np.array_equal(np.asarray(env.jobs, dtype=np.float64), jobs)
    ag = PCN(
        env, device="cpu", state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1.0, 1.0, 1.0 / max(1, NJ)], dtype=np.float32),
        learning_rate=1e-3, batch_size=512,
        hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
        project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False,
    )
    st = th.load(CKPT, map_location="cpu", weights_only=False)
    ag.model.load_state_dict(st.get("model_state_dict", st), strict=False)
    ag.model.eval()
    mx = np.full(2, np.inf, dtype=np.float32)

    cache = os.path.join(DATA_DIR, f"refsample_nj{NJ}_seed{SEED}_neps{N_EPS}.npz")
    if os.path.exists(cache) and os.environ.get("REFRESH", "0") != "1":
        d = np.load(cache)
        ach_ref, ref_wall = d["ach_ref"], float(d["ref_wall"])
        print(f"[b] 参照キャッシュ: {cache}")
    else:
        ach_ref = np.zeros((N_EPS, 2), dtype=np.float64)
        t0 = _time.perf_counter()
        for i in range(N_EPS):
            th.manual_seed(100000 + i)               # multinomial 乱数の独立化
            r = ag._run_episode(env, dr_mid.copy(), np.float32(hz0), mx, eval_mode=False)
            cost, _, avg_wait = env.calc_objective_values()
            ach_ref[i] = [float(cost), float(avg_wait)]
            if (i + 1) % 100 == 0:
                el = _time.perf_counter() - t0
                print(f"    ref {i + 1}/{N_EPS} ({el:.0f}s, {el / (i + 1):.2f}s/ep)",
                      flush=True)
        ref_wall = _time.perf_counter() - t0
        os.makedirs(DATA_DIR, exist_ok=True)
        np.savez_compressed(cache, ach_ref=ach_ref, ref_wall=ref_wall)
    print(f"[b] CPU Actor {N_EPS}ep = {ref_wall:.0f}s ({ref_wall / N_EPS:.2f}s/ep 逐次1コア)")

    # --- 工場: 同一指令 N_EPS 本 ---
    policy = SamplingPolicy(CKPT, NJ, device="cuda" if th.cuda.is_available() else "cpu")
    commands = [(dr_mid, hz0)] * N_EPS
    t0 = _time.perf_counter()
    res = run_factory(
        jobs, N_EPS, commands=commands, policy=policy, seed0=777,
        n_on=int(ref["n_on"]), n_cl=int(ref["n_cl"]), n_window=int(ref["n_window"]),
    )
    fac_wall = _time.perf_counter() - t0
    ach_fac = res["achieved"][:, [0, 2]]                 # [cost, avg_wait]
    print(f"[b] 工場 {N_EPS}ep = {fac_wall:.0f}s (B={N_EPS} 一括)")

    # --- 統計比較 ---
    try:
        from scipy.stats import ks_2samp
        ks = [ks_2samp(ach_ref[:, j], ach_fac[:, j]) for j in range(2)]
        ks_txt = [f"D={k.statistic:.4f} p={k.pvalue:.3f}" for k in ks]
        ks_pass = all(k.pvalue > 0.01 for k in ks)
    except ImportError:
        ks_txt = ["scipy無し"] * 2
        ks_pass = True
    qs = [10, 25, 50, 75, 90]
    print(f"[b] 指令 dr={dr_mid} hz={hz0} の達成分布 (N={N_EPS}):")
    for j, name in enumerate(["cost", "avg_wait"]):
        qr = np.percentile(ach_ref[:, j], qs)
        qf = np.percentile(ach_fac[:, j], qs)
        print(
            f"    {name:8s} mean ref={ach_ref[:, j].mean():.1f} fac={ach_fac[:, j].mean():.1f} "
            f"(Δ={ach_fac[:, j].mean() - ach_ref[:, j].mean():+.1f}, "
            f"{(ach_fac[:, j].mean() / max(1e-9, ach_ref[:, j].mean()) - 1) * 100:+.2f}%) | KS {ks_txt[j]}"
        )
        print(f"      q{qs} ref={np.round(qr, 1)} fac={np.round(qf, 1)}")
    print(f"[b] KS 検定 (p>0.01 = サンプリング乱数由来の範囲): {'PASS' if ks_pass else 'FAIL'}")
    return ks_pass


# ---------------------------------------------------------------------------
# [c] 決定性: 同一シード 2 回 + チャンク分割不変
# ---------------------------------------------------------------------------
def step_c_determinism() -> bool:
    import torch as th

    from factory_jax import SamplingPolicy, episodes_to_transitions, factory_chunks, run_factory

    ref = load_ref(NJ)
    jobs = np.asarray(ref["jobs"], dtype=np.float64)
    policy = SamplingPolicy(CKPT, NJ, device="cuda" if th.cuda.is_available() else "cpu")
    kw = dict(
        commands=[(ref["commands"][i % 40], float(NJ)) for i in range(64)],
        policy=policy, seed0=42,
        n_on=int(ref["n_on"]), n_cl=int(ref["n_cl"]), n_window=int(ref["n_window"]),
    )
    r1 = run_factory(jobs, 64, **kw)
    r2 = run_factory(jobs, 64, **kw)
    same = (
        np.array_equal(r1["actions"], r2["actions"])
        and np.array_equal(r1["obs"].view(np.uint32), r2["obs"].view(np.uint32))
        and np.array_equal(r1["rewards"], r2["rewards"])
        and np.array_equal(r1["achieved"], r2["achieved"])
    )
    print(f"[c] 同一シード2回 → actions/obs/rewards/achieved ビット同一: {same}")

    # チャンク分割不変: B=64 一括と chunk_b=16×4 で同一エピソード列
    eps_full = episodes_to_transitions(r1)
    eps_chunk = [
        e for ch in factory_chunks(jobs, 64, chunk_b=16, **kw) for e in ch
    ]
    same_chunk = len(eps_full) == len(eps_chunk) and all(
        len(a) == len(b)
        and all(
            np.array_equal(x.observation.view(np.uint32), y.observation.view(np.uint32))
            and x.action == y.action
            and np.array_equal(x.reward, y.reward)
            for x, y in zip(a, b)
        )
        for a, b in zip(eps_full, eps_chunk)
    )
    print(f"[c] チャンク分割(16×4) = 一括(64) 同一: {same_chunk}")

    # alloc 最適化の結果不変: 全OFF(v1 / 資源別ビュー全幅 / prune無し) vs 既定(全ON:
    # v1h + 資源別 gran=32 + prune_seg=64) の全出力ビット同一
    T = jobs.shape[0]
    r3 = run_factory(jobs, 64, alloc_gran=T, prune_seg=0, kernel="v1", **kw)
    same_gran = (
        np.array_equal(r1["actions"], r3["actions"])
        and np.array_equal(r1["obs"].view(np.uint32), r3["obs"].view(np.uint32))
        and np.array_equal(r1["rewards"], r3["rewards"])
        and np.array_equal(r1["achieved"], r3["achieved"])
    )
    print(f"[c] alloc最適化 全OFF(v1/全幅/prune無) = 既定(v1h/gran32/prune64) ビット同一: {same_gran}")
    return same and same_chunk and same_gran


def main():
    results = {}
    if "0" in STEPS:
        results["0_obs_diff"] = step0_obs_diff()
    if "a" in STEPS:
        results["a_smoke"] = step_a_smoke()
    if "b" in STEPS:
        results["b_distribution"] = step_b_distribution()
    if "c" in STEPS:
        results["c_determinism"] = step_c_determinism()
    print("\n== verify_factory 結果 ==")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
