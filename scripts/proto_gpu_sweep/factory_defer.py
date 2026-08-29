"""defer(3行動)対応の GPU 工場 rollout — フル per-node env(有限cloud) + ジョブ列回転。

torch env (event_native_env.py step :128-149 / _defer_rotate :202-216) と同一意味論:
  - SCHEDULER_ALLOW_DEFER=1 かつ action==2 のとき、現ジョブの defer 回数が
    SCHEDULER_DEFER_MAX 未満かつ残ジョブ>1 なら、現ジョブを SCHEDULER_DEFER_OFFSET
    後ろへ回転して配置しない(scheduled=False・報酬0・idx 不変)。
  - 上限到達 or 最終ジョブは強制 onprem 配置(ただし Transition の action は 2 のまま
    = CPU Actor が env_action をそのまま記録するのと同じ)。
  - desired_horizon は scheduled のときだけ減算(distributed_pcn.py:1586-1588)。
  - defer カウントはジョブに紐づく(torch は job_id キーの dict)。ここでは並び位置の
    カウント配列をジョブ行と一緒に回転することで同値にする。

env は batch_env_jax(make_initial_state/make_alloc_fn/make_apply_fn = 本体 env と
整数演算で一致する系譜; verify_batch_env 検証済み)をそのまま使う。既存 count 工場
(run_fused_count_*)と違い cloud も有限 per-node で追跡するため、固定行動列で本体と
cost/wait が一致する(弾性 cloud 近似なし)。

defer の実装トリック: defer する b は apply に done=True を偽装して渡す。apply は
active=~done で記録/報酬/idx 前進を全てスキップし、返り値の done は idx>=n_jobs で
再計算されるので偽装は残らない(apply 本体は無変更)。

エピソード長は可変(= n_jobs + defer 実行回数, 上限 n_jobs*(1+defer_max))なので
res["lengths"] を返す。build_factory_array_episodes が per-episode に切り詰める。

行動ソース(mode, jit static):
  "sample"    方策 categorical サンプル(Phase3 生産; torch _act の multinomial と同分布)
  "greedy"    方策 argmax(学習中 eval)
  "bernoulli" 固定 p の 0/1(Phase1; CPU 側 random_action_prob と同じく defer は出さない)
  "fixed"     与えた行動列(0/1/2; torch⇔JAX 一致検証用)
"""
from __future__ import annotations

import functools
import os
import sys
import time

import numpy as np

_MAIN = "/home/noguchi/scheduler-sim-for-cb"
_MAIN_PROTO = os.path.join(_MAIN, "scripts", "proto_gpu_sweep")
if _MAIN_PROTO not in sys.path:
    sys.path.insert(0, _MAIN_PROTO)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from batch_env_jax import (  # noqa: E402
    make_alloc_fn,
    make_apply_fn,
    make_initial_state,
)
from factory_fused import (  # noqa: E402
    _COST_HOLD,
    _DR_UB,
    _OBS_EFFICIENCY,
    _OBS_OCC_PRIOR,
    _OBS_OCCUPANCY,
    _ORACLE_NPZ,
    EFF_COST_K,
    EFF_GAIN_K,
    EFF_RATIO_K,
    _S_EMB_DROPOUT_P,
    _SLOWDOWN,
    _oracle_vals_np,
    load_pcn_params,
    obs_extra_dim,
    pcn_logits,  # 方策 forward(3行動対応, PCN_FC_DEPTH/FILM/OBS_LOG/Fourier/attn/dropout 全対応)
)
from factory_jax import (  # noqa: E402
    EVENT_FEATURES,
    JOB_QUEUE_SIZE,
    N_EVENTS_OBS,
)

# 追加観測列(occupancy/oracle)の次元(モジュール静的; 0 なら既定パス=従来 scan と同一グラフ)
_KEX = obs_extra_dim()


# ---------------------------------------------------------------------------
# 観測: _obs_from_buffer(factory_fused)の per-B job_queue 版
# ---------------------------------------------------------------------------
def _jq_from_state(jobs_full, idx, n_jobs):
    """job_queue 観測 40 要素を現在のジョブ並びから作る(B,40)。

    CPU _refresh_job_queue + _to_queue_job(roll(-1)) と同値: jobs[idx:idx+5] を
    各行左1回転して flatten、残ジョブ不足はゼロ埋め。defer でジョブ並びが動的に
    変わるため事前計算(precompute_job_queue)は使えず、毎ステップ gather する。
    """
    B, N, _ = jobs_full.shape
    pos = idx[:, None] + jnp.arange(5, dtype=idx.dtype)[None, :]          # (B,5)
    ok = pos < n_jobs[:, None]
    posc = jnp.minimum(pos, N - 1).astype(jnp.int32)
    rows = jnp.take_along_axis(jobs_full, posc[:, :, None], axis=1)       # (B,5,8)
    rolled = jnp.concatenate([rows[:, :, 1:], rows[:, :, :1]], axis=2)    # 各行 roll(-1)
    rolled = jnp.where(ok[:, :, None], rolled, 0.0)
    return rolled.reshape(B, 5 * 8)


def _obs_from_buffer_jqb(ob, ol, tm, jq_b, pw, *, n_window, norm_time, norm_nodes,
                         extras=None):
    """factory_fused._obs_from_buffer と同一演算で jq を per-B(B,40) にした版。
    extras: (B,k) f32 追加観測列(urgency の後ろ; None で従来 221 次元=bit不変)。"""
    B, E, _ = ob.shape
    ws = jnp.maximum(0, tm - n_window)
    starts = ob[:, :, 0]
    ends = ob[:, :, 1]
    keep = (jnp.arange(E)[None, :] < ol[:, None]) & (
        ends >= ws[:, None].astype(ob.dtype)
    )
    kk = min(N_EVENTS_OBS, E)
    idx64 = jnp.arange(E, dtype=jnp.int64)
    key = jnp.where(
        keep, starts.astype(jnp.int64) * jnp.int64(E + 1) + idx64[None, :],
        jnp.int64(-1),
    )
    topv, _ = jax.lax.top_k(key, kk)
    n_take = (topv >= 0).sum(axis=1).astype(jnp.int32)
    k = jnp.arange(N_EVENTS_OBS, dtype=jnp.int32)
    row_ok = k[None, :] < n_take[:, None]
    desc_pos = jnp.clip(n_take[:, None] - 1 - k[None, :], 0, kk - 1)
    sel = jnp.take_along_axis(topv, desc_pos[:, :kk] if kk < N_EVENTS_OBS else desc_pos, axis=1)
    if kk < N_EVENTS_OBS:
        sel = jnp.pad(sel, ((0, 0), (0, N_EVENTS_OBS - kk)), constant_values=-1)
    src_idx = (sel % jnp.int64(E + 1)).astype(jnp.int32)
    feats = jnp.take_along_axis(ob, src_idx[:, :, None], axis=1, mode="clip")
    wsf = ws[:, None].astype(jnp.float64)
    ev6 = jnp.stack(
        [
            jnp.clip((feats[:, :, 0] - wsf) / norm_time, 0.0, 1.0),
            jnp.clip((feats[:, :, 1] - wsf) / norm_time, 0.0, 1.0),
            jnp.clip(feats[:, :, 2] / norm_time, 0.0, 1.0),
            feats[:, :, 3],
            jnp.clip(feats[:, :, 4] / norm_nodes, 0.0, 1.0),
            jnp.clip(feats[:, :, 5] / norm_nodes, 0.0, 1.0),
        ],
        axis=2,
    )
    ev6 = jnp.where(row_ok[:, :, None], ev6, 0.0)
    ev_flat = ev6.reshape(B, N_EVENTS_OBS * EVENT_FEATURES).astype(jnp.float32)
    jq = jq_b.astype(jnp.float32)
    urg = jnp.clip(jnp.log1p(pw.astype(jnp.float64)) / 16.0, 0.0, 1.0).astype(jnp.float32)
    parts = [ev_flat, jq, urg[:, None]]
    if extras is not None:
        parts.append(extras.astype(jnp.float32))
    return jnp.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# 融合 scan 本体
# ---------------------------------------------------------------------------
def _rot2(x, src, bidx, jdst, cur, m):
    """(B,N) 配列の defer 回転: 位置 i..jdst-1 を1つ前へ、jdst に現ジョブ行。"""
    rot = jnp.take_along_axis(x, src, axis=1)
    rot = rot.at[bidx, jdst].set(cur)
    return jnp.where(m[:, None], rot, x)


def _rot3(x, src, bidx, jdst, cur, m):
    """(B,N,K) 配列の defer 回転(同上)。"""
    rot = jnp.take_along_axis(x, src[:, :, None], axis=1)
    rot = rot.at[bidx, jdst].set(cur)
    return jnp.where(m[:, None, None], rot, x)


@functools.partial(
    jax.jit,
    static_argnames=("n_on", "n_cl", "h_max", "n_window", "norm_time", "norm_nodes",
                     "kernel", "t_scan", "defer_max", "defer_offset", "mode"),
)
def _fused_defer_run(state0, dcnt0, params, key, probs, fixed_actions, dr0, hz0, ora, *,
                     n_on, n_cl, h_max, n_window, norm_time, norm_nodes, kernel,
                     t_scan, defer_max, defer_offset, mode):
    """defer 対応の融合 scan。state は batch_env_jax.EnvState(フル per-node)。

    scan 長 t_scan は静的(既定 N*(1+defer_max) = 総 defer 上限込みの最大長)。done 後は
    apply の active マスクで空回し(報酬0)。実長は steps(B,) で数え lengths として返す。
    ora: (N,) f64 = SCHEDULER_ORACLE_NPZ の per-position 真解雲率(OFF 時はゼロ配列)。
    本体 _front_job_oracle は index_next_job(位置)で引く=ジョブ回転に追従しないので、
    工場も位置 idx で引く(同挙動)。
    """
    B, N = state0.jobs_w.shape
    bidx = jnp.arange(B)
    alloc = make_alloc_fn(h_max, kernel, None)
    apply = make_apply_fn(n_on, n_cl, h_max)
    # [SCHEDULER_OBS_OCCUPANCY] 占有量 rank の分母集合(pt*nodes の多重集合)は defer 回転で
    # 不変なので初期状態から一度だけ作る(本体 _front_job_leverage の _occ_sorted と同値)。
    occ_all0 = (state0.jobs_w * state0.jobs_h.astype(jnp.int64)).astype(jnp.float64)

    def body(carry, xs):
        state, dcnt, dr, hz, steps = carry
        if mode == "fixed":
            t, fa_t = xs
        else:
            t = xs
        j = jnp.minimum(state.idx, N - 1)
        arr = state.jobs_arr[bidx, j]
        s_on, nd_on, s_cl, nd_cl = alloc(state)
        pw = jnp.maximum(0, s_on - arr)
        jq_b = _jq_from_state(state.jobs_full, state.idx, state.n_jobs)
        extras = None
        if _KEX:
            active = state.idx < state.n_jobs        # 本体: j>=len(jobs) では extras=0
            ex_cols = []
            if _OBS_OCCUPANCY:
                w_j = state.jobs_w[bidx, j]
                h_j = state.jobs_h[bidx, j]
                if _OBS_OCC_PRIOR:
                    # event_c_env._front_job_leverage(OCC_PRIOR): 物理的負荷寄与率の対数正規化
                    occ_frac = (w_j.astype(jnp.float64) / norm_time) * (
                        h_j.astype(jnp.float64) / max(1.0, float(n_on)))
                    v = jnp.clip(jnp.log1p(occ_frac * 1.0e3) / 4.3, 0.0, 1.0)
                else:
                    # rank = searchsorted(sorted, occ, side='right')/N = mean(occ_all <= occ)
                    occ_j = (w_j * h_j.astype(jnp.int64)).astype(jnp.float64)
                    v = (occ_all0 <= occ_j[:, None]).sum(axis=1).astype(jnp.float64) / N
                    v = jnp.clip(v, 0.0, 1.0)
                ex_cols.append(jnp.where(active, v, 0.0))
            if _ORACLE_NPZ:
                ex_cols.append(jnp.where(active, ora[j], 0.0))
            if _OBS_EFFICIENCY:
                # event_c_env._front_job_efficiency と同一式（factory_defer_ev と同じ）。
                a_e = (state.jobs_w[bidx, j]
                       * state.jobs_h[bidx, j].astype(jnp.int64)).astype(jnp.float64)
                b_e = jnp.maximum(0, s_on - s_cl).astype(jnp.float64)
                la_e = jnp.log1p(a_e)
                lb_e = jnp.log1p(b_e)
                ex_cols.append(jnp.where(
                    active, jnp.clip(la_e / EFF_COST_K, 0.0, 1.0), 0.0))
                ex_cols.append(jnp.where(
                    active, jnp.clip(lb_e / EFF_GAIN_K, 0.0, 1.0), 0.0))
                ex_cols.append(jnp.where(
                    active, jnp.clip((lb_e - la_e) / EFF_RATIO_K + 0.5, 0.0, 1.0), 0.0))
            extras = jnp.stack(ex_cols, axis=1)
        obs = _obs_from_buffer_jqb(state.ev_obs, state.idx, state.time, jq_b, pw,
                                   n_window=n_window, norm_time=norm_time,
                                   norm_nodes=norm_nodes, extras=extras)
        # [PCN_S_EMB_DROPOUT] CPU actor は train モードのまま rollout/eval するため、
        # フラグ>0 なら greedy でも per-step dropout ノイズが乗る(同分布で再現)。
        _dk = (jax.random.fold_in(jax.random.fold_in(key, t), 0x0D)
               if _S_EMB_DROPOUT_P > 0.0 else None)
        if mode == "greedy":
            logits = pcn_logits(params, obs, dr, hz, dkey=_dk)
            a_raw = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        elif mode == "sample":
            logits = pcn_logits(params, obs, dr, hz, dkey=_dk)
            a_raw = jax.random.categorical(
                jax.random.fold_in(key, t), logits, axis=-1
            ).astype(jnp.int32)
        elif mode == "bernoulli":
            u = jax.random.uniform(jax.random.fold_in(key, t), (B,))
            a_raw = (u < probs).astype(jnp.int32)
        else:  # fixed
            a_raw = fa_t.astype(jnp.int32)

        # --- defer 判定(本体 step :136-149)---
        is_d = a_raw == 2
        can = (dcnt[bidx, j] < defer_max) & ((state.n_jobs - state.idx) > 1)
        do_defer = is_d & can & (~state.done)
        a01 = jnp.where(is_d, 0, a_raw)   # cap到達/最終ジョブの action=2 は強制 onprem

        # defer する b は done を偽装して apply を空回し(報酬0・記録なし・idx 不変)。
        # apply 返り値の done は idx>=n_jobs で再計算されるので偽装は残らない。
        st_mask = state._replace(done=state.done | do_defer)
        st2, out = apply(st_mask, a01, s_on, nd_on, s_cl, nd_cl)

        # --- defer 回転(本体 _defer_rotate: i=idx の行を jdst=min(idx+offset,N-1) へ)---
        dcnt = dcnt.at[bidx, j].add(do_defer.astype(dcnt.dtype))  # 加算→回転(行と一緒に動く)
        jdst = jnp.minimum(state.idx + defer_offset, N - 1)
        pos = jnp.arange(N, dtype=jnp.int32)[None, :]
        in_shift = (pos >= state.idx[:, None]) & (pos < jdst[:, None])
        src = jnp.where(in_shift, pos + 1, pos)                    # pos+1 <= jdst <= N-1
        st2 = st2._replace(
            jobs_w=_rot2(st2.jobs_w, src, bidx, jdst, st2.jobs_w[bidx, j], do_defer),
            jobs_h=_rot2(st2.jobs_h, src, bidx, jdst, st2.jobs_h[bidx, j], do_defer),
            jobs_arr=_rot2(st2.jobs_arr, src, bidx, jdst, st2.jobs_arr[bidx, j], do_defer),
            jobs_full=_rot3(st2.jobs_full, src, bidx, jdst, st2.jobs_full[bidx, j], do_defer),
        )
        dcnt = _rot2(dcnt, src, bidx, jdst, dcnt[bidx, j], do_defer)

        # 指令更新: actor(distributed_pcn.py:1579-1584)と同一演算列
        dr2 = (dr - out.rewards.astype(jnp.float32)).astype(jnp.float32)
        if _COST_HOLD:
            # [anti-ration] cost目標(index1)を初期指令で一定保持(:1580-1581)。
            # eval 側(pcn_agent._run_episode :4370,4405)も同適用なので全モード共通。
            dr2 = dr2.at[:, 1].set(dr0[:, 1])
        if _DR_UB is not None and mode != "greedy":
            # [元PCN復元] 上限クリップ(:1582-1584)。torch では Actor 探索 rollout のみ適用・
            # eval(pcn_agent._run_episode eval_mode=True)は不適用なので greedy では掛けない。
            dr2 = jnp.minimum(dr2, jnp.float32(_DR_UB))
        hz2 = jnp.where(out.scheduled, jnp.maximum(hz - 1.0, 1.0), hz)  # scheduled 時のみ減
        steps2 = steps + (~state.done).astype(jnp.int32)
        return (st2, dcnt, dr2, hz2, steps2), (
            a_raw.astype(jnp.int8), out.rewards, obs.astype(jnp.float32))

    ts = jnp.arange(t_scan, dtype=jnp.int64)
    xs = (ts, fixed_actions) if mode == "fixed" else ts
    carry0 = (state0, dcnt0, dr0, hz0, jnp.zeros((B,), dtype=jnp.int32))
    (final, dcnt, dr, hz, steps), (actions, rewards, obs_seq) = lax.scan(body, carry0, xs)

    # 最終 obs(done 後: job_queue 空・urgency 0・追加観測 0)
    obs_last = _obs_from_buffer_jqb(
        final.ev_obs, final.idx, final.time,
        jnp.zeros((B, JOB_QUEUE_SIZE), dtype=jnp.float64),
        jnp.zeros((B,), dtype=jnp.int64),
        n_window=n_window, norm_time=norm_time, norm_nodes=norm_nodes,
        extras=jnp.zeros((B, _KEX), dtype=jnp.float32) if _KEX else None)
    cost = final.cost_total
    if _SLOWDOWN:
        # slowdown では achieved 待ち = Σ wmetric = -Σ rewards[:,:,0]
        # (apply が rewards[0]=-wait/pt を返す。defer 空回しステップは 0 で寄与しない)
        wsum = -(rewards[:, :, 0].sum(axis=0))
    else:
        wsum = final.waits.sum(axis=1)
    mksp = jnp.max(final.ev_end, axis=(1, 2))
    all_done = final.done.all()
    return cost, mksp, wsum, steps, all_done, actions, rewards, obs_seq, obs_last


def run_fused_defer(jobs, commands=None, *, params=None, ckpt_path=None, probs=None,
                    fixed_actions=None, greedy=False, seed=0, episode_id0=0,
                    n_on=256, n_cl=1024, n_window=100, kernel="v1h",
                    defer_max=3, defer_offset=1, t_scan=None):
    """defer(3行動)対応 rollout。返り値は run_fused_count_cmd / run_fused_count_random と
    同形式 + lengths(B,)。overflow は「t_scan 内に全エピソード完走できなかった」フラグ
    (既定 t_scan=N*(1+defer_max) では構造的に発生しない)。

    - commands 指定: mode=sample(既定)/greedy(greedy=True)。params or ckpt_path 必須。
    - probs 指定: mode=bernoulli(Phase1 互換; 0/1 のみで defer は出ない)。t_scan=N。
    - fixed_actions 指定 (B, t_scan) int: mode=fixed(検証用)。
    - jobs は (T,8)=全エピソード共通(従来; broadcast) または (B,T,8)=エピソード別
      ([PCN_MIX_REGIMES] レジーム混合用。各行が別 workload)。h_max はレジーム間で同一
      (arrival scale は col0 のみ倍率)なので jit 再コンパイルは起きない。
    """
    jobs = np.asarray(jobs, dtype=np.float64)
    per_ep = jobs.ndim == 3          # (B,T,8) エピソード別ジョブ列(regmix)
    T = int(jobs.shape[1] if per_ep else jobs.shape[0])
    if fixed_actions is not None:
        mode = "fixed"
        B = int(np.asarray(fixed_actions).shape[0])
    elif probs is not None:
        mode = "bernoulli"
        B = len(probs)
        if t_scan is None:
            t_scan = T  # 0/1 のみ = defer なし = 長さ N 固定
    else:
        mode = "greedy" if greedy else "sample"
        B = len(commands)
    if t_scan is None:
        t_scan = T * (1 + int(defer_max))
    if mode in ("sample", "greedy") and params is None:
        params = load_pcn_params(ckpt_path)

    if per_ep and int(jobs.shape[0]) != B:
        raise ValueError(f"per-episode jobs の先頭次元 {jobs.shape[0]} != バッチ数 {B}")
    jobs_b = jobs if per_ep else np.broadcast_to(jobs, (B, T, 8))
    state0 = make_initial_state(jobs_b, n_on=n_on, n_cl=n_cl)
    h_max = int(jobs_b[..., 2].max())
    dcnt0 = jnp.zeros((B, T), dtype=jnp.int32)
    if commands is not None:
        dr0 = jnp.asarray(np.stack([np.asarray(c[0], np.float32) for c in commands]))
        hz0 = jnp.asarray(np.array([float(c[1]) for c in commands], np.float32))
    else:
        dr0 = jnp.zeros((B, 2), dtype=jnp.float32)
        hz0 = jnp.ones((B,), dtype=jnp.float32)
    probs_dev = (jnp.asarray(np.asarray(probs, dtype=np.float32))
                 if probs is not None else jnp.zeros((B,), dtype=jnp.float32))
    fa_dev = (jnp.asarray(np.asarray(fixed_actions, dtype=np.int32)).T
              if fixed_actions is not None else jnp.zeros((int(t_scan), B), dtype=jnp.int32))
    params_in = params if params is not None else {}
    key = jax.random.PRNGKey(seed)
    # [SCHEDULER_ORACLE_NPZ] 位置 idx で引く per-position 配列(OFF はゼロ; 本文参照)
    ora_dev = jnp.asarray(_oracle_vals_np(T) if _ORACLE_NPZ
                          else np.zeros(T, dtype=np.float64))

    t0 = time.perf_counter()
    cost, mksp, wsum, steps, all_done, actions, rewards, obs_seq, obs_last = \
        _fused_defer_run(
            state0, dcnt0, params_in, key, probs_dev, fa_dev, dr0, hz0, ora_dev,
            n_on=n_on, n_cl=n_cl, h_max=h_max, n_window=n_window,
            norm_time=float(max(1, n_window)), norm_nodes=float(max(n_on, n_cl)),
            kernel=kernel, t_scan=int(t_scan), defer_max=int(defer_max),
            defer_offset=int(defer_offset), mode=mode)
    jax.block_until_ready(actions)
    wall = time.perf_counter() - t0

    obs_all = np.concatenate(
        [np.asarray(obs_seq).transpose(1, 0, 2), np.asarray(obs_last)[:, None, :]], axis=1,
    ).astype(np.float32)
    res = {
        "obs": obs_all,                                          # (B, t_scan+1, D)
        "actions": np.asarray(actions).T.astype(np.int8),        # (B, t_scan)
        "rewards": np.asarray(rewards).transpose(1, 0, 2),       # (B, t_scan, 2)
        "achieved": np.stack([np.asarray(cost, np.float64), np.asarray(mksp, np.float64),
                              np.asarray(wsum, np.float64) / T], axis=1),
        "lengths": np.asarray(steps, np.int32),                  # (B,) 実エピソード長
        "mode": {"sample": "command", "greedy": "command"}.get(mode, mode),
        "episode_ids": np.arange(episode_id0, episode_id0 + B, dtype=np.int64),
        "seed0": int(seed),
        "stats": {"wall": float(wall)},
        "overflow": not bool(np.asarray(all_done)),
    }
    if commands is not None:
        res["commands0"] = np.stack([np.asarray(c[0], np.float32) for c in commands])
    if probs is not None:
        res["probs"] = np.asarray(probs, dtype=np.float32)
    return res
