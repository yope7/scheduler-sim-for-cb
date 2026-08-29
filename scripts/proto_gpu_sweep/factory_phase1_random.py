"""段A: Phase1 ランダム収集の工場版（カウント表現・固定 Bernoulli(p)・obs 列付き）。

本体 scripts/proto_gpu_sweep/factory_fused.py の _fused_count_run（オンプレ N軸なしカウント
配置 + 弾性cloud, KS一致検証済み）と同一の env 演算列を用い、方策 forward の代わりに
per-episode 固定 p でサンプルし、各 step の obs(221) を scan 出力に載せる。
返り値 dict は run_factory(random_probs=...) と同形式 → episodes_to_transitions が使える。

【重要】このモジュールは worktree 上の追加ファイル。factory 内部関数は「本体(main repo)側」の
scripts/proto_gpu_sweep から import する（worktree の stale なコピーを掴まないよう MAIN_PROTO を
sys.path 先頭に固定）。本体ソースは一切変更していない（run_fused_count の兄弟関数を新規追加した形）。
"""
from __future__ import annotations

import functools
import os
import sys
import time

import numpy as np

_MAIN = "/home/noguchi/scheduler-sim-for-cb"
_MAIN_PROTO = os.path.join(_MAIN, "scripts", "proto_gpu_sweep")
# 本体側 factory モジュールを最優先で解決（worktree の同名 stale を避ける）
if _MAIN_PROTO not in sys.path:
    sys.path.insert(0, _MAIN_PROTO)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

# --- 本体 factory 内部（同一実装を再利用; 本体ソースは無変更）---
from batch_env_jax import INF64  # noqa: E402
from batch_eval_jax import precompute_job_queue  # noqa: E402
from factory_fused import (  # noqa: E402
    _COST_HOLD, _DR_UB, _S_EMB_DROPOUT_P, _SLOWDOWN,
    _compact_count, _compact_obs, _obs_from_buffer,
    job_extra_feats, load_pcn_params, obs_extra_dim, pcn_p_cloud,
)
from gpu_sweep_jax import _candidates as _cand_fn  # noqa: E402

# 追加観測列(occupancy/oracle)の次元(モジュール静的; 0 なら既定パス=従来 scan と同一グラフ)
_KEX = obs_extra_dim()


@functools.partial(
    jax.jit,
    static_argnames=("n_on", "h_max", "n_window", "norm_time", "norm_nodes",
                     "e_alloc", "e_obs", "kcompact"),
)
def _fused_count_random_run(jobs_cols, probs, key, jq, jq_last, *, n_on, h_max,
                            n_window, norm_time, norm_nodes, e_alloc, e_obs, kcompact):
    """区間/カウント表現の融合 scan（固定 Bernoulli(p)・obs 列も返す）。

    factory_fused._fused_count_run の body と同一演算列。差分のみ:
    p=probs 固定 / dr,hz carry なし / per-step obs を出力 / 最終 obs も返す。
    """
    arr_c, w_c, h_c, suff_c, ex_c = jobs_cols
    B = probs.shape[0]
    T = arr_c.shape[0]
    bidx = jnp.arange(B)

    es = jnp.full((B, e_alloc), INF64, dtype=jnp.int64)
    ee = jnp.full((B, e_alloc), -1, dtype=jnp.int64)
    eh = jnp.zeros((B, e_alloc), dtype=jnp.int32)
    oon = jnp.zeros((B, e_alloc, n_on), dtype=bool)
    el = jnp.zeros((B,), dtype=jnp.int32)
    n_idx = jnp.arange(n_on, dtype=jnp.int32)
    ob = jnp.zeros((B, e_obs, 6), dtype=jnp.float64)
    ol = jnp.zeros((B,), dtype=jnp.int32)
    tm = jnp.zeros((B,), dtype=jnp.int64)
    cost = jnp.zeros((B,), dtype=jnp.int64)
    # [SCHEDULER_WAIT_METRIC=slowdown] wmetric=wait/pt は実数 → 累積を f64 に切替
    wsum = jnp.zeros((B,), dtype=jnp.float64 if _SLOWDOWN else jnp.int64)
    mksp = jnp.zeros((B,), dtype=jnp.int64)
    ovf = jnp.zeros((), dtype=bool)

    def obs_of(ob, ol, tm, jq_t, pw, ex_t=None):
        extras = (jnp.broadcast_to(ex_t.astype(jnp.float32)[None, :], (B, _KEX))
                  if _KEX else None)
        return _obs_from_buffer(ob, ol, tm, jq_t, pw, n_window=n_window,
                                norm_time=norm_time, norm_nodes=norm_nodes,
                                extras=extras)

    def body(carry, xs):
        es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, ovf = carry
        t, jq_t, arr, w, h, suffnext, ex_t = xs
        cand, valid = _cand_fn(es, ee, jnp.full((B,), arr, jnp.int64))
        t_end = cand + w
        ov = (es[:, None, :] < t_end[:, :, None]) & (ee[:, None, :] > cand[:, :, None])
        occupied = jnp.einsum("bte,be->bt", ov.astype(jnp.int32), eh)
        free_cnt = n_on - occupied
        feasible = (free_cnt >= h) & valid
        t_idx = jnp.argmax(feasible, axis=1)
        found = feasible.any(axis=1)
        fb = jnp.maximum(arr, jnp.max(ee, axis=1))
        s_on = jnp.where(found, jnp.take_along_axis(cand, t_idx[:, None], 1)[:, 0], fb)
        ov_at = (es < (s_on + w)[:, None]) & (ee > s_on[:, None])
        per_node = jnp.einsum("be,ben->bn", ov_at.astype(jnp.int32), oon.astype(jnp.int32))
        free_node = per_node == 0
        sort_key = jnp.where(free_node, n_idx[None, :], n_on + n_idx[None, :])
        nodes_sorted = jnp.sort(sort_key, axis=1)
        sel_nodes = nodes_sorted[:, :h_max]
        occ_at = sel_nodes[:, 0]
        pw = jnp.maximum(0, s_on - arr)
        # --- この step の obs → 固定 p でサンプル ---
        obs = obs_of(ob, ol, tm, jq_t, pw, ex_t)
        u = jax.random.uniform(jax.random.fold_in(key, t), (B,))
        use_cloud = u < probs
        start = jnp.where(use_cloud, arr, s_on)
        end = start + w
        wait = start - arr
        cost_step = jnp.where(use_cloud, w * h, 0).astype(jnp.int64)
        if _SLOWDOWN:
            # 本体 event_native_env.step :177-180: wmetric = wait / max(1, pt)
            wm = wait.astype(jnp.float64) / jnp.maximum(1.0, w.astype(jnp.float64))
            rew = jnp.stack([-wm, -cost_step.astype(jnp.float64)], axis=1)
        else:
            wm = wait
            rew = jnp.stack([-wait, -cost_step], axis=1).astype(jnp.float64)
        is_on = ~use_cloud
        slot = jnp.where(is_on, el, e_alloc)
        es = es.at[bidx, slot].set(start, mode="drop")
        ee = ee.at[bidx, slot].set(end, mode="drop")
        eh = eh.at[bidx, slot].set(jnp.int32(h), mode="drop")
        karr = jnp.arange(h_max, dtype=jnp.int32)
        valid_k = karr[None, :] < jnp.int32(h)
        nd_i = jnp.where(valid_k & is_on[:, None], sel_nodes, n_on)
        oon = oon.at[bidx[:, None], slot[:, None], nd_i].set(True, mode="drop")
        el = el + is_on.astype(jnp.int32)
        snode = jnp.where(use_cloud, 0, occ_at).astype(jnp.float64)
        feat = jnp.stack([start.astype(jnp.float64), end.astype(jnp.float64),
                          jnp.full((B,), w, jnp.float64), use_cloud.astype(jnp.float64),
                          snode, jnp.full((B,), h, jnp.float64)], axis=1)
        ob = ob.at[bidx, ol].set(feat, mode="drop")
        ol = ol + 1
        ovf = ovf | (el.max() >= e_alloc) | (ol.max() >= e_obs)
        tm = start
        cost = cost + cost_step
        wsum = wsum + wm
        mksp = jnp.maximum(mksp, end)

        def do_compact(op):
            es, ee, eh, oon, ob, ol, tm = op
            st, en, hh, c = _compact_count(es, ee, eh, suffnext)
            E = ee.shape[1]
            order = jnp.argsort(~(ee >= suffnext), axis=1, stable=True)
            pos_ok = jnp.arange(E)[None, :] < c[:, None]
            oon2 = jnp.take_along_axis(oon, order[:, :, None], axis=1) & pos_ok[:, :, None]
            ob2, ol2 = _compact_obs(ob, ol, tm - n_window)
            return st, en, hh, oon2, c, ob2, ol2

        def no_compact(op):
            es, ee, eh, oon, ob, ol, tm = op
            return es, ee, eh, oon, el, ob, ol

        es, ee, eh, oon, el, ob, ol = lax.cond(
            (t + 1) % kcompact == 0, do_compact, no_compact,
            (es, ee, eh, oon, ob, ol, tm))

        carry = (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, ovf)
        return carry, (use_cloud.astype(jnp.int8), rew, obs)

    xs = (jnp.arange(T, dtype=jnp.int64), jq, arr_c, w_c, h_c, suff_c, ex_c)
    carry0 = (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, ovf)
    carry, (actions, rewards, obs_seq) = lax.scan(body, carry0, xs)
    (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, ovf) = carry
    obs_last = obs_of(ob, ol, tm, jq_last, jnp.zeros((B,), dtype=jnp.int64),
                      jnp.zeros((_KEX,), dtype=jnp.float64) if _KEX else None)
    return cost, mksp, wsum, ovf, actions, rewards, obs_seq, obs_last


def run_fused_count_random(jobs, probs, *, seed=0, episode_id0=0, n_on=1520,
                           n_cl=65536, n_window=100, e_alloc=512, e_obs=1152,
                           kcompact=64):
    """Phase1 ランダム収集の工場版。返り値は run_factory(random_probs=...) と同形式。

    n_on は feasibility(free_cnt=n_on-occupied)用の実オンプレ数。obs のノード正規化は
    本体 CPU env._norm_nodes=max(n_on,n_cl) に合わせる（train/eval 一貫性のため必須;
    run_fused_count は norm_nodes=n_on だが Phase1 種は CPU obs と同スケールでなければ
    Phase3/eval 収集の obs と分布がズレる）。
    """
    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    B = len(probs)
    arr = jobs[:, 0].astype(np.int64)
    w = jobs[:, 1].astype(np.int64)
    h = jobs[:, 2].astype(np.int64)
    suffmin = np.minimum.accumulate(arr[::-1])[::-1]
    suffnext = np.empty(T, dtype=np.int64)
    suffnext[:-1] = suffmin[1:]
    suffnext[-1] = np.iinfo(np.int64).max // 4
    h_max = int(h.max())
    jq_np = precompute_job_queue(jobs)
    jq = jnp.asarray(jq_np)
    jq_last = jnp.zeros((jq_np.shape[1],), dtype=jnp.float64)
    ex_np = job_extra_feats(jobs, n_window, n_on)         # (T,_KEX) 静的追加観測列
    jobs_cols = (jnp.asarray(arr), jnp.asarray(w), jnp.asarray(h), jnp.asarray(suffnext),
                 jnp.asarray(ex_np))
    probs_dev = jnp.asarray(np.asarray(probs, dtype=np.float32))
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    cost, mksp, wsum, ovf, actions, rewards, obs_seq, obs_last = _fused_count_random_run(
        jobs_cols, probs_dev, key, jq, jq_last, n_on=n_on, h_max=h_max, n_window=n_window,
        norm_time=float(max(1, n_window)), norm_nodes=float(max(n_on, n_cl)),
        e_alloc=int(e_alloc), e_obs=int(e_obs), kcompact=int(kcompact))
    jax.block_until_ready(actions)
    wall = time.perf_counter() - t0
    obs_all = np.concatenate(
        [np.asarray(obs_seq).transpose(1, 0, 2),
         np.asarray(obs_last)[:, None, :]], axis=1,
    ).astype(np.float32)
    return {
        "obs": obs_all,
        "actions": np.asarray(actions).T.astype(np.int8),
        "rewards": np.asarray(rewards).transpose(1, 0, 2),
        "achieved": np.stack([np.asarray(cost, np.float64), np.asarray(mksp, np.float64),
                              np.asarray(wsum, np.float64) / T], axis=1),
        "mode": "random",
        "probs": np.asarray(probs, dtype=np.float32),
        "episode_ids": np.arange(episode_id0, episode_id0 + B, dtype=np.int64),
        "seed0": int(seed),
        "stats": {"wall": float(wall)},
        "overflow": bool(np.asarray(ovf)),
    }


# ---------------------------------------------------------------------------
# Phase 3 用: カウント表現 + 指令サンプリング(方策forward) + obs 列を返す版。
#   本体 factory_fused.run_fused_count と同一 env 演算列(O(N)オンプレ+弾性cloud)だが、
#   各 step の obs(221) を scan 出力に載せて episodes_to_transitions(mode="command") 互換の
#   res を返す。run_fused_count は obs を返さない=Transition化できないので、その穴を埋める。
#   段Bの Phase3(指令付き rollout)を O(N) 工場で回すための primitive。
# ---------------------------------------------------------------------------
@functools.partial(
    jax.jit,
    static_argnames=("n_on", "h_max", "n_window", "norm_time", "norm_nodes",
                     "e_alloc", "e_obs", "kcompact", "greedy"),
)
def _fused_count_cmd_run(jobs_cols, params, key, jq, jq_last, dr0, hz0, *, n_on, h_max,
                         n_window, norm_time, norm_nodes, e_alloc, e_obs, kcompact,
                         greedy=False):
    """区間/カウント表現の融合 scan（指令サンプリング・obs 列も返す）。

    factory_fused._fused_count_run と同一 env 演算列。差分: per-step obs を出力・最終 obs も返す
    （p=pcn_p_cloud(params,obs,dr,hz) と dr/hz carry は run_fused_count と同一）。
    """
    arr_c, w_c, h_c, suff_c, ex_c = jobs_cols
    B = dr0.shape[0]
    T = arr_c.shape[0]
    bidx = jnp.arange(B)

    es = jnp.full((B, e_alloc), INF64, dtype=jnp.int64)
    ee = jnp.full((B, e_alloc), -1, dtype=jnp.int64)
    eh = jnp.zeros((B, e_alloc), dtype=jnp.int32)
    oon = jnp.zeros((B, e_alloc, n_on), dtype=bool)
    el = jnp.zeros((B,), dtype=jnp.int32)
    n_idx = jnp.arange(n_on, dtype=jnp.int32)
    ob = jnp.zeros((B, e_obs, 6), dtype=jnp.float64)
    ol = jnp.zeros((B,), dtype=jnp.int32)
    tm = jnp.zeros((B,), dtype=jnp.int64)
    cost = jnp.zeros((B,), dtype=jnp.int64)
    # [SCHEDULER_WAIT_METRIC=slowdown] wmetric=wait/pt は実数 → 累積を f64 に切替
    wsum = jnp.zeros((B,), dtype=jnp.float64 if _SLOWDOWN else jnp.int64)
    mksp = jnp.zeros((B,), dtype=jnp.int64)
    ovf = jnp.zeros((), dtype=bool)

    def obs_of(ob, ol, tm, jq_t, pw, ex_t=None):
        extras = (jnp.broadcast_to(ex_t.astype(jnp.float32)[None, :], (B, _KEX))
                  if _KEX else None)
        return _obs_from_buffer(ob, ol, tm, jq_t, pw, n_window=n_window,
                                norm_time=norm_time, norm_nodes=norm_nodes,
                                extras=extras)

    def body(carry, xs):
        es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf = carry
        t, jq_t, arr, w, h, suffnext, ex_t = xs
        cand, valid = _cand_fn(es, ee, jnp.full((B,), arr, jnp.int64))
        t_end = cand + w
        ov = (es[:, None, :] < t_end[:, :, None]) & (ee[:, None, :] > cand[:, :, None])
        occupied = jnp.einsum("bte,be->bt", ov.astype(jnp.int32), eh)
        free_cnt = n_on - occupied
        feasible = (free_cnt >= h) & valid
        t_idx = jnp.argmax(feasible, axis=1)
        found = feasible.any(axis=1)
        fb = jnp.maximum(arr, jnp.max(ee, axis=1))
        s_on = jnp.where(found, jnp.take_along_axis(cand, t_idx[:, None], 1)[:, 0], fb)
        ov_at = (es < (s_on + w)[:, None]) & (ee > s_on[:, None])
        per_node = jnp.einsum("be,ben->bn", ov_at.astype(jnp.int32), oon.astype(jnp.int32))
        free_node = per_node == 0
        sort_key = jnp.where(free_node, n_idx[None, :], n_on + n_idx[None, :])
        nodes_sorted = jnp.sort(sort_key, axis=1)
        sel_nodes = nodes_sorted[:, :h_max]
        occ_at = sel_nodes[:, 0]
        pw = jnp.maximum(0, s_on - arr)
        obs = obs_of(ob, ol, tm, jq_t, pw, ex_t)
        # [PCN_S_EMB_DROPOUT] CPU actor は train モードで rollout/eval するため同分布で再現
        _dk = (jax.random.fold_in(jax.random.fold_in(key, t), 0x0D)
               if _S_EMB_DROPOUT_P > 0.0 else None)
        p = pcn_p_cloud(params, obs, dr, hz, dkey=_dk)
        u = jax.random.uniform(jax.random.fold_in(key, t), (B,))
        # greedy(eval): argmax 相当 = P(cloud)>0.5 で決定的（方策の greedy 評価）。
        use_cloud = (p > 0.5) if greedy else (u < p)
        start = jnp.where(use_cloud, arr, s_on)
        end = start + w
        wait = start - arr
        cost_step = jnp.where(use_cloud, w * h, 0).astype(jnp.int64)
        if _SLOWDOWN:
            # 本体 event_native_env.step :177-180: wmetric = wait / max(1, pt)
            wm = wait.astype(jnp.float64) / jnp.maximum(1.0, w.astype(jnp.float64))
            rew = jnp.stack([-wm, -cost_step.astype(jnp.float64)], axis=1)
        else:
            wm = wait
            rew = jnp.stack([-wait, -cost_step], axis=1).astype(jnp.float64)
        is_on = ~use_cloud
        slot = jnp.where(is_on, el, e_alloc)
        es = es.at[bidx, slot].set(start, mode="drop")
        ee = ee.at[bidx, slot].set(end, mode="drop")
        eh = eh.at[bidx, slot].set(jnp.int32(h), mode="drop")
        karr = jnp.arange(h_max, dtype=jnp.int32)
        valid_k = karr[None, :] < jnp.int32(h)
        nd_i = jnp.where(valid_k & is_on[:, None], sel_nodes, n_on)
        oon = oon.at[bidx[:, None], slot[:, None], nd_i].set(True, mode="drop")
        el = el + is_on.astype(jnp.int32)
        snode = jnp.where(use_cloud, 0, occ_at).astype(jnp.float64)
        feat = jnp.stack([start.astype(jnp.float64), end.astype(jnp.float64),
                          jnp.full((B,), w, jnp.float64), use_cloud.astype(jnp.float64),
                          snode, jnp.full((B,), h, jnp.float64)], axis=1)
        ob = ob.at[bidx, ol].set(feat, mode="drop")
        ol = ol + 1
        ovf = ovf | (el.max() >= e_alloc) | (ol.max() >= e_obs)
        tm = start
        cost = cost + cost_step
        wsum = wsum + wm
        mksp = jnp.maximum(mksp, end)
        # 指令更新: actor(distributed_pcn.py:1579-1584)と同一演算列
        dr = (dr - rew.astype(jnp.float32)).astype(jnp.float32)
        if _COST_HOLD:
            # [anti-ration] cost目標(index1)を初期指令で一定保持(:1580-1581)。
            # eval 側(pcn_agent._run_episode :4370,4405)も同適用なので greedy でも掛ける。
            dr = dr.at[:, 1].set(dr0[:, 1])
        if _DR_UB is not None and not greedy:
            # [元PCN復元] 上限クリップ(:1582-1584)。torch では Actor 探索 rollout のみ適用・
            # eval(eval_mode=True)は不適用なので greedy では掛けない。
            dr = jnp.minimum(dr, jnp.float32(_DR_UB))
        hz = jnp.maximum(hz - 1.0, 1.0)

        def do_compact(op):
            es, ee, eh, oon, ob, ol, tm = op
            st, en, hh, c = _compact_count(es, ee, eh, suffnext)
            E = ee.shape[1]
            order = jnp.argsort(~(ee >= suffnext), axis=1, stable=True)
            pos_ok = jnp.arange(E)[None, :] < c[:, None]
            oon2 = jnp.take_along_axis(oon, order[:, :, None], axis=1) & pos_ok[:, :, None]
            ob2, ol2 = _compact_obs(ob, ol, tm - n_window)
            return st, en, hh, oon2, c, ob2, ol2

        def no_compact(op):
            es, ee, eh, oon, ob, ol, tm = op
            return es, ee, eh, oon, el, ob, ol

        es, ee, eh, oon, el, ob, ol = lax.cond(
            (t + 1) % kcompact == 0, do_compact, no_compact,
            (es, ee, eh, oon, ob, ol, tm))

        carry = (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf)
        return carry, (use_cloud.astype(jnp.int8), rew, obs)

    xs = (jnp.arange(T, dtype=jnp.int64), jq, arr_c, w_c, h_c, suff_c, ex_c)
    carry0 = (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr0, hz0, ovf)
    carry, (actions, rewards, obs_seq) = lax.scan(body, carry0, xs)
    (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf) = carry
    obs_last = obs_of(ob, ol, tm, jq_last, jnp.zeros((B,), dtype=jnp.int64),
                      jnp.zeros((_KEX,), dtype=jnp.float64) if _KEX else None)
    return cost, mksp, wsum, ovf, actions, rewards, obs_seq, obs_last


def run_fused_count_cmd(jobs, commands, *, params=None, ckpt_path=None, seed=0,
                        episode_id0=0, n_on=1520, n_cl=65536, n_window=100,
                        e_alloc=512, e_obs=1152, kcompact=64, greedy=False):
    """Phase3 指令付き rollout の O(N) 工場版。返り値は run_factory(commands=...) と同形式
    （episodes_to_transitions(mode="command") 互換; obs 付き）。

    greedy=True で argmax 相当の決定的行動（学習中 eval を O(N) 工場で回すため）。"""
    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    B = len(commands)
    if params is None:
        params = load_pcn_params(ckpt_path)
    arr = jobs[:, 0].astype(np.int64)
    w = jobs[:, 1].astype(np.int64)
    h = jobs[:, 2].astype(np.int64)
    suffmin = np.minimum.accumulate(arr[::-1])[::-1]
    suffnext = np.empty(T, dtype=np.int64)
    suffnext[:-1] = suffmin[1:]
    suffnext[-1] = np.iinfo(np.int64).max // 4
    h_max = int(h.max())
    jq_np = precompute_job_queue(jobs)
    jq = jnp.asarray(jq_np)
    jq_last = jnp.zeros((jq_np.shape[1],), dtype=jnp.float64)
    ex_np = job_extra_feats(jobs, n_window, n_on)         # (T,_KEX) 静的追加観測列
    jobs_cols = (jnp.asarray(arr), jnp.asarray(w), jnp.asarray(h), jnp.asarray(suffnext),
                 jnp.asarray(ex_np))
    dr0 = jnp.asarray(np.stack([np.asarray(c[0], np.float32) for c in commands]))
    hz0 = jnp.asarray(np.array([float(c[1]) for c in commands], np.float32))
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    cost, mksp, wsum, ovf, actions, rewards, obs_seq, obs_last = _fused_count_cmd_run(
        jobs_cols, params, key, jq, jq_last, dr0, hz0, n_on=n_on, h_max=h_max,
        n_window=n_window, norm_time=float(max(1, n_window)),
        norm_nodes=float(max(n_on, n_cl)), e_alloc=int(e_alloc), e_obs=int(e_obs),
        kcompact=int(kcompact), greedy=bool(greedy))
    jax.block_until_ready(actions)
    wall = time.perf_counter() - t0
    obs_all = np.concatenate(
        [np.asarray(obs_seq).transpose(1, 0, 2), np.asarray(obs_last)[:, None, :]], axis=1,
    ).astype(np.float32)
    return {
        "obs": obs_all,
        "actions": np.asarray(actions).T.astype(np.int8),
        "rewards": np.asarray(rewards).transpose(1, 0, 2),
        "achieved": np.stack([np.asarray(cost, np.float64), np.asarray(mksp, np.float64),
                              np.asarray(wsum, np.float64) / T], axis=1),
        "mode": "command",
        "commands0": np.stack([np.asarray(c[0], np.float32) for c in commands]),
        "episode_ids": np.arange(episode_id0, episode_id0 + B, dtype=np.int64),
        "seed0": int(seed),
        "stats": {"wall": float(wall)},
        "overflow": bool(np.asarray(ovf)),
    }
