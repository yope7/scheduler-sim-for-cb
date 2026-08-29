"""sweep-line 配置探索の GPU バッチ実装（JAX / XLA, C 版 event_sweep_alloc と出力一致）。

B 問題を 1 バッチで解く。設計:
  - パディング: イベント E_max・候補時刻 T=E_max+1 に固定シェイプ化。無効イベントは
    (start=INF64, end=-1) で overlap 条件から自然に排除（end > t が偽; t >= arrival >= 0）。
    無効候補は BIG(2^60) パディング → sort 後尾に集まり valid_cand マスクで除外。
    候補の重複はユニーク化しない（同じ t を二度試しても最初の成立 t は不変 = 結果同一で、
    ソートだけで済む）。
  - 占有カウント: overlap(B,T,E) と occ(B,E,N) の fp16 行列積で count(B,T,N) を構築
    （tensor core; 非負値の和が 0 ⇔ 全項 0 なので free 判定はビット正確）。
  - 連続 run 探索: occupied の累積和 cs について window 和 cs[n+h]-cs[n]==0 が
    「n から h 連続 free」。argmax で最低位開始（C 版と同じ最低位優先）。
  - 最初に成立した候補時刻 = feasible(B,T) の時刻昇順 argmax。
  - 非連続 pick / フォールバックもマスク選択（C 版と同じ最低位 height 個 / max-end + range）。

C 版とのタイブレーク一致の要点: 候補は昇順、run は node 0 から最低位、非連続 pick は
free ノードの昇順先頭 height 個、フォールバック start = max(arrival, max(ends))。
"""
from __future__ import annotations

import functools
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)  # 時刻は int64（C 版と同一型）

import jax.numpy as jnp
import numpy as np

INF64 = np.int64(1) << 62  # 無効イベント start
BIG = np.int64(1) << 60    # 無効候補時刻（BIG + width が INF64 未満に収まる関係を維持）


@functools.partial(jax.jit, static_argnames=("h_max",))
def _sweep_alloc_kernel(starts, ends, occ, width, height, arrival, cont_only, *, h_max):
    """v1: 占有カウントを overlap(B,T,E) × occ(B,E,N) の fp16 行列積で構築（O(T·E·N)）。

    starts/ends:(B,E) int64, occ:(B,E,N) bool, width/arrival:(B,) int64,
    height:(B,) int32, cont_only:(B,) bool。
    返り値: start(B,) int64, is_contig(B,) bool, nodes(B,h_max) int32, found(B,), t_idx(B,)。
    """
    cand, valid_cand = _candidates(starts, ends, arrival)

    # ---- overlap(B,T,E): starts < t+width かつ ends > t ----
    t_end = cand + width[:, None]                                      # (B,T)
    ov = (starts[:, None, :] < t_end[:, :, None]) & (ends[:, None, :] > cand[:, :, None])

    # ---- 占有カウント → free(B,T,N)。fp16 積和（非負値の和==0 判定は丸めの影響を受けない） ----
    cnt = jnp.einsum(
        "bte,ben->btn",
        ov.astype(jnp.float16),
        occ.astype(jnp.float16),
        preferred_element_type=jnp.float32,
    )
    free = cnt == 0.0                                                  # (B,T,N)
    return _finish_from_free(free, cand, valid_cand, ends, height, arrival, cont_only, h_max)


@functools.partial(jax.jit, static_argnames=("h_max",))
def _sweep_alloc_kernel_v1h(starts, ends, occ, width, height, arrival, cont_only, *, h_max):
    """v1h: v1 の cnt 蓄積を fp32→fp16 に（count(B,T,N) の書き出し帯域半減）。

    count は整数値で E ≤ 2048 なら fp16 で厳密（和の各段が整数のまま表現可能）。
    使うのは free = cnt==0 の bool だけなので v1 と結果は完全一致
    （verify_batch_env.py KERNEL=v1h で全ビット一致を確認済みにすること）。
    Phase 3.5 の工場用。eval 経路（batch_eval_jax）は従来どおり v1 を使う。
    """
    cand, valid_cand = _candidates(starts, ends, arrival)
    t_end = cand + width[:, None]
    ov = (starts[:, None, :] < t_end[:, :, None]) & (ends[:, None, :] > cand[:, :, None])
    cnt = jnp.einsum(
        "bte,ben->btn",
        ov.astype(jnp.float16),
        occ.astype(jnp.float16),
        preferred_element_type=jnp.float16,
    )
    free = cnt == jnp.float16(0.0)
    return _finish_from_free(free, cand, valid_cand, ends, height, arrival, cont_only, h_max)


@functools.partial(jax.jit, static_argnames=("h_max",))
def _sweep_alloc_kernel_v2(starts, ends, occ, width, height, arrival, cont_only, *, h_max):
    """v2: 占有カウントを「イベント→候補区間の差分 scatter + 時刻軸 cumsum」で構築。

    イベント e が候補 t と重なる ⇔ starts_e-width < t < ends_e ⇔ t_idx ∈ [lo_e, hi_e)
    （候補列は昇順なので連続区間）。delta(B,T+1,N) に occ_e を lo で +1 / hi で -1 して
    時刻軸 cumsum → count。演算量 O((E+T)·N)（v1 の O(T·E·N) から E 倍削減）で、
    C 版の増分 sweep と同じオーダーを GPU バッチで実現する。count は整数なので free 判定は正確。
    無効イベントは occ 全 False 前提（scatter しても 0 なので無害）。
    """
    B, E = starts.shape
    N = occ.shape[2]
    cand, valid_cand = _candidates(starts, ends, arrival)
    T = cand.shape[1]

    # overlap する候補 index 区間 [lo, hi): lo = 最初の cand > starts-width, hi = 最初の cand >= ends
    lo = jax.vmap(lambda c, v: jnp.searchsorted(c, v, side="right"))(
        cand, starts - width[:, None]
    )                                                                  # (B,E)
    hi = jax.vmap(lambda c, v: jnp.searchsorted(c, v, side="left"))(cand, ends)
    lo = jnp.minimum(lo, hi)  # end<=start 等で区間が空(lo>hi)なら +1/-1 を同位置で相殺

    occ32 = occ.astype(jnp.int32)  # int16 は GPU に atomicAdd がなく scatter が遅いパスに落ちる
    bidx = jnp.broadcast_to(jnp.arange(B)[:, None], (B, E))
    delta = jnp.zeros((B, T + 1, N), dtype=jnp.int32)
    delta = delta.at[bidx, lo].add(occ32)
    delta = delta.at[bidx, hi].add(-occ32)
    cnt = jnp.cumsum(delta, axis=1)[:, :T]                             # (B,T,N) int32
    free = cnt == 0
    return _finish_from_free(free, cand, valid_cand, ends, height, arrival, cont_only, h_max)


@functools.partial(jax.jit, static_argnames=("h_max",))
def _sweep_alloc_kernel_v3(starts, ends, occ, width, height, arrival, cont_only, *, h_max):
    """v3: v2 と同じ O((E+T)·N) を XLA scatter なしで実現（sort + prefix-sum + gather）。

    cnt[b,t,n] = Σ_e occ[e,n]·[lo_e <= t < hi_e]
               = P_lo[b,n,#(lo<=t)] - P_hi[b,n,#(hi<=t)]
    （P_* はそれぞれ lo / hi 昇順に並べたイベントの占有 prefix-sum）。
    全段が gather/cumsum（cumsum は最終軸）で、XLA の遅い scatter を踏まない。
    count は整数なので free 判定は正確。無効イベントは occ 全 False 前提。
    """
    B, E = starts.shape
    N = occ.shape[2]
    cand, valid_cand = _candidates(starts, ends, arrival)
    T = cand.shape[1]

    # overlap する候補 index 区間 [lo, hi)（v2 と同一）
    lo = jax.vmap(lambda c, v: jnp.searchsorted(c, v, side="right"))(
        cand, starts - width[:, None]
    ).astype(jnp.int32)                                                # (B,E)
    hi = jax.vmap(lambda c, v: jnp.searchsorted(c, v, side="left"))(
        cand, ends
    ).astype(jnp.int32)
    lo = jnp.minimum(lo, hi)  # 空区間 (end<=start 等) を無効化

    occ_t = occ.transpose(0, 2, 1).astype(jnp.int16)                   # (B,N,E)
    t_range = jnp.arange(T, dtype=jnp.int32)

    def prefix_at_t(keys):
        """keys(B,E) 昇順ソート順の occ prefix-sum を各候補 t で引く → (B,N,T)。"""
        order = jnp.argsort(keys, axis=1)                              # (B,E)
        keys_s = jnp.take_along_axis(keys, order, axis=1)
        occ_s = jnp.take_along_axis(occ_t, order[:, None, :], axis=2)  # (B,N,E)
        p = jnp.pad(jnp.cumsum(occ_s, axis=2), ((0, 0), (0, 0), (1, 0)))  # (B,N,E+1)
        k = jax.vmap(lambda ks: jnp.searchsorted(ks, t_range, side="right"))(keys_s)  # (B,T)
        return jnp.take_along_axis(p, k[:, None, :], axis=2)           # (B,N,T)

    cnt_t = prefix_at_t(lo) - prefix_at_t(hi)                          # (B,N,T) int16
    free = (cnt_t == 0).transpose(0, 2, 1)                             # (B,T,N)
    return _finish_from_free(free, cand, valid_cand, ends, height, arrival, cont_only, h_max)


def _candidates(starts, ends, arrival):
    """候補時刻 = {arrival} ∪ {ends >= arrival} を昇順に（無効は BIG で後置・重複はそのまま）。"""
    cand_ev = jnp.where(ends >= arrival[:, None], ends, BIG)           # (B,E)
    cand = jnp.sort(jnp.concatenate([arrival[:, None], cand_ev], 1), 1)  # (B,T)
    return cand, cand < BIG


def _finish_from_free(free, cand, valid_cand, ends, height, arrival, cont_only, h_max):
    """free(B,T,N) から run 探索・成立判定・ノード選択・フォールバックまで（v1/v2 共通）。"""
    B, _, N = free.shape
    n_idx = jnp.arange(N, dtype=jnp.int32)

    # ---- 連続 run 探索: cs[n+h]-cs[n]==0 ⇔ [n,n+h) 全 free。argmax=最低位開始 ----
    # int16(和は最大 N<32767 で正確)+ take_along_axis の暗黙 broadcast で帯域半減（実測 2.2x）
    cs = jnp.cumsum((~free).astype(jnp.int16), axis=2)                 # (B,T,N)
    cs_pad = jnp.pad(cs, ((0, 0), (0, 0), (1, 0)))                     # (B,T,N+1)
    idx_hi = jnp.minimum(n_idx[None, :] + height[:, None], N)          # (B,N)
    win = jnp.take_along_axis(cs_pad, idx_hi[:, None, :], axis=2) - cs_pad[:, :, :N]
    in_range = (n_idx[None, :] + height[:, None]) <= N                 # (B,N)
    ok_run = (win == 0) & in_range[:, None, :]                         # (B,T,N)
    has_run = ok_run.any(axis=2)                                       # (B,T)
    run_start = jnp.argmax(ok_run, axis=2).astype(jnp.int32)           # (B,T) 最低位

    # ---- 成立判定と「最初に成立した候補時刻」 ----
    # cs は int16。N >= 32768 (例: 正容量 cloud 65536/90800) では N - cs を int16 のまま
    # 計算すると折り返して free_cnt が負になり全候補 infeasible → フォールバック配置
    # (待ち爆発)。int32 へ昇格してから引く。N < 32768 では値不変 = 従来 bit 一致。
    free_cnt = jnp.int32(N) - cs[:, :, -1].astype(jnp.int32)           # (B,T)
    feasible = (
        (free_cnt >= height[:, None])
        & (has_run | ~cont_only[:, None])
        & valid_cand
    )
    t_idx = jnp.argmax(feasible, axis=1)                               # (B,) 最早
    found = feasible.any(axis=1)

    take_t = lambda a: jnp.take_along_axis(a, t_idx[:, None], axis=1)[:, 0]
    start_sel = take_t(cand)
    run_sel = take_t(run_start)
    has_run_sel = take_t(has_run)
    free_sel = jnp.take_along_axis(
        free, t_idx[:, None, None].repeat(N, 2), axis=1
    )[:, 0]                                                            # (B,N)

    j = jnp.arange(h_max, dtype=jnp.int32)

    # ---- ノード選択 ----
    contig_nodes = run_sel[:, None] + j[None, :]                       # 連続 run
    key = jnp.where(free_sel, n_idx, N + n_idx)                        # 非連続: 最低位 free 順
    sorted_key = jnp.sort(key, axis=1)                                 # (B,N)
    if h_max > N:  # height > n_nodes は必ずフォールバック側なので中身は不問（形状合わせのみ）
        pad_tail = jnp.broadcast_to(
            jnp.arange(N, h_max, dtype=jnp.int32)[None, :], (B, h_max - N)
        )
        sorted_key = jnp.concatenate([sorted_key, pad_tail], axis=1)
    noncontig_nodes = sorted_key[:, :h_max].astype(jnp.int32)
    fb_nodes = jnp.broadcast_to(j[None, :], (B, h_max))                # フォールバック range(h)

    # 非連続 pick の実連続性（C 版: j<height の隣接のみ検査）
    adj_ok = (noncontig_nodes[:, 1:] == noncontig_nodes[:, :-1] + 1) | (
        j[None, 1:] >= height[:, None]
    )
    noncontig_contig = adj_ok.all(axis=1)

    # ---- フォールバック: start = max(arrival, max ends), nodes=range(height) ----
    fb_start = jnp.maximum(arrival, jnp.max(ends, axis=1))             # 無効 end=-1 は無害

    use_run = found & has_run_sel
    use_nc = found & ~has_run_sel
    nodes = jnp.where(
        use_run[:, None], contig_nodes, jnp.where(use_nc[:, None], noncontig_nodes, fb_nodes)
    )
    start = jnp.where(found, start_sel, fb_start)
    is_contig = jnp.where(use_nc, noncontig_contig, True)
    return start, is_contig, nodes, found, t_idx


def solve_batch_gpu(
    batch, chunk: int | None = None, h_max: int | None = None, kernel: str = "v1"
):
    """gen_problems.gen_batch のバッチを GPU で解く（B をチャンク分割して VRAM 制御）。

    kernel: "v1"=行列積 count（O(T·E·N)） / "v2"=差分 scatter+cumsum（O((E+T)·N)）。
    返り値: (start(B,) int64, is_contig(B,) bool, nodes(B,h_max) int32) の numpy。
    """
    B, E = batch["starts"].shape
    N = int(batch["n_nodes"])
    kfn = {"v1": _sweep_alloc_kernel, "v2": _sweep_alloc_kernel_v2,
           "v3": _sweep_alloc_kernel_v3}[kernel]
    if h_max is None:
        h_max = int(batch["height"].max())
    if chunk is None:
        chunk = default_chunk(E, N)
    chunk = min(chunk, B)
    cont = np.full(B, bool(batch["continuous_only"]))
    outs = []
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        args = _pad_chunk(batch, cont, lo, hi, chunk)
        s, c, nd, _, _ = kfn(*args, h_max=h_max)
        take = hi - lo
        outs.append((np.asarray(s)[:take], np.asarray(c)[:take], np.asarray(nd)[:take]))
    return (
        np.concatenate([o[0] for o in outs]),
        np.concatenate([o[1] for o in outs]),
        np.concatenate([o[2] for o in outs]),
    )


def default_chunk(E: int, N: int) -> int:
    """(B,T,E)+(B,T,N) 系テンソルが数 GB に収まるチャンク幅。"""
    t = E + 1
    bytes_per_b = t * (E + 3 * N) * 4  # cnt/cs/win int32/fp 系の概算
    budget = 6 << 30
    return max(32, min(4096, int(budget // bytes_per_b)))


def _pad_chunk(batch, cont, lo, hi, chunk):
    """チャンクを固定シェイプ chunk へパディング（末尾は無害なダミー問題）。"""
    sl = slice(lo, hi)
    take = hi - lo
    pad = chunk - take

    def pad0(a, fill):
        if pad == 0:
            return a[sl]
        shape = (pad,) + a.shape[1:]
        return np.concatenate([a[sl], np.full(shape, fill, dtype=a.dtype)])

    starts = pad0(batch["starts"], INF64)
    ends = pad0(batch["ends"], -1)
    occ = (
        batch["occ"][sl]
        if pad == 0
        else np.concatenate(
            [batch["occ"][sl], np.zeros((pad,) + batch["occ"].shape[1:], dtype=bool)]
        )
    )
    width = pad0(batch["width"], 1)
    height = pad0(batch["height"], 1).astype(np.int32)
    arrival = pad0(batch["arrival"], 0)
    cont_p = pad0(cont, False)
    return (
        jnp.asarray(starts),
        jnp.asarray(ends),
        jnp.asarray(occ),
        jnp.asarray(width),
        jnp.asarray(height),
        jnp.asarray(arrival),
        jnp.asarray(cont_p),
    )
