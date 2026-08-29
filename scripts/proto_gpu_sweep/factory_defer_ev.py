"""defer(3行動) GPU 工場のイベント駆動カーネル版 (factory_defer_ev) — 大容量正容量で OOM しない。

背景: 旧 defer 工場(factory_defer + batch_env_jax)は occ(B,E,N) 密マスクと
候補×ノード格子(B,T,N)を作るため、4096ジョブ×正容量(22700/90800)で OOM した
(B=8 で 18.9GiB)。CPU env(event_native_env.py)自体はイベント駆動でノード×時刻格子を
持たない。本カーネルはその思想を JAX に正しく移植する。

== 厳密一致の判定根拠(torch env 読解) ==
env の配置規則(_find_event_allocation/_pick_free_from_count):
  候補時刻 = {arrival} ∪ {end>=arrival} を昇順に試し、窓 [t, t+w) と重なるイベントの
  占有ノード「和集合」に対して
    - onprem: 空きノード総数 >= h なら成立(連続 run 優先、無ければ昇順先頭 h 個=非連続可)
    - cloud : 連続 h ノードの空き run が必要(cont_only)
  ⇒ ノード同一性は結果(開始時刻)に影響する。乖離源は「窓が時間非重複(=ノード共有し得る)
  イベント対を跨ぐ」ときの 和集合 < 高さ合計。count 工場の既知近似(待ち~0.1%ずれ)の
  正体はまさにこの差(+cloud 弾性化)。
  ⇒ 占有ノード集合を「区間 RLE」(イベント毎の [lo,hi) 区間リスト)で持てば、ノード×時刻
  格子なし・イベント数オーダーのメモリで和集合サイズ・連続 run が厳密に計算できる
  (区間端点をノード座標でソート → 被覆カウント cumsum → セグメント長集計)。
  非連続割当の断片数は有限(weekAfull4096 実測: 非連続 7%・断片<=9)なので、
  区間プール + overflow フラグで厳密性を担保する(サイレント破綻なし)。

== カーネル 3 層構造(全候補×全ノードのテンソルを作らないための計算削減) ==
  [1] 全候補の安価な確実判定(O(T log E), ノード軸なし):
      free_lb = n - Σ h_e(窓重なり高さ合計)     … free_lb>=h ⇒ 確実に成立(和集合<=合計)
      free_ub = n - max時点被覆(窓内)           … free_ub<h  ⇒ 確実に不成立
      (時点被覆 = その時刻に生きているイベントの h 合計。時間重複イベントはノード互いに素
       (配置規則の不変量)なので 和集合 >= 任意時点の被覆。窓内 max は es グリッド上の
       sparse table range-max で O(1)/候補)
  [2] 曖昧候補(上下界の間)だけ厳密評価: 先頭 n_amb 個に限り区間端点 cumsum で
      和集合(onprem)/最長 run(cloud) を計算。答えは min(最初の厳密成立曖昧候補, 最初の
      確実成立候補)。曖昧候補が n_amb を超えかつ前半全滅の場合のみ ovf_amb フラグ
      (答えを誤らせず「不確実」を検知する保守的フラグ)。
  [3] 選択候補でのノード pick(単一候補・厳密): 連続 run 最低位 or 昇順空きノード
      greedy 充填(env の free_head[:h] と同一集合)。断片は隣接マージして区間化。

defer 機構(3行動 logits・ジョブ行回転・dcnt 上限・強制 onprem)・観測構築・指令更新は
factory_defer と同一(関数を直接 import して共有)。返り値も run_fused_defer と同形式。

イベントバッファ E(資源別)・区間プール P は静的で、kcompact ステップ毎に
生存(end >= 残ジョブ最小到着)前詰め圧縮。溢れは全て overflow フラグで検知:
  ovf_ev(イベントバッファ満杯) / ovf_pool(区間プール満杯) /
  ovf_frag(pick 断片数 > k_pick) / ovf_amb(曖昧候補 > n_amb で不確実)

usage: run_fused_defer(factory_defer) と同じ引数 + e_on/e_cl/pool/n_amb/k_pick/kcompact。
"""
from __future__ import annotations

import functools
import os
import sys
import time
from typing import NamedTuple

import numpy as np

_MAIN = "/home/noguchi/scheduler-sim-for-cb"
_MAIN_PROTO = os.path.join(_MAIN, "scripts", "proto_gpu_sweep")
if _MAIN_PROTO not in sys.path:
    sys.path.insert(0, _MAIN_PROTO)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from gpu_sweep_jax import INF64, _candidates  # noqa: E402
from factory_defer import (  # noqa: E402
    _KEX,
    _jq_from_state,
    _obs_from_buffer_jqb,
    _rot2,
    _rot3,
)
from factory_fused import (  # noqa: E402
    _COST_HOLD,
    _DR_UB,
    _OBS_BUDGET_RATIO,
    _OBS_EFFICIENCY,
    _OBS_OCC_PRIOR,
    _OBS_OCCUPANCY,
    _ORACLE_NPZ,
    BUD_A_K,
    BUD_COVER_K,
    BUD_FRAC_K,
    EFF_COST_K,
    EFF_GAIN_K,
    EFF_RATIO_K,
    _S_EMB_DROPOUT_P,
    _SLOWDOWN,
    _oracle_vals_np,
    load_pcn_params,
    pcn_logits,
)
from factory_jax import JOB_QUEUE_SIZE  # noqa: E402


# ---------------------------------------------------------------------------
# 状態(SoA pytree)。占有はノード区間 RLE: onprem=可変断片(プール), cloud=常に連続1区間。
# ---------------------------------------------------------------------------
class EvState(NamedTuple):
    # ジョブ列(defer 回転で書き換わる)
    jobs_w: jax.Array      # (B,N) i64
    jobs_h: jax.Array      # (B,N) i32
    jobs_arr: jax.Array    # (B,N) i64
    jobs_full: jax.Array   # (B,N,8) f64
    n_jobs: jax.Array      # (B,) i32
    # 進行
    idx: jax.Array         # (B,) i32
    time: jax.Array        # (B,) i64
    done: jax.Array        # (B,) bool
    # onprem イベント(区間はプール参照)
    es_on: jax.Array       # (B,Eo) i64 (無効 INF64)
    ee_on: jax.Array       # (B,Eo) i64 (無効 -1)
    eh_on: jax.Array       # (B,Eo) i32 (無効 0)
    el_on: jax.Array       # (B,) i32
    iv_lo: jax.Array       # (B,P) i32 (無効 lo=hi=n_on)
    iv_hi: jax.Array       # (B,P) i32
    iv_ev: jax.Array       # (B,P) i32 (親イベント slot)
    pl: jax.Array          # (B,) i32 プール使用数
    # cloud イベント(常に連続 [cl_lo, cl_lo+eh_cl))
    es_cl: jax.Array       # (B,Ec) i64
    ee_cl: jax.Array       # (B,Ec) i64
    eh_cl: jax.Array       # (B,Ec) i32
    el_cl: jax.Array       # (B,) i32
    cl_lo: jax.Array       # (B,Ec) i32 (無効 n_cl)
    # 観測・集計
    ev_obs: jax.Array      # (B,N,6) f64 追加順イベント6特徴(圧縮しない)
    cost_total: jax.Array  # (B,) i64
    waits: jax.Array       # (B,N) i64
    mksp: jax.Array        # (B,) i64  running max(end) — 圧縮でイベントを落とすため


def make_initial_state_ev(jobs: np.ndarray, *, n_on: int, n_cl: int,
                          e_on: int, e_cl: int, pool: int) -> EvState:
    jobs = np.asarray(jobs, dtype=np.float64)
    assert jobs.ndim == 3 and jobs.shape[2] == 8, f"jobs shape {jobs.shape}"
    B, N, _ = jobs.shape
    return EvState(
        jobs_w=jnp.asarray(jobs[:, :, 1], dtype=jnp.int64),
        jobs_h=jnp.asarray(jobs[:, :, 2], dtype=jnp.int32),
        jobs_arr=jnp.asarray(jobs[:, :, 0], dtype=jnp.int64),
        jobs_full=jnp.asarray(jobs),
        n_jobs=jnp.full((B,), N, dtype=jnp.int32),
        idx=jnp.zeros((B,), dtype=jnp.int32),
        time=jnp.zeros((B,), dtype=jnp.int64),
        done=jnp.zeros((B,), dtype=bool),
        es_on=jnp.full((B, e_on), INF64, dtype=jnp.int64),
        ee_on=jnp.full((B, e_on), -1, dtype=jnp.int64),
        eh_on=jnp.zeros((B, e_on), dtype=jnp.int32),
        el_on=jnp.zeros((B,), dtype=jnp.int32),
        iv_lo=jnp.full((B, pool), n_on, dtype=jnp.int32),
        iv_hi=jnp.full((B, pool), n_on, dtype=jnp.int32),
        iv_ev=jnp.zeros((B, pool), dtype=jnp.int32),
        pl=jnp.zeros((B,), dtype=jnp.int32),
        es_cl=jnp.full((B, e_cl), INF64, dtype=jnp.int64),
        ee_cl=jnp.full((B, e_cl), -1, dtype=jnp.int64),
        eh_cl=jnp.zeros((B, e_cl), dtype=jnp.int32),
        el_cl=jnp.zeros((B,), dtype=jnp.int32),
        cl_lo=jnp.full((B, e_cl), n_cl, dtype=jnp.int32),
        ev_obs=jnp.zeros((B, N, 6), dtype=jnp.float64),
        cost_total=jnp.zeros((B,), dtype=jnp.int64),
        waits=jnp.zeros((B, N), dtype=jnp.int64),
        mksp=jnp.zeros((B,), dtype=jnp.int64),
    )


# ---------------------------------------------------------------------------
# 層[1]: 全候補の安価な上下界(ノード軸なし O(T log E))
# ---------------------------------------------------------------------------
def _res_bounds(es, ee, eh, cand, w):
    """occ_sum(t)=窓重なり高さ合計 / maxpt(t)=窓内 max 時点被覆 を全候補で返す((B,T) i64)。

    occ_sum: 窓 [t,t+w) と重なる ⇔ es<t+w & ee>t。ee<=t の項は es<ee<=t<t+w で必ず
    es<t+w 側にも数えられているので、Σ h[es<t+w] − Σ h[ee<=t] が厳密に一致する。
    maxpt: 時点被覆 pc(q)=Σ h[es<=q<ee]。窓内 max は max(pc(t), max_{es∈(t,t+w)} pc(es))
    (被覆は es でのみ増えるので ee ブレークポイントは max を更新しない)。
    es グリッド上の pc に sparse table を張り range-max を O(1)/候補で引く。
    """
    B, E = es.shape
    bidx = jnp.arange(B)
    h64 = eh.astype(jnp.int64)
    t_end = cand + w[:, None]

    o_es = jnp.argsort(es, axis=1)
    es_s = jnp.take_along_axis(es, o_es, axis=1)
    pre_es = jnp.pad(jnp.cumsum(jnp.take_along_axis(h64, o_es, axis=1), axis=1),
                     ((0, 0), (1, 0)))                                   # (B,E+1)
    o_ee = jnp.argsort(ee, axis=1)
    ee_s = jnp.take_along_axis(ee, o_ee, axis=1)
    pre_ee = jnp.pad(jnp.cumsum(jnp.take_along_axis(h64, o_ee, axis=1), axis=1),
                     ((0, 0), (1, 0)))

    ss_l = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="left"))
    ss_r = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="right"))
    S1 = jnp.take_along_axis(pre_es, ss_l(es_s, t_end), axis=1)
    S2 = jnp.take_along_axis(pre_ee, ss_r(ee_s, cand), axis=1)
    occ_sum = S1 - S2                                                    # (B,T)

    # 時点被覆 pc を es グリッドで(同値 es は searchsorted 'right' で全数計上)
    pcA = jnp.take_along_axis(pre_es, ss_r(es_s, es_s), axis=1)
    pcB = jnp.take_along_axis(pre_ee, ss_r(ee_s, es_s), axis=1)
    pc = pcA - pcB                                                       # (B,E)

    # sparse table (B,L,E): tabs[k][i] = max(pc[i : i+2^k])
    tabs = [pc]
    d = 1
    while d < E:
        t_prev = tabs[-1]
        shifted = jnp.concatenate(
            [t_prev[:, d:], jnp.full((B, d), jnp.int64(-1))], axis=1)
        tabs.append(jnp.maximum(t_prev, shifted))
        d <<= 1
    tab = jnp.stack(tabs, axis=1)                                        # (B,L,E)
    n_lv = len(tabs)

    lo = ss_r(es_s, cand).astype(jnp.int32)                              # es > t の最初
    hi = ss_l(es_s, t_end).astype(jnp.int32)                             # es >= t+w の最初
    ln = hi - lo
    k = jnp.zeros_like(ln)
    for l in range(n_lv - 1):                                            # k = floor(log2(ln))
        k = k + (ln >= (1 << (l + 1))).astype(k.dtype)
    Ei = jnp.int32(E - 1)
    g1 = tab[bidx[:, None], k, jnp.clip(lo, 0, Ei)]
    g2 = tab[bidx[:, None], k, jnp.clip(hi - (1 << k).astype(jnp.int32), 0, Ei)]
    rngmax = jnp.where(ln > 0, jnp.maximum(g1, g2), jnp.int64(-1))

    pc_t = (jnp.take_along_axis(pre_es, ss_r(es_s, cand), axis=1)
            - jnp.take_along_axis(pre_ee, ss_r(ee_s, cand), axis=1))
    maxpt = jnp.maximum(pc_t, rngmax)
    return occ_sum, maxpt


# ---------------------------------------------------------------------------
# 層[2]/[3] 共通: 区間端点構造(ステップ毎 1 回) と 被覆プロファイル(候補毎)
# ---------------------------------------------------------------------------
def _iv_prep(ivlo, ivhi, n):
    """端点をノード座標で昇順ソート。セグメント s=0..M(M=2P):
    [seg_start[s], seg_end[s]) — s=0 は [0, pos_s[0]), s=M は [pos_s[M-1], n)。
    無効区間(lo=hi)は同座標 ±1 が相殺し、正長セグメントの被覆に一切影響しない。"""
    B, P = ivlo.shape
    pos = jnp.concatenate([ivlo, ivhi], axis=1)                          # (B,2P)
    order = jnp.argsort(pos, axis=1)
    pos_s = jnp.take_along_axis(pos, order, axis=1)
    dlt_s = jnp.where(order < P, jnp.int16(1), jnp.int16(-1))            # (B,2P)
    src = jnp.where(order < P, order, order - P)                         # 区間 index
    seg_end = jnp.concatenate(
        [pos_s, jnp.full((B, 1), jnp.int32(n))], axis=1)                 # (B,2P+1)
    seg_start = jnp.concatenate(
        [jnp.zeros((B, 1), jnp.int32), pos_s], axis=1)
    seglen = seg_end - seg_start
    return pos_s, dlt_s, src, seg_start, seg_end, seglen


def _cov_full(es_pt, ee_pt, dlt_s, t_sel, w):
    """被覆カウント(セグメント別) (B,C,M+1) i16。cov[s] = セグメント s 上の重なり区間数。

    es_pt/ee_pt はソート済み端点順に並べた親イベントの start/end (B,2P)
    (= take_along(es_iv, src); gather を候補軸の外に出して (B,C,P) 中間を作らない)。
    """
    act = (es_pt[:, None, :] < (t_sel + w[:, None])[:, :, None]) & (
        ee_pt[:, None, :] > t_sel[:, :, None])                           # (B,C,2P)
    cov = jnp.cumsum(act.astype(jnp.int16) * dlt_s[:, None, :], axis=2)
    return jnp.pad(cov, ((0, 0), (0, 0), (1, 0)))                        # (B,C,2P+1)


def _tail_free_bound(ivhi_eff, ee_iv, cand):
    """末尾ギャップ証明の材料: sufmax(t) = max{hi : 区間の親イベント ee > t}((B,T) i32)。

    ee > t の区間は窓と重なり得る区間の上位集合なので [sufmax(t), n) は窓内で確実に空き。
    n - sufmax(t) >= h なら末尾に連続 run があり確実に成立(cont_only 含む・sound)。
    ゼロ長(無効)区間は hi_eff=0 で不干渉。"""
    o = jnp.argsort(ee_iv, axis=1)
    ee_s = jnp.take_along_axis(ee_iv, o, axis=1)
    hi_s = jnp.take_along_axis(ivhi_eff, o, axis=1)
    suf = lax.cummax(hi_s[:, ::-1], axis=1)[:, ::-1]                     # (B,P)
    suf_pad = jnp.concatenate(
        [suf, jnp.zeros((suf.shape[0], 1), suf.dtype)], axis=1)
    idx = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="right"))(ee_s, cand)
    return jnp.take_along_axis(suf_pad, idx, axis=1)                     # (B,T)


# ---------------------------------------------------------------------------
# 資源 1 つの厳密配置(層[1]+[2]+[3])。cloud は cont_only=True(区間は常に 1 本)。
# ---------------------------------------------------------------------------
def _alloc_resource(es, ee, eh, ivlo, ivhi, ivev, w, h, arr, *,
                    n, cont_only, n_amb, k_pick):
    """返り値: start(B,)i64, frag_lo/hi(B,k_pick)i32, nfrag(B,)i32, ovf_amb, ovf_frag。

    env _find_event_allocation と同値(整数演算・タイブレーク一致):
    候補昇順の最初の成立時刻、連続 run 最低位 or 昇順空き greedy(非連続)。
    """
    B, E = es.shape
    T = E + 1
    P = ivlo.shape[1]
    bidx = jnp.arange(B)
    h64 = h.astype(jnp.int64)

    cand, valid = _candidates(es, ee, arr)                               # (B,T)
    occ_sum, maxpt = _res_bounds(es, ee, eh, cand, w)
    es_iv = es[bidx[:, None], ivev]                                      # (B,P)
    ee_iv = ee[bidx[:, None], ivev]
    ivhi_eff = jnp.where(ivhi > ivlo, ivhi, 0)
    sufmax = _tail_free_bound(ivhi_eff, ee_iv, cand)                     # (B,T)
    tail_ok = (jnp.int64(n) - sufmax.astype(jnp.int64)) >= h64[:, None]  # 末尾 run 証明
    if cont_only:
        cf = valid & ((occ_sum == 0) | tail_ok)  # 窓空 or 末尾ギャップ ⇒ run 成立(h<=n)
    else:
        cf = valid & (((n - occ_sum) >= h64[:, None]) | tail_ok)
    has_cf = cf.any(axis=1)
    t_c_idx = jnp.where(has_cf, jnp.argmax(cf, axis=1), T).astype(jnp.int32)
    poss = valid & ((n - maxpt) >= h64[:, None]) & (~cf)
    amb = poss & (jnp.arange(T, dtype=jnp.int32)[None, :] < t_c_idx[:, None])
    m_amb = amb.sum(axis=1)

    pos_s, dlt_s, src, seg_start, seg_end, seglen = _iv_prep(ivlo, ivhi, n)
    es_pt = jnp.take_along_axis(es_iv, src, axis=1)                      # (B,2P)
    ee_pt = jnp.take_along_axis(ee_iv, src, axis=1)
    amb_order = jnp.argsort(~amb, axis=1, stable=True)                   # 曖昧候補昇順

    def _amb_exact(C: int):
        """先頭 C 個の曖昧候補を厳密評価 → (first_ex(B,)i32, any_ex(B,))。"""
        sel = amb_order[:, :C].astype(jnp.int32)
        amb_sel = jnp.take_along_axis(amb, sel, axis=1)                  # (B,C)
        t_sel = jnp.take_along_axis(cand, sel, axis=1)
        cov = _cov_full(es_pt, ee_pt, dlt_s, t_sel, w)                   # (B,C,M+1)
        if cont_only:
            lastocc = lax.cummax(
                jnp.where(cov > 0, seg_end[:, None, :], jnp.int32(0)), axis=2)
            runlen = seg_end[:, None, :] - lastocc
            maxrun = jnp.max(jnp.where(cov == 0, runlen, 0), axis=2)     # (B,C)
            feas_ex = amb_sel & (maxrun >= h[:, None])
        else:
            u = jnp.einsum("bcm,bm->bc", (cov > 0).astype(jnp.float32),
                           seglen.astype(jnp.float32),
                           preferred_element_type=jnp.float32)           # 値<2^24 で厳密
            feas_ex = amb_sel & ((n - u.astype(jnp.int64)) >= h64[:, None])
        any_ex = feas_ex.any(axis=1)
        first_ex = jnp.where(
            any_ex,
            jnp.take_along_axis(sel, jnp.argmax(feas_ex, axis=1)[:, None],
                                axis=1)[:, 0],
            T,
        ).astype(jnp.int32)
        return first_ex, any_ex

    # 二段評価: まず C1=min(64, n_amb) だけ厳密評価。全 b が「成立を発見」or
    # 「曖昧候補を全数検査済み」なら確定(prefix 性質により C=n_amb の結果と同一)。
    # 未解決の b が 1 つでもあるステップだけ lax.cond で C=n_amb のフル評価に落ちる。
    # → 混雑 frontier 以外のステップで曖昧テンソル (B,C,M) の帯域を 1/8 に削減。
    C1 = min(64, n_amb)
    first1, any1 = _amb_exact(C1)
    if n_amb > C1:
        resolved = any1 | (m_amb <= C1)
        first_ex, any_ex = lax.cond(
            jnp.all(resolved),
            lambda _: (first1, any1),
            lambda _: _amb_exact(n_amb),
            None,
        )
    else:
        first_ex, any_ex = first1, any1
    t_star = jnp.minimum(first_ex, t_c_idx)                              # (B,)
    found = t_star < T
    # 曖昧候補が n_amb を超え、かつ検査した先頭 n_amb が全滅 → 未検査に成立があり得る
    ovf_amb = jnp.any((m_amb > n_amb) & (~any_ex))

    fb = jnp.maximum(arr, jnp.max(ee, axis=1))                           # h>n の保険のみ
    start = jnp.where(
        found, jnp.take_along_axis(cand, t_star[:, None], axis=1)[:, 0], fb)

    # --- 層[3]: 選択候補での厳密 pick(単一候補) ---
    cov0 = _cov_full(es_pt, ee_pt, dlt_s, start[:, None], w)[:, 0]       # (B,M+1)
    free0 = cov0 == 0
    lastocc0 = lax.cummax(jnp.where(~free0, seg_end, jnp.int32(0)), axis=1)
    runlen0 = seg_end - lastocc0
    okrun = free0 & (runlen0 >= h[:, None])
    has_run = okrun.any(axis=1)
    srun = jnp.argmax(okrun, axis=1)
    run_lo = lastocc0[bidx, srun]                                        # run 先頭 = 最低位

    kar = jnp.arange(k_pick, dtype=jnp.int32)
    if cont_only:
        # cloud は非連続 pick なし。found ⇒ run あり(層[2]の成立条件が run 存在)。
        lo_pick = jnp.where(found, run_lo, 0)
        frag_lo = jnp.where(kar[None, :] == 0, lo_pick[:, None], n).astype(jnp.int32)
        frag_hi = jnp.where(kar[None, :] == 0, lo_pick[:, None] + h[:, None], n
                            ).astype(jnp.int32)
        nfrag = jnp.ones((B,), dtype=jnp.int32)
        ovf_frag = jnp.zeros((), dtype=bool)
        return start, frag_lo, frag_hi, nfrag, ovf_amb, ovf_frag

    # onprem: 連続 run 優先、無ければ昇順空きノード greedy 充填(env free_head[:h] と同一)
    freelen = jnp.where(free0, seglen, 0)
    prior = jnp.cumsum(freelen, axis=1) - freelen                        # exclusive
    take = jnp.clip(h[:, None] - prior, 0, freelen)                      # (B,M+1)
    tk = take > 0
    full_prev = jnp.pad((tk & (take == seglen))[:, :-1], ((0, 0), (1, 0)))
    newf = tk & (~full_prev)                                             # 断片開始(隣接マージ)
    fid = jnp.cumsum(newf.astype(jnp.int32), axis=1) - 1
    ncfrag = jnp.where(tk, fid + 1, 0).max(axis=1)                       # 断片数
    fid_w = jnp.where(tk, fid, k_pick)                                   # 非採用→drop
    nc_lo = jnp.full((B, k_pick), jnp.int32(n)).at[
        bidx[:, None], fid_w].min(seg_start, mode="drop")
    nc_hi = jnp.zeros((B, k_pick), dtype=jnp.int32).at[
        bidx[:, None], fid_w].max(seg_start + take, mode="drop")

    use_run = found & has_run
    use_nc = found & (~has_run)
    ovf_frag = jnp.any(use_nc & (ncfrag > k_pick))

    run_frag_lo = jnp.where(kar[None, :] == 0, run_lo[:, None], n).astype(jnp.int32)
    run_frag_hi = jnp.where(kar[None, :] == 0, run_lo[:, None] + h[:, None], n
                            ).astype(jnp.int32)
    fb_lo = jnp.where(kar[None, :] == 0, 0, n).astype(jnp.int32)
    fb_hi = jnp.where(kar[None, :] == 0, h[:, None], n).astype(jnp.int32)
    frag_lo = jnp.where(use_run[:, None], run_frag_lo,
                        jnp.where(use_nc[:, None], nc_lo, fb_lo))
    frag_hi = jnp.where(use_run[:, None], run_frag_hi,
                        jnp.where(use_nc[:, None], nc_hi, fb_hi))
    nfrag = jnp.where(use_run, 1, jnp.where(use_nc, jnp.minimum(ncfrag, k_pick), 1))
    return start, frag_lo, frag_hi, nfrag, ovf_amb, ovf_frag


@functools.partial(jax.jit, static_argnames=("n", "cont_only", "n_amb", "k_pick"))
def ev_alloc_single(es, ee, eh, ivlo, ivhi, ivev, w, h, arr, *,
                    n, cont_only, n_amb, k_pick):
    """単発配置クエリ(検証用): _alloc_resource の jit ラッパ。"""
    return _alloc_resource(es, ee, eh, ivlo, ivhi, ivev, w, h, arr,
                           n=n, cont_only=cont_only, n_amb=n_amb, k_pick=k_pick)


# ---------------------------------------------------------------------------
# 融合 scan 本体(_fused_defer_run と同構造; env は上の厳密イベント駆動カーネル)
# ---------------------------------------------------------------------------
@functools.partial(
    jax.jit,
    static_argnames=("n_on", "n_cl", "n_window", "norm_time", "norm_nodes",
                     "t_scan", "defer_max", "defer_offset", "mode",
                     "e_on", "e_cl", "pool", "n_amb_on", "n_amb_cl", "k_pick",
                     "kcompact"),
)
def _fused_defer_ev_run(state0, dcnt0, params, key, probs, fixed_actions, dr0, hz0,
                        ora, *, n_on, n_cl, n_window, norm_time, norm_nodes,
                        t_scan, defer_max, defer_offset, mode,
                        e_on, e_cl, pool, n_amb_on, n_amb_cl, k_pick, kcompact):
    B, N = state0.jobs_w.shape
    bidx = jnp.arange(B)
    kar = jnp.arange(k_pick, dtype=jnp.int32)
    occ_all0 = (state0.jobs_w * state0.jobs_h.astype(jnp.int64)).astype(jnp.float64)
    # overflow フラグ 6 種: [ev_on, ev_cl, pool, frag, amb_on, amb_cl]
    ZOVF = jnp.zeros((6,), dtype=bool)

    def body(carry, xs):
        state, dcnt, dr, hz, steps, ovf = carry
        if mode == "fixed":
            t, fa_t = xs
        else:
            t = xs
        j = jnp.minimum(state.idx, N - 1)
        arr = state.jobs_arr[bidx, j]
        w = state.jobs_w[bidx, j]
        h = state.jobs_h[bidx, j]

        # ---- 配置探索(両資源; 本体 :161 と同値・urgency 観測共有) ----
        s_on, flo_on, fhi_on, nfr_on, oa_on, of_on = _alloc_resource(
            state.es_on, state.ee_on, state.eh_on,
            state.iv_lo, state.iv_hi, state.iv_ev, w, h, arr,
            n=n_on, cont_only=False, n_amb=n_amb_on, k_pick=k_pick)
        s_cl, flo_cl, fhi_cl, nfr_cl, oa_cl, of_cl = _alloc_resource(
            state.es_cl, state.ee_cl, state.eh_cl,
            state.cl_lo, state.cl_lo + state.eh_cl,
            jnp.broadcast_to(jnp.arange(e_cl, dtype=jnp.int32)[None, :], (B, e_cl)),
            w, h, arr, n=n_cl, cont_only=True, n_amb=n_amb_cl, k_pick=k_pick)

        pw = jnp.maximum(0, s_on - arr)
        jq_b = _jq_from_state(state.jobs_full, state.idx, state.n_jobs)
        extras = None
        if _KEX:
            active_x = state.idx < state.n_jobs
            ex_cols = []
            if _OBS_OCCUPANCY:
                w_j = state.jobs_w[bidx, j]
                h_j = state.jobs_h[bidx, j]
                if _OBS_OCC_PRIOR:
                    occ_frac = (w_j.astype(jnp.float64) / norm_time) * (
                        h_j.astype(jnp.float64) / max(1.0, float(n_on)))
                    v = jnp.clip(jnp.log1p(occ_frac * 1.0e3) / 4.3, 0.0, 1.0)
                else:
                    occ_j = (w_j * h_j.astype(jnp.int64)).astype(jnp.float64)
                    v = (occ_all0 <= occ_j[:, None]).sum(axis=1).astype(jnp.float64) / N
                    v = jnp.clip(v, 0.0, 1.0)
                ex_cols.append(jnp.where(active_x, v, 0.0))
            if _ORACLE_NPZ:
                ex_cols.append(jnp.where(active_x, ora[j], 0.0))
            if _OBS_EFFICIENCY:
                # event_c_env._front_job_efficiency と同一式:
                #   a = pt*nodes / b = max(0, s_on - s_cl) / 3列を log 正規化。
                a_e = (w * h.astype(jnp.int64)).astype(jnp.float64)
                b_e = jnp.maximum(0, s_on - s_cl).astype(jnp.float64)
                la_e = jnp.log1p(a_e)
                lb_e = jnp.log1p(b_e)
                ex_cols.append(jnp.where(
                    active_x, jnp.clip(la_e / EFF_COST_K, 0.0, 1.0), 0.0))
                ex_cols.append(jnp.where(
                    active_x, jnp.clip(lb_e / EFF_GAIN_K, 0.0, 1.0), 0.0))
                ex_cols.append(jnp.where(
                    active_x, jnp.clip((lb_e - la_e) / EFF_RATIO_K + 0.5, 0.0, 1.0), 0.0))
            if _OBS_BUDGET_RATIO:
                # event_c_env._front_job_budget_ratio と同一式。
                #   a_b    = 現ジョブの雲送りコスト(pt*nodes; 効率特徴のaと同一・答え非依存)
                #   budget = dr[:,1](残り予算; agentのG規約・anti-ration等の変換を経た「今の」値)
                #            を 0 未満クランプしたもの
                #   rem    = 現在位置 j(自分含む)以降の state.jobs_w/jobs_h(=defer回転後でも
                #            "現在" の並び)からの雲コスト合計。suffix sum を毎ステップ計算する
                #            ことで、ジョブ列がrotateしても position-dependent に正しく一致する。
                a_b = (w * h.astype(jnp.int64)).astype(jnp.float64)
                occ_row = (state.jobs_w.astype(jnp.float64)
                          * state.jobs_h.astype(jnp.float64))
                pos = jnp.arange(N, dtype=jnp.int32)[None, :]
                rem_mask = pos >= j[:, None]
                rem = jnp.sum(jnp.where(rem_mask, occ_row, 0.0), axis=1)
                budget = jnp.maximum(0.0, -dr[:, 1].astype(jnp.float64))
                rb1 = a_b / jnp.maximum(budget, 1.0)
                rb2 = a_b / jnp.maximum(rem, 1.0)
                rb3 = budget / jnp.maximum(rem, 1.0)
                ex_cols.append(jnp.where(
                    active_x, jnp.clip(jnp.log1p(rb1) / BUD_A_K, 0.0, 1.0), 0.0))
                ex_cols.append(jnp.where(
                    active_x, jnp.clip(jnp.log1p(rb2) / BUD_FRAC_K, 0.0, 1.0), 0.0))
                ex_cols.append(jnp.where(
                    active_x, jnp.clip(jnp.log1p(rb3) / BUD_COVER_K, 0.0, 1.0), 0.0))
            extras = jnp.stack(ex_cols, axis=1)
        obs = _obs_from_buffer_jqb(state.ev_obs, state.idx, state.time, jq_b, pw,
                                   n_window=n_window, norm_time=norm_time,
                                   norm_nodes=norm_nodes, extras=extras)
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

        # ---- defer 判定(本体 step :136-149 / factory_defer と同一) ----
        is_d = a_raw == 2
        can = (dcnt[bidx, j] < defer_max) & ((state.n_jobs - state.idx) > 1)
        do_defer = is_d & can & (~state.done)
        a01 = jnp.where(is_d, 0, a_raw)

        # ---- apply(配置反映; done/defer は active マスクで空回し) ----
        active = (~state.done) & (~do_defer)
        use_cloud = a01 == 1
        start = jnp.where(use_cloud, s_cl, s_on)
        wait = jnp.where(active, start - arr, 0)
        cost_step = jnp.where(active & use_cloud, w * h.astype(jnp.int64), 0)
        if _SLOWDOWN:
            wm = wait.astype(jnp.float64) / jnp.maximum(1.0, w.astype(jnp.float64))
            rewards = jnp.stack([-wm, -cost_step.astype(jnp.float64)], axis=1)
        else:
            rewards = jnp.stack([-wait, -cost_step], axis=1).astype(jnp.float64)
        rewards = rewards * active[:, None]
        end = start + w

        wr_on = active & (~use_cloud)
        wr_cl = active & use_cloud
        # onprem イベント + プール断片
        slot_o = state.el_on
        so_w = jnp.where(wr_on, slot_o, e_on)
        es_on = state.es_on.at[bidx, so_w].set(start, mode="drop")
        ee_on = state.ee_on.at[bidx, so_w].set(end, mode="drop")
        eh_on = state.eh_on.at[bidx, so_w].set(h, mode="drop")
        el_on = state.el_on + wr_on.astype(jnp.int32)
        posP = state.pl[:, None] + kar[None, :]
        vfrag = (fhi_on > flo_on) & wr_on[:, None] & (kar[None, :] < nfr_on[:, None])
        posP_w = jnp.where(vfrag, posP, pool)
        iv_lo = state.iv_lo.at[bidx[:, None], posP_w].set(flo_on, mode="drop")
        iv_hi = state.iv_hi.at[bidx[:, None], posP_w].set(fhi_on, mode="drop")
        iv_ev = state.iv_ev.at[bidx[:, None], posP_w].set(
            jnp.broadcast_to(slot_o[:, None], (B, k_pick)), mode="drop")
        pl = state.pl + jnp.where(wr_on, nfr_on, 0)
        # cloud イベント(常に 1 区間)
        slot_c = state.el_cl
        sc_w = jnp.where(wr_cl, slot_c, e_cl)
        es_cl = state.es_cl.at[bidx, sc_w].set(start, mode="drop")
        ee_cl = state.ee_cl.at[bidx, sc_w].set(end, mode="drop")
        eh_cl = state.eh_cl.at[bidx, sc_w].set(h, mode="drop")
        cl_lo = state.cl_lo.at[bidx, sc_w].set(flo_cl[:, 0], mode="drop")
        el_cl = state.el_cl + wr_cl.astype(jnp.int32)

        # overflow 検知(イベント/プール満杯・断片超過・曖昧超過; 種別フラグで報告)
        ovf = ovf | jnp.stack([
            el_on.max() >= e_on,
            el_cl.max() >= e_cl,
            jnp.any(wr_on & (state.pl + nfr_on > pool)),
            of_on | of_cl,
            oa_on,
            oa_cl,
        ])

        # 追加順イベント6特徴(本体 _event_buf.append と同値; start_node=最低位ノード)
        snode = jnp.where(use_cloud, flo_cl[:, 0], flo_on[:, 0])
        feat = jnp.stack(
            [start.astype(jnp.float64), end.astype(jnp.float64),
             w.astype(jnp.float64), use_cloud.astype(jnp.float64),
             snode.astype(jnp.float64), h.astype(jnp.float64)], axis=1)
        ev_obs = state.ev_obs.at[bidx, jnp.where(active, j, N), :].set(
            feat, mode="drop")

        waits = state.waits.at[bidx, jnp.where(active, j, N)].set(wait, mode="drop")
        cost_total = state.cost_total + cost_step
        mksp = jnp.maximum(state.mksp, jnp.where(active, end, 0))
        idx = state.idx + active.astype(jnp.int32)
        done = idx >= state.n_jobs
        time_ = jnp.where(active, start, state.time)

        st2 = state._replace(
            idx=idx, time=time_, done=done,
            es_on=es_on, ee_on=ee_on, eh_on=eh_on, el_on=el_on,
            iv_lo=iv_lo, iv_hi=iv_hi, iv_ev=iv_ev, pl=pl,
            es_cl=es_cl, ee_cl=ee_cl, eh_cl=eh_cl, el_cl=el_cl, cl_lo=cl_lo,
            ev_obs=ev_obs, cost_total=cost_total, waits=waits, mksp=mksp)

        # ---- defer 回転(factory_defer と同一) ----
        dcnt = dcnt.at[bidx, j].add(do_defer.astype(dcnt.dtype))
        jdst = jnp.minimum(state.idx + defer_offset, N - 1)
        posn = jnp.arange(N, dtype=jnp.int32)[None, :]
        in_shift = (posn >= state.idx[:, None]) & (posn < jdst[:, None])
        srcn = jnp.where(in_shift, posn + 1, posn)
        st2 = st2._replace(
            jobs_w=_rot2(st2.jobs_w, srcn, bidx, jdst, st2.jobs_w[bidx, j], do_defer),
            jobs_h=_rot2(st2.jobs_h, srcn, bidx, jdst, st2.jobs_h[bidx, j], do_defer),
            jobs_arr=_rot2(st2.jobs_arr, srcn, bidx, jdst, st2.jobs_arr[bidx, j],
                           do_defer),
            jobs_full=_rot3(st2.jobs_full, srcn, bidx, jdst, st2.jobs_full[bidx, j],
                            do_defer),
        )
        dcnt = _rot2(dcnt, srcn, bidx, jdst, dcnt[bidx, j], do_defer)

        # ---- 圧縮(kcompact 毎): 生存 end >= 残ジョブ最小到着 で前詰め ----
        def do_compact(st):
            mask = (posn >= st.idx[:, None]) & (posn < st.n_jobs[:, None])
            thr = jnp.min(jnp.where(mask, st.jobs_arr, INF64), axis=1)   # (B,)
            # onprem イベント
            al_o = st.ee_on >= thr[:, None]
            ord_o = jnp.argsort(~al_o, axis=1, stable=True)
            c_o = al_o.sum(axis=1).astype(jnp.int32)
            ok_o = jnp.arange(e_on)[None, :] < c_o[:, None]
            es2 = jnp.where(ok_o, jnp.take_along_axis(st.es_on, ord_o, axis=1), INF64)
            ee2 = jnp.where(ok_o, jnp.take_along_axis(st.ee_on, ord_o, axis=1), -1)
            eh2 = jnp.where(ok_o, jnp.take_along_axis(st.eh_on, ord_o, axis=1), 0)
            rank_o = (jnp.cumsum(al_o.astype(jnp.int32), axis=1) - 1)
            # プール(親イベント生存 & 正長のみ; iv_ev を新 slot へ再マップ)
            al_iv = (st.iv_hi > st.iv_lo) & jnp.take_along_axis(
                al_o, jnp.clip(st.iv_ev, 0, e_on - 1), axis=1)
            newev = jnp.take_along_axis(rank_o, jnp.clip(st.iv_ev, 0, e_on - 1),
                                        axis=1)
            ord_p = jnp.argsort(~al_iv, axis=1, stable=True)
            c_p = al_iv.sum(axis=1).astype(jnp.int32)
            ok_p = jnp.arange(pool)[None, :] < c_p[:, None]
            lo2 = jnp.where(ok_p, jnp.take_along_axis(st.iv_lo, ord_p, axis=1), n_on)
            hi2 = jnp.where(ok_p, jnp.take_along_axis(st.iv_hi, ord_p, axis=1), n_on)
            ev2 = jnp.where(ok_p, jnp.take_along_axis(newev, ord_p, axis=1), 0)
            # cloud イベント
            al_c = st.ee_cl >= thr[:, None]
            ord_c = jnp.argsort(~al_c, axis=1, stable=True)
            c_c = al_c.sum(axis=1).astype(jnp.int32)
            ok_c = jnp.arange(e_cl)[None, :] < c_c[:, None]
            es3 = jnp.where(ok_c, jnp.take_along_axis(st.es_cl, ord_c, axis=1), INF64)
            ee3 = jnp.where(ok_c, jnp.take_along_axis(st.ee_cl, ord_c, axis=1), -1)
            eh3 = jnp.where(ok_c, jnp.take_along_axis(st.eh_cl, ord_c, axis=1), 0)
            cl2 = jnp.where(ok_c, jnp.take_along_axis(st.cl_lo, ord_c, axis=1), n_cl)
            return st._replace(
                es_on=es2, ee_on=ee2, eh_on=eh2, el_on=c_o,
                iv_lo=lo2, iv_hi=hi2, iv_ev=ev2, pl=c_p,
                es_cl=es3, ee_cl=ee3, eh_cl=eh3, el_cl=c_c, cl_lo=cl2)

        st2 = lax.cond((t + 1) % kcompact == 0, do_compact, lambda s: s, st2)

        # ---- 指令更新(actor :1579-1584 / factory_defer と同一) ----
        dr2 = (dr - rewards.astype(jnp.float32)).astype(jnp.float32)
        if _COST_HOLD:
            dr2 = dr2.at[:, 1].set(dr0[:, 1])
        if _DR_UB is not None and mode != "greedy":
            dr2 = jnp.minimum(dr2, jnp.float32(_DR_UB))
        hz2 = jnp.where(active, jnp.maximum(hz - 1.0, 1.0), hz)
        steps2 = steps + (~state.done).astype(jnp.int32)
        return (st2, dcnt, dr2, hz2, steps2, ovf), (
            a_raw.astype(jnp.int8), rewards, obs.astype(jnp.float32))

    ts = jnp.arange(t_scan, dtype=jnp.int64)
    xs = (ts, fixed_actions) if mode == "fixed" else ts
    carry0 = (state0, dcnt0, dr0, hz0, jnp.zeros((B,), dtype=jnp.int32), ZOVF)
    (final, dcnt, dr, hz, steps, ovf), (actions, rewards, obs_seq) = lax.scan(
        body, carry0, xs)

    obs_last = _obs_from_buffer_jqb(
        final.ev_obs, final.idx, final.time,
        jnp.zeros((B, JOB_QUEUE_SIZE), dtype=jnp.float64),
        jnp.zeros((B,), dtype=jnp.int64),
        n_window=n_window, norm_time=norm_time, norm_nodes=norm_nodes,
        extras=jnp.zeros((B, _KEX), dtype=jnp.float32) if _KEX else None)
    cost = final.cost_total
    if _SLOWDOWN:
        wsum = -(rewards[:, :, 0].sum(axis=0))
    else:
        wsum = final.waits.sum(axis=1)
    mksp = final.mksp
    all_done = final.done.all()
    return cost, mksp, wsum, steps, all_done, ovf, actions, rewards, obs_seq, obs_last


def run_fused_defer_ev(jobs, commands=None, *, params=None, ckpt_path=None, probs=None,
                       fixed_actions=None, greedy=False, seed=0, episode_id0=0,
                       n_on=256, n_cl=1024, n_window=100,
                       defer_max=3, defer_offset=1, t_scan=None,
                       e_on=None, e_cl=None, pool=None, n_amb=None, n_amb_cl=None,
                       k_pick=None, kcompact=None):
    """イベント駆動 defer rollout。返り値は run_fused_defer と同形式 + overflow。

    overflow=True はバッファ溢れ/断片超過/曖昧候補超過のいずれかで「結果が torch env と
    一致しない可能性」を示す(該当バッファを env var で拡大して再実行)。
    静的形状(env var 既定):
      PCN_GPU_DEFER_E_ALLOC(既定 min(N+1,2560)) イベントバッファ(資源別)
      PCN_GPU_DEFER_POOL   (既定 2*e_on)        onprem 区間プール
      PCN_GPU_DEFER_NAMB   (既定 512)           厳密評価する曖昧候補数(onprem)
      PCN_GPU_DEFER_NAMB_CL(既定 512)           同(cloud; 混雑 frontier が広いため大きめ)
      PCN_GPU_DEFER_KPICK  (既定 16)            pick 断片上限
      PCN_GPU_DEFER_KCOMPACT(既定 64)           圧縮周期
    """
    jobs = np.asarray(jobs, dtype=np.float64)
    per_ep = jobs.ndim == 3
    T = int(jobs.shape[1] if per_ep else jobs.shape[0])
    if fixed_actions is not None:
        mode = "fixed"
        B = int(np.asarray(fixed_actions).shape[0])
    elif probs is not None:
        mode = "bernoulli"
        B = len(probs)
        if t_scan is None:
            t_scan = T
    else:
        mode = "greedy" if greedy else "sample"
        B = len(commands)
    if t_scan is None:
        t_scan = T * (1 + int(defer_max))
    if mode in ("sample", "greedy") and params is None:
        params = load_pcn_params(ckpt_path)

    def _env_int(name, default):
        v = os.environ.get(name, "").strip()
        return int(v) if v else int(default)

    if e_on is None:
        e_on = min(T + 1, _env_int("PCN_GPU_DEFER_E_ALLOC", 2560))
    if e_cl is None:
        e_cl = min(T + 1, _env_int("PCN_GPU_DEFER_E_ALLOC", 2560))
    if pool is None:
        pool = _env_int("PCN_GPU_DEFER_POOL", 2 * int(e_on))
    if n_amb is None:
        n_amb = _env_int("PCN_GPU_DEFER_NAMB", 512)
    if n_amb_cl is None:
        n_amb_cl = _env_int("PCN_GPU_DEFER_NAMB_CL", 512)
    if k_pick is None:
        k_pick = _env_int("PCN_GPU_DEFER_KPICK", 16)
    if kcompact is None:
        kcompact = _env_int("PCN_GPU_DEFER_KCOMPACT", 64)
    n_amb = min(int(n_amb), int(e_on) + 1)
    n_amb_cl = min(int(n_amb_cl), int(e_cl) + 1)

    if per_ep and int(jobs.shape[0]) != B:
        raise ValueError(f"per-episode jobs の先頭次元 {jobs.shape[0]} != バッチ数 {B}")
    jobs_b = jobs if per_ep else np.broadcast_to(jobs, (B, T, 8))
    state0 = make_initial_state_ev(jobs_b, n_on=n_on, n_cl=n_cl,
                                   e_on=int(e_on), e_cl=int(e_cl), pool=int(pool))
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
              if fixed_actions is not None
              else jnp.zeros((int(t_scan), B), dtype=jnp.int32))
    params_in = params if params is not None else {}
    key = jax.random.PRNGKey(seed)
    ora_dev = jnp.asarray(_oracle_vals_np(T) if _ORACLE_NPZ
                          else np.zeros(T, dtype=np.float64))

    t0 = time.perf_counter()
    (cost, mksp, wsum, steps, all_done, ovf, actions, rewards, obs_seq,
     obs_last) = _fused_defer_ev_run(
        state0, dcnt0, params_in, key, probs_dev, fa_dev, dr0, hz0, ora_dev,
        n_on=n_on, n_cl=n_cl, n_window=n_window,
        norm_time=float(max(1, n_window)), norm_nodes=float(max(n_on, n_cl)),
        t_scan=int(t_scan), defer_max=int(defer_max),
        defer_offset=int(defer_offset), mode=mode,
        e_on=int(e_on), e_cl=int(e_cl), pool=int(pool), n_amb_on=int(n_amb),
        n_amb_cl=int(n_amb_cl), k_pick=int(k_pick), kcompact=int(kcompact))
    jax.block_until_ready(actions)
    wall = time.perf_counter() - t0
    flags = np.asarray(ovf)
    ovf_detail = {k: bool(v) for k, v in zip(
        ("ev_on", "ev_cl", "pool", "frag", "amb_on", "amb_cl"), flags)}

    obs_all = np.concatenate(
        [np.asarray(obs_seq).transpose(1, 0, 2), np.asarray(obs_last)[:, None, :]],
        axis=1,
    ).astype(np.float32)
    res = {
        "obs": obs_all,
        "actions": np.asarray(actions).T.astype(np.int8),
        "rewards": np.asarray(rewards).transpose(1, 0, 2),
        "achieved": np.stack([np.asarray(cost, np.float64),
                              np.asarray(mksp, np.float64),
                              np.asarray(wsum, np.float64) / T], axis=1),
        "lengths": np.asarray(steps, np.int32),
        "mode": {"sample": "command", "greedy": "command"}.get(mode, mode),
        "episode_ids": np.arange(episode_id0, episode_id0 + B, dtype=np.int64),
        "seed0": int(seed),
        "stats": {"wall": float(wall),
                  "e_on": int(e_on), "e_cl": int(e_cl), "pool": int(pool),
                  "n_amb": int(n_amb), "n_amb_cl": int(n_amb_cl),
                  "k_pick": int(k_pick), "kcompact": int(kcompact),
                  "ovf": ovf_detail, "all_done": bool(np.asarray(all_done))},
        "overflow": bool(flags.any()) or not bool(np.asarray(all_done)),
    }
    if commands is not None:
        res["commands0"] = np.stack([np.asarray(c[0], np.float32) for c in commands])
    if probs is not None:
        res["probs"] = np.asarray(probs, dtype=np.float32)
    return res
