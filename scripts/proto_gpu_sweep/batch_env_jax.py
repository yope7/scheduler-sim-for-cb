"""フルenv バッチ化 Phase 1: SchedulingEnvEventNative の step ループ全体
（配置探索・報酬・イベント蓄積）を B エピソード同時に進める JAX 実装。

対象: src/envs/scheduling_variants/event_native_env.py の step() :120-186。
obs 構築（221次元観測）は Phase 2 スコープ。ただし Phase 2 が必要とする情報
（ジョブ生配列 / 追加順イベント6特徴 / time / idx）は state に保持済み。

設計:
  - 状態は SoA の pytree（EnvState）。イベントは資源別（0=onprem, 1=cloud）に
    固定長 E_max でパディング（無効: start=INF64, end=-1, occ=全False。
    前フェーズカーネルの規約と同一で、overlap/候補の両条件から自然に排除される）。
  - prune（event_native_env.py :244-251 の間引き）はしない＝全イベント保持。
    prune は「結果に影響しない候補削減」であり、一致検証（verify_batch_env.py）で
    prune 有り本体env と全ステップ一致することを確認する。
  - 配置探索は前フェーズの勝者カーネル gpu_sweep_jax._sweep_alloc_kernel（v1,
    C 版 event_sweep_alloc と全ビット一致検証済み）を onprem/cloud で 2 回呼び、
    action で選択する。cloud 側判定が不要でも onprem 側は Phase 2 の urgency 観測
    （毎ステップ先頭ジョブの onprem 配置クエリ）と共有できるため、この形が自然。
  - step は jit + lax.scan 可能: step(state, actions(B,)) -> (state', StepOut)。
    エピソード非同期終了は done マスクで空回し（本体 step :121-123 と同じ返り値規約）。
  - B 本は「同一ジョブ列×異なる行動列」（evalユース）も「異なるジョブ列」（工場ユース）も
    jobs(B,N,8) の与え方だけで両対応。
  - 時刻/待ち/コストは全て整数（int64）演算 = 本体（Python int）と厳密一致可能。
    rewards は本体と同じ float64（整数値のキャストなので 2^53 未満で厳密一致）。

defer（action=2, 今回OFF）を後から足す場所:
  - step 冒頭で is_defer = actions == 2 を分岐し、jobs_w/h/arr/full の
    [idx, idx+offset] 間ローテーション + defer_counts(B,N) の増分 + cap 到達時の
    強制 onprem 化を実装する（本体 :128-141）。イベント配列・配置カーネルは不変。
    jobs_* が step 内で書き換わるようになるだけで SoA 設計はそのまま使える。

wait_metric は既定 "wait" のみ実装。slowdown は wmetric = wait / max(1, pt) を
rewards[0] の生成箇所で置き換えれば対応可（本体 :169-172）。
"""
from __future__ import annotations

import functools
import os
import sys
from typing import NamedTuple

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np

# [SCHEDULER_WAIT_METRIC=slowdown] 本体 event_native_env.step :177-180 と同じ
# wmetric = wait / max(1, pt) を rewards[0] に使う（既定 "wait" では従来どおり=bit不変）。
_SLOWDOWN = os.environ.get("SCHEDULER_WAIT_METRIC", "wait") == "slowdown"

from gpu_sweep_jax import (  # noqa: E402  (jax x64 有効化もここで行われる)
    INF64,
    _sweep_alloc_kernel,
    _sweep_alloc_kernel_v1h,
    _sweep_alloc_kernel_v2,
    _sweep_alloc_kernel_v3,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

_KERNELS = {
    "v1": _sweep_alloc_kernel,
    "v1h": _sweep_alloc_kernel_v1h,  # cnt fp16 版（free 判定不変=結果一致; 工場用）
    "v2": _sweep_alloc_kernel_v2,
    "v3": _sweep_alloc_kernel_v3,
}


class EnvState(NamedTuple):
    """B エピソード分の環境状態（SoA pytree）。

    ジョブ列（defer 無しでは不変）:
      jobs_w   (B,N) int64   処理時間 pt（=イベント幅 width; 本体 job[0]）
      jobs_h   (B,N) int32   必要ノード数（=height; 本体 job[1]）
      jobs_arr (B,N) int64   到着時刻（本体 raw_job[0]）
      jobs_full(B,N,8) f64   生ジョブ8列（Phase 2 の job_queue 観測用に保持）
      n_jobs   (B,)  int32   実ジョブ数（全 B 同一なら N）
    進行状態:
      idx      (B,)  int32   index_next_job
      time     (B,)  int64   本体 self.time（= 直近配置の start_time; 観測窓の基準）
      done     (B,)  bool
    イベント（資源別 r: 0=onprem, 1=cloud）:
      ev_start (B,2,E) int64  無効は INF64
      ev_end   (B,2,E) int64  無効は -1
      occ_on   (B,E,N_on) bool  onprem 占有 dense マスク（非連続割当がありうる）
      occ_cl   (B,E,N_cl) bool  cloud 占有 dense マスク（実際は常に連続矩形だが
                                 v1 カーネルの入力形式に合わせ dense で保持）
      ev_len   (B,2) int32    資源別有効イベント数
      ev_obs   (B,N,6) f64    追加順イベント6特徴 (start,end,width,use_cloud,
                              start_node,height) = 本体 _event_buf 相当（Phase 2 の
                              イベント観測 30×6 の材料。今回は書き込みのみ）
    集計:
      cost_total (B,) int64
      waits      (B,N) int64  ジョブ位置別 waiting_time（avgwait 計算・検証用）
    """

    jobs_w: jax.Array
    jobs_h: jax.Array
    jobs_arr: jax.Array
    jobs_full: jax.Array
    n_jobs: jax.Array
    idx: jax.Array
    time: jax.Array
    done: jax.Array
    ev_start: jax.Array
    ev_end: jax.Array
    occ_on: jax.Array
    occ_cl: jax.Array
    ev_len: jax.Array
    ev_obs: jax.Array
    cost_total: jax.Array
    waits: jax.Array


class StepOut(NamedTuple):
    """1 step の出力（本体 step の返り値 + 検証用の配置詳細）。"""

    rewards: jax.Array    # (B,2) f64  [-wait, -cost]（done 済みは 0）
    scheduled: jax.Array  # (B,)  bool 本体の第3返り値（done 済み空回しは False）
    wait: jax.Array       # (B,)  i64  本体の第4返り値 waiting_time（done 済みは 0）
    done: jax.Array       # (B,)  bool
    start: jax.Array      # (B,)  i64  配置 start_time（検証用）
    nodes: jax.Array      # (B,h_max) i32 割当ノード（先頭 height 個が有効; 検証用）
    use_cloud: jax.Array  # (B,)  bool
    cost: jax.Array       # (B,)  i64


def make_initial_state(
    jobs: np.ndarray,
    n_on: int = 256,
    n_cl: int = 1024,
    e_max: int | None = None,
) -> EnvState:
    """jobs(B,N,8) float64（本体 env.jobs と同列順）から初期状態を作る。

    E_max 既定 = N（1 step = 1 イベント追加なので片資源に全部寄っても溢れない）。
    """
    jobs = np.asarray(jobs, dtype=np.float64)
    assert jobs.ndim == 3 and jobs.shape[2] == 8, f"jobs shape {jobs.shape}"
    B, N, _ = jobs.shape
    E = int(e_max) if e_max is not None else N
    assert E >= N, "E_max はエピソード内最大イベント数 N 以上が必要"
    return EnvState(
        jobs_w=jnp.asarray(jobs[:, :, 1], dtype=jnp.int64),
        jobs_h=jnp.asarray(jobs[:, :, 2], dtype=jnp.int32),
        jobs_arr=jnp.asarray(jobs[:, :, 0], dtype=jnp.int64),
        jobs_full=jnp.asarray(jobs),
        n_jobs=jnp.full((B,), N, dtype=jnp.int32),
        idx=jnp.zeros((B,), dtype=jnp.int32),
        time=jnp.zeros((B,), dtype=jnp.int64),
        done=jnp.zeros((B,), dtype=bool),
        ev_start=jnp.full((B, 2, E), INF64, dtype=jnp.int64),
        ev_end=jnp.full((B, 2, E), -1, dtype=jnp.int64),
        occ_on=jnp.zeros((B, E, n_on), dtype=bool),
        occ_cl=jnp.zeros((B, E, n_cl), dtype=bool),
        ev_len=jnp.zeros((B, 2), dtype=jnp.int32),
        ev_obs=jnp.zeros((B, N, 6), dtype=jnp.float64),
        cost_total=jnp.zeros((B,), dtype=jnp.int64),
        waits=jnp.zeros((B, N), dtype=jnp.int64),
    )


def make_alloc_fn(h_max: int, kernel: str = "v1", e_view: int | None = None):
    """alloc(state) -> (s_on, nd_on, s_cl, nd_cl) を返す（jit 可能）。

    現ジョブ（state.idx）を onprem / cloud それぞれに置いた場合の配置
    (start, nodes) を両方解く。make_step の配置探索部と同一計算。
    Phase 2 の eval ループはこれを step 前に呼び、
      - s_on から urgency 観測（本体 _front_job_urgency と同値）を作り、
      - 結果を make_apply_fn の apply へ渡して二重計算を避ける
    （本体の _REUSE_URGENCY_ALLOC と同じ再利用。obs→step 間で state 不変なので同値）。
    """
    kfn = _KERNELS[kernel]

    def alloc(state: EnvState):
        B, N = state.jobs_w.shape
        E = state.ev_start.shape[2]
        bidx = jnp.arange(B)

        # ---- 現ジョブ（done 済みは j を clip; 結果は apply 側 active マスクで無効化）----
        j = jnp.minimum(state.idx, N - 1)
        w = state.jobs_w[bidx, j]                       # (B,) i64
        h = state.jobs_h[bidx, j]                       # (B,) i32
        arr = state.jobs_arr[bidx, j]                   # (B,) i64

        # ---- 配置探索: 両資源で解く（本体 :161 相当; prune 無し全保持）----
        ev = slice(0, e_view if e_view is not None else E)
        s_on, _, nd_on, _, _ = kfn(
            state.ev_start[:, 0, ev], state.ev_end[:, 0, ev], state.occ_on[:, ev],
            w, h, arr, jnp.zeros((B,), dtype=bool), h_max=h_max,
        )
        s_cl, _, nd_cl, _, _ = kfn(
            state.ev_start[:, 1, ev], state.ev_end[:, 1, ev], state.occ_cl[:, ev],
            w, h, arr, jnp.ones((B,), dtype=bool), h_max=h_max,
        )
        return s_on, nd_on, s_cl, nd_cl

    return alloc


def make_apply_fn(n_on: int, n_cl: int, h_max: int):
    """apply(state, actions, s_on, nd_on, s_cl, nd_cl) -> (state', StepOut)。

    make_step の「配置探索より後ろ」全体（報酬・イベント記録・進行）。
    make_step(=Phase 1 検証済み) は alloc∘apply の合成として定義される
    （計算列は分割前と同一 = ビット一致は verify_batch_env.py が担保し続ける）。
    """

    def apply(state: EnvState, actions: jax.Array, s_on, nd_on, s_cl, nd_cl):
        B, N = state.jobs_w.shape
        E = state.ev_start.shape[2]
        bidx = jnp.arange(B)
        active = ~state.done

        j = jnp.minimum(state.idx, N - 1)
        w = state.jobs_w[bidx, j]                       # (B,) i64
        h = state.jobs_h[bidx, j]                       # (B,) i32
        arr = state.jobs_arr[bidx, j]                   # (B,) i64
        use_cloud = actions == 1                        # 0=onprem, 1=cloud（2=defer は将来）

        start = jnp.where(use_cloud, s_cl, s_on)        # (B,) i64
        nodes = jnp.where(use_cloud[:, None], nd_cl, nd_on)  # (B,h_max) i32

        # ---- 報酬（本体 :163-174, :612-615。slowdown は wmetric = wait / max(1, pt)）----
        wait = jnp.where(active, start - arr, 0)        # (B,) i64
        cost = jnp.where(active & use_cloud, w * h.astype(jnp.int64), 0)
        if _SLOWDOWN:
            wm = wait.astype(jnp.float64) / jnp.maximum(1.0, w.astype(jnp.float64))
            rewards = jnp.stack([-wm, -cost.astype(jnp.float64)], axis=1)
        else:
            rewards = jnp.stack([-wait, -cost], axis=1).astype(jnp.float64)

        # ---- イベント記録（本体 _record_event_schedule :568-610）----
        r = use_cloud.astype(jnp.int32)                 # 資源 index
        slot = state.ev_len[bidx, r]                    # (B,) 追記位置
        slot_w = jnp.where(active, slot, E)             # done 済みは範囲外 → drop
        end = start + w
        ev_start = state.ev_start.at[bidx, r, slot_w].set(start, mode="drop")
        ev_end = state.ev_end.at[bidx, r, slot_w].set(end, mode="drop")

        karr = jnp.arange(h_max, dtype=jnp.int32)
        valid_k = karr[None, :] < h[:, None]            # (B,h_max) 先頭 height 個
        tgt_on = active & ~use_cloud
        tgt_cl = active & use_cloud
        nd_idx_on = jnp.where(valid_k & tgt_on[:, None], nodes, n_on)   # 範囲外→drop
        nd_idx_cl = jnp.where(valid_k & tgt_cl[:, None], nodes, n_cl)
        occ_on = state.occ_on.at[bidx[:, None], slot[:, None], nd_idx_on].set(
            True, mode="drop"
        )
        occ_cl = state.occ_cl.at[bidx[:, None], slot[:, None], nd_idx_cl].set(
            True, mode="drop"
        )
        ev_len = state.ev_len.at[bidx, r].add(active.astype(jnp.int32))

        # 追加順イベント6特徴（本体 _event_buf.append :598-606 と同値; Phase 2 観測の材料）。
        # start_node は連続なら先頭、非連続でも free 昇順先頭 = min なので常に nodes[:,0]。
        feat = jnp.stack(
            [
                start.astype(jnp.float64),
                end.astype(jnp.float64),
                w.astype(jnp.float64),
                use_cloud.astype(jnp.float64),
                nodes[:, 0].astype(jnp.float64),
                h.astype(jnp.float64),
            ],
            axis=1,
        )                                               # (B,6)
        ev_obs = state.ev_obs.at[bidx, jnp.where(active, j, N), :].set(
            feat, mode="drop"
        )

        # ---- 進行（本体 :176-186）----
        waits = state.waits.at[bidx, jnp.where(active, j, N)].set(wait, mode="drop")
        cost_total = state.cost_total + cost
        idx = state.idx + active.astype(jnp.int32)
        done = idx >= state.n_jobs
        time = jnp.where(active, start, state.time)     # 本体: time = int(start_time)

        new_state = EnvState(
            jobs_w=state.jobs_w,
            jobs_h=state.jobs_h,
            jobs_arr=state.jobs_arr,
            jobs_full=state.jobs_full,
            n_jobs=state.n_jobs,
            idx=idx,
            time=time,
            done=done,
            ev_start=ev_start,
            ev_end=ev_end,
            occ_on=occ_on,
            occ_cl=occ_cl,
            ev_len=ev_len,
            ev_obs=ev_obs,
            cost_total=cost_total,
            waits=waits,
        )
        out = StepOut(
            rewards=rewards * active[:, None],
            scheduled=active,
            wait=wait,
            done=done,
            start=start,
            nodes=nodes,
            use_cloud=use_cloud,
            cost=cost,
        )
        return new_state, out

    return apply


def make_step(n_on: int, n_cl: int, h_max: int, kernel: str = "v1",
              e_view: int | None = None):
    """step(state, actions(B,) int32) -> (state', StepOut) を返す（jit 可能）。

    h_max はカーネルの nodes 出力幅（静的）。バッチ内最大 height 以上にすること
    （make_initial_state 後に int(jobs[:,:,2].max()) で決めるのが簡単）。

    e_view: カーネルに渡すイベントバッファの先頭幅（静的）。step t 開始時の
    有効イベント数は高々 t なので、t < e_view が保証される区間では先頭 e_view 個
    だけ見れば十分（無効スロットは start=INF64/end=-1/occ=False で自然排除される
    ため結果不変）。None はフル幅。エピソード序盤の O(T·E·N) を削る用
    （rollout_segmented が段階的に広げて使う）。

    実装は alloc（配置探索; make_alloc_fn）∘ apply（反映; make_apply_fn）の合成。
    Phase 2 eval が alloc 結果（urgency 観測）を再利用できるよう分割したもので、
    計算列は分割前と同一（verify_batch_env.py で全ステップ一致を確認済み）。
    """
    alloc = make_alloc_fn(h_max, kernel, e_view)
    apply = make_apply_fn(n_on, n_cl, h_max)

    def step(state: EnvState, actions: jax.Array):
        return apply(state, actions, *alloc(state))

    return step


@functools.partial(jax.jit, static_argnames=("n_on", "n_cl", "h_max", "kernel"))
def _rollout_impl(state, actions_tb, *, n_on, n_cl, h_max, kernel):
    step = make_step(n_on, n_cl, h_max, kernel)

    def body(s, a):
        s2, out = step(s, a)
        return s2, (out.rewards, out.done)

    final_state, (rewards, dones) = lax.scan(body, state, actions_tb)
    return final_state, rewards, dones


@functools.partial(
    jax.jit, static_argnames=("n_on", "n_cl", "h_max", "kernel", "e_view", "record")
)
def _rollout_seg_impl(state, actions_tb, *, n_on, n_cl, h_max, kernel, e_view, record):
    step = make_step(n_on, n_cl, h_max, kernel, e_view=e_view)

    def body(s, a):
        s2, out = step(s, a)
        return s2, (out if record else (out.rewards, out.done))

    return lax.scan(body, state, actions_tb)


def rollout_segmented(
    state: EnvState,
    actions_tb,
    n_on=256,
    n_cl=1024,
    h_max=None,
    kernel="v1",
    seg_len: int = 64,
    record: bool = False,
):
    """rollout の E 段階拡大版: step t 開始時の有効イベント数 ≤ t を利用し、
    エピソードを seg_len 刻みに分けてセグメント k ではイベントバッファ先頭
    min(E, (k+1)·seg_len) 個だけをカーネルに渡す（結果不変・実測 ~2倍）。

    record=True で StepOut 全項目（start/nodes 含む）を (T,B,...) で返す（検証用）。
    False は (rewards(T,B,2), dones(T,B))。コンパイルはセグメントごと
    （e_view が違うため; seg_len を粗くするとコンパイル回数と引き換えに削減率が落ちる）。
    """
    if h_max is None:
        h_max = int(jnp.max(state.jobs_h))
    actions_tb = jnp.asarray(actions_tb, dtype=jnp.int32)
    T = actions_tb.shape[0]
    E_full = state.ev_start.shape[2]
    chunks = []
    for t0 in range(0, T, seg_len):
        t1 = min(T, t0 + seg_len)
        e_view = min(E_full, t1)
        state, ys = _rollout_seg_impl(
            state, actions_tb[t0:t1], n_on=n_on, n_cl=n_cl, h_max=int(h_max),
            kernel=kernel, e_view=e_view, record=record,
        )
        chunks.append(ys)
    merged = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)
    return state, merged


def rollout(state: EnvState, actions_tb, n_on=256, n_cl=1024, h_max=None, kernel="v1"):
    """actions_tb(T,B) int32 でエピソード全体を lax.scan で回す（ベンチ/工場ユース）。

    返り値: (final_state, rewards(T,B,2) f64, dones(T,B) bool)。
    配置詳細（start/nodes）が必要な検証は step 単発を Python ループで回すこと
    （verify_batch_env.py 参照; ys に nodes を積むと (T,B,h_max) が無駄に大きい）。
    """
    if h_max is None:
        h_max = int(jnp.max(state.jobs_h))
    actions_tb = jnp.asarray(actions_tb, dtype=jnp.int32)
    return _rollout_impl(
        state, actions_tb, n_on=n_on, n_cl=n_cl, h_max=int(h_max), kernel=kernel
    )
