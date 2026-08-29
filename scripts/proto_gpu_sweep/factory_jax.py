"""Phase 3: rollout データ工場 — バッチenv + バッチ方策サンプリングで学習用エピソードを量産する。

Ray 16 CPU Actor の rollout 収集（distributed_pcn.Actor._run_episode :1283-1502）を
置き換え/増強する土台。eval（Phase 2）と違い**工場データは探索用なのでビット一致不要**。
この自由度で obs 構築をフル GPU 化する（Phase 2 の C 呼び経路はホスト律速だった）。

■ obs フル GPU 化（make_obs_fn）
  Phase 2 は本体と同一の C 関数 get_observation_event をホストで B 回/step 呼んで
  タイ順までビット再現した。工場は同じ仕様を JAX で実装する:
    window_start = max(0, time - n_window)
    filter: end >= window_start（イベントは追加順 = state.ev_obs の先頭 t 件）
    start 昇順 stable argsort → 末尾 min(nf,30) 件を出力行 0.. へ
    正規化 f64 演算 → f32 格納（clip01; scheduling_env_core.c:1093-1098 と同式）
  相違は「start が同値のイベントの順序」のみ: C は qsort（不安定）、ここは stable
  argsort（Python fォールバック _get_observation_python の mergesort と同じ側）。
  値集合は同一で行の並びだけが入れ替わりうる（verify_factory.py が定量化）。
  job_queue 40 は per-step 共通（idx=t）で事前計算、urgency は alloc の s_on から。

■ サンプリング方策（SamplingPolicy）
  本体 _act(eval_mode=False)（pcn_agent.py:3812-3821）は LogSoftmax logits →
  probs=exp(scores) → th.multinomial。2 行動系では Bernoulli(p_cloud) と同分布なので
  action = 1[u_t(b) < p_cloud] に置き換える。u_t(b) は per-episode シード
  SeedSequence([seed0, episode_id]) の一様列 = **バッチ構成に依存せず再現可能**
  （チャンク分割を変えても同じ episode_id なら同じ乱数列）。
  forward は Phase 2 と同じ torch バッチ（GEMM のバッチ/デバイス差 ~1e-5 は
  サンプリング確率の摂動としてそのまま許容。tie 再計算は不要）。

■ 指令付き/ランダム両対応（run_factory）
  - commands=[(dr(2,) f32, hz f32), ...]: 学習 Phase 3 相当。カウントダウンは
    Actor._run_episode :1421-1429 と同一演算列
    （dr ← f32(dr - f32(reward)); hz ← f32(max(hz-1,1)); クリップ無し・COST_HOLD 無し）。
  - random_probs=[p, ...]: 学習 Phase 1 相当（Bernoulli(p) 固定・NN 呼び無し）。

■ Transition 変換（episodes_to_transitions）
  本体 Actor._run_episode の出力と同じ構造を作る:
    episode = List[Transition(observation f32(221,), action int,
                              reward np.float32(2,)(per-step; 累積は Learner._add_episode),
                              next_observation f32(221,), terminal bool)]
    先頭 Transition の動的属性:
      objective_values = [cost, makespan, avg_wait]（env.calc_objective_values と同義）
      指令モード: command_return(f32(2,)) / solution_execution_time(float)
      ランダムモード: _pcn_episode_uid = "gpufactory:{seed0}:{eid}:{p}" / random_action_prob
  next_observation は本体 env.step の返す n_obs 相当（学習 update は observation しか
  読まない get_training_batch :1879-1979 が、形式・ハッシュ計算で必要）。
  最終 step の n_obs = 全イベント・time=最終start・job_queue 空・urgency 0。

■ チャンク送出（factory_chunks）
  obs_all(B,T+1,221) は B=1024/T=512 で ~464MB。chunk_b ずつ生成して yield し
  ピークメモリを抑える（episode_id はグローバル通し番号なので分割不変）。

usage（ライブラリ; verify_factory.py / bench_factory.py から使用）:
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. で import
"""
from __future__ import annotations

import functools
import sys
import time
from typing import Iterator, List, Sequence

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np

from batch_env_jax import (  # noqa: E402  (jax x64 有効化)
    INF64,
    EnvState,
    make_alloc_fn,
    make_apply_fn,
    make_initial_state,
)
from batch_eval_jax import precompute_job_queue  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

N_EVENTS_OBS = 30
EVENT_FEATURES = 6
JOB_QUEUE_SIZE = 40
OBS_DIM = N_EVENTS_OBS * EVENT_FEATURES + JOB_QUEUE_SIZE + 1  # 221


# ---------------------------------------------------------------------------
# obs フル GPU 化
# ---------------------------------------------------------------------------
def make_obs_fn(n_window: int, norm_time: float, norm_nodes: float):
    """obs(B,221) f32 を JAX で構築する関数を返す（jit 可能）。

    引数:
      ev_view (B,E,6) f64  追加順イベント6特徴（state.ev_obs の先頭 e_view 件ビュー）
      n_valid ()      i32  有効イベント数（2行動系では現 step t）
      time_now(B,)    i64  本体 self.time（直近配置 start）
      jq_t    (40,)   f64  job_queue 観測（全エピソード共通; 事前計算）
      pw      (B,)    i64  現ジョブ onprem 予測待ち max(0, start_on - arrival)。
                           最終観測(idx>=n_jobs)は 0 を渡す（本体 _front_job_urgency と同じ）。
    """

    def obs_fn(ev_view, n_valid, time_now, jq_t, pw):
        B, E, _ = ev_view.shape
        ws = jnp.maximum(0, time_now - n_window)                    # (B,) i64
        starts = ev_view[:, :, 0]                                    # f64
        ends = ev_view[:, :, 1]
        keep = (jnp.arange(E)[None, :] < n_valid) & (
            ends >= ws[:, None].astype(ev_view.dtype)
        )
        # 必要なのは「keep 行の start 上位30件を昇順で」だけなので全ソートせず top_k。
        # 複合キー start*(E+1)+idx により同値 start のタイも追加順（stable argsort /
        # Python fallback mergesort と同一）のまま = 置換前実装と完全一致（verify[0]で担保）。
        # start は整数値(時刻) < 2^40 想定 → i64 複合キーは厳密。無効行は -1。
        kk = min(N_EVENTS_OBS, E)                        # E は静的形状 → Python int
        idx64 = jnp.arange(E, dtype=jnp.int64)
        key = jnp.where(
            keep, starts.astype(jnp.int64) * jnp.int64(E + 1) + idx64[None, :],
            jnp.int64(-1),
        )
        topv, _ = jax.lax.top_k(key, kk)                             # (B,kk) 降順
        n_take = (topv >= 0).sum(axis=1).astype(jnp.int32)           # = min(nf,30)
        k = jnp.arange(N_EVENTS_OBS, dtype=jnp.int32)
        row_ok = k[None, :] < n_take[:, None]
        # 出力行 k(昇順) ← 降順位置 n_take-1-k
        desc_pos = jnp.clip(n_take[:, None] - 1 - k[None, :], 0, kk - 1)
        sel = jnp.take_along_axis(topv, desc_pos[:, :kk] if kk < N_EVENTS_OBS else desc_pos, axis=1)
        if kk < N_EVENTS_OBS:  # E<30 の極小ケースのみ発生（形状合わせ）
            sel = jnp.pad(sel, ((0, 0), (0, N_EVENTS_OBS - kk)), constant_values=-1)
        src_idx = (sel % jnp.int64(E + 1)).astype(jnp.int32)         # 無効(-1)→E→clipで無害
        feats = jnp.take_along_axis(
            ev_view, src_idx[:, :, None], axis=1, mode="clip"
        )                                                            # (B,30,6) f64
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
        jq = jnp.broadcast_to(
            jq_t.astype(jnp.float32)[None, :], (B, JOB_QUEUE_SIZE)
        )
        urg = jnp.clip(
            jnp.log1p(pw.astype(jnp.float64)) / 16.0, 0.0, 1.0
        ).astype(jnp.float32)
        return jnp.concatenate([ev_flat, jq, urg[:, None]], axis=1)  # (B,221) f32

    return obs_fn


@functools.lru_cache(maxsize=None)
def _obs_jit_cached(n_window: int, norm_time: float, norm_nodes: float):
    return jax.jit(make_obs_fn(n_window, norm_time, norm_nodes))


@functools.lru_cache(maxsize=None)
def _alloc_jit_cached(h_max: int, kernel: str, e_view: int):
    return jax.jit(make_alloc_fn(h_max, kernel, e_view))


def make_alloc_res_fn(h_max: int, kernel: str, r: int, e_view: int):
    """資源 r（0=onprem, 1=cloud）単独の配置探索（make_alloc_fn の片側と同一計算列）。

    大N高速化の柱: v1 カーネルは候補数 T=E+1 なので O(E²·N)。両資源に同じ e_view を
    渡す代わりに「その資源の実イベント数 max を覆う幅」に絞ると二乗で効く
    （イベント総数は t で分配されるので中央方策なら各 ~t/2 → alloc ~4x減）。
    有効イベントは必ず全部ビューに含まれる（追記位置 ev_len ≤ 幅）ため結果不変。
    """
    from batch_env_jax import _KERNELS

    kfn = _KERNELS[kernel]

    def alloc_res(state):
        B, N = state.jobs_w.shape
        bidx = jnp.arange(B)
        j = jnp.minimum(state.idx, N - 1)
        w = state.jobs_w[bidx, j]
        h = state.jobs_h[bidx, j]
        arr = state.jobs_arr[bidx, j]
        ev = slice(0, e_view)
        occ = (state.occ_on if r == 0 else state.occ_cl)[:, ev]
        cont = jnp.full((B,), bool(r))                  # on=False / cl=True（従来と同一）
        s, _, nd, _, _ = kfn(
            state.ev_start[:, r, ev], state.ev_end[:, r, ev], occ,
            w, h, arr, cont, h_max=h_max,
        )
        return s, nd

    return alloc_res


@functools.lru_cache(maxsize=None)
def _alloc_res_jit_cached(h_max: int, kernel: str, r: int, e_view: int):
    return jax.jit(make_alloc_res_fn(h_max, kernel, r, e_view))


def _bucket(n: int, T: int, gran: int) -> int:
    """イベント数 n を覆う gran 刻みのビュー幅（jit 形状のバケット化; 最小 gran）。"""
    return min(T, max(gran, -(-n // gran) * gran))


def make_compact_fn(r: int):
    """資源 r のイベント配列を「生存イベント（end >= 将来ジョブの最小 arrival）」で前詰めする。

    大N高速化の柱2（結果不変の prune）: 死んだイベントは以後の全 alloc で
      - overlap 条件 ends > t（t >= arrival >= suffmin）が偽
      - 候補条件 ends >= arrival が偽
    となり集合から除いても結果不変（本体 env の prune :244-251 と同じ理屈。
    arrival 非単調にも suffix-min で対応）。前詰め後は ev_len = 生存数となり、
    以後の alloc ビュー幅（_bucket）が「生存数+セグメント増分」まで縮む。
    イベントの行順は alloc に無関係（集合として扱われる）なので並べ替えは自由。
    obs 用の追加順 6 特徴は別配列 state.ev_obs（job index ベース）で無傷。
    """

    def compact(state: EnvState, suffmin):
        E = state.ev_start.shape[2]
        ends_r = state.ev_end[:, r]                                  # (B,E)
        alive = ends_r >= suffmin                                    # 無効(-1)も dead
        order = jnp.argsort(~alive, axis=1, stable=True)             # alive 前詰め
        cnt = alive.sum(axis=1).astype(jnp.int32)                    # (B,)
        pos_ok = jnp.arange(E)[None, :] < cnt[:, None]
        st = jnp.take_along_axis(state.ev_start[:, r], order, axis=1)
        en = jnp.take_along_axis(ends_r, order, axis=1)
        st = jnp.where(pos_ok, st, INF64)                            # 残骸を無効化
        en = jnp.where(pos_ok, en, -1)
        occ = state.occ_on if r == 0 else state.occ_cl
        occ2 = jnp.take_along_axis(occ, order[:, :, None], axis=1) & pos_ok[:, :, None]
        kw = dict(
            ev_start=state.ev_start.at[:, r].set(st),
            ev_end=state.ev_end.at[:, r].set(en),
            ev_len=state.ev_len.at[:, r].set(cnt),
        )
        kw["occ_on" if r == 0 else "occ_cl"] = occ2
        return state._replace(**kw), cnt

    return compact


@functools.lru_cache(maxsize=None)
def _compact_jit_cached(r: int):
    return jax.jit(make_compact_fn(r))


@functools.lru_cache(maxsize=None)
def _apply_jit_cached(n_on: int, n_cl: int, h_max: int):
    return jax.jit(make_apply_fn(n_on, n_cl, h_max), donate_argnums=0)


# ---------------------------------------------------------------------------
# サンプリング方策
# ---------------------------------------------------------------------------
class SamplingPolicy:
    """torch バッチ forward → P(cloud)。tie 再計算なし（サンプリングなので不要）。"""

    def __init__(
        self,
        ckpt_path: str | None,
        n_jobs: int,
        hidden_dim: int = 512,
        state_dim: int = OBS_DIM,
        device: str = "cuda",
    ):
        import torch as th

        from src.agents.pcn_agent import DiscreteActionsDefaultModel

        self.th = th
        self.device = device
        sf = np.array([1.0, 1.0, 1.0 / max(1, n_jobs)], dtype=np.float32)
        m = DiscreteActionsDefaultModel(state_dim, 2, 2, sf, hidden_dim=hidden_dim)
        if ckpt_path is not None:
            st = th.load(ckpt_path, map_location="cpu", weights_only=False)
            m.load_state_dict(st.get("model_state_dict", st), strict=False)
        # ckpt_path=None: 重みは後で load_state_dict で同期する（学習ループ統合用）
        m.eval()
        self.model = m.to(device)

    def load_state_dict(self, sd) -> None:
        """重み同期（学習ループ統合用; state_dict は CPU/GPU どちらでも可）。"""
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()

    def p_cloud(self, obs: np.ndarray, dr: np.ndarray, hz: np.ndarray) -> np.ndarray:
        """obs(B,221) f32, dr(B,2) f32, hz(B,) f32 → P(action=1) (B,) f32。

        本体 _act: LogSoftmax logits → probs=exp(scores) → multinomial。
        2 行動では multinomial(probs) ≡ Bernoulli(probs[1])（softmax なので和1）。
        """
        th = self.th
        with th.no_grad():
            lp = self.model(
                th.tensor(obs, device=self.device),
                th.tensor(dr, device=self.device),
                th.tensor(hz[:, None], device=self.device),
            ).cpu().numpy()
        return np.exp(lp[:, 1])


# ---------------------------------------------------------------------------
# 工場本体
# ---------------------------------------------------------------------------
def episode_uniforms(seed0: int, episode_id: int, T: int) -> np.ndarray:
    """episode_id 決定の一様乱数列 u_t（バッチ構成非依存・再現可能）。"""
    return np.random.default_rng(np.random.SeedSequence([seed0, episode_id])).random(T)


def run_factory(
    jobs: np.ndarray,
    n_episodes: int,
    *,
    commands: Sequence | None = None,
    random_probs: Sequence[float] | None = None,
    policy: SamplingPolicy | None = None,
    seed0: int = 0,
    episode_id0: int = 0,
    n_on: int = 256,
    n_cl: int = 1024,
    n_window: int = 100,
    kernel: str = "v1h",
    seg_len: int = 64,
    alloc_gran: int = 32,
    prune_seg: int = 64,
) -> dict:
    """jobs(T,8) × B=n_episodes のサンプリング rollout（全エピソード同一ジョブ列）。

    commands: [(dr(2,) f32, hz float), ...] 長さ B → 指令付き（Phase 3 相当）
    random_probs: [p, ...] 長さ B → Bernoulli(p) 固定（Phase 1 相当; policy 不要）

    返り値 dict:
      obs       (B,T+1,221) f32  step t の観測 + 最終観測（next_observation 用）
      actions   (B,T) int8
      rewards   (B,T,2) f64      per-step [-wait, -cost]（整数値）
      achieved  (B,3) f64        [cost, makespan, avg_wait]
      commands0 (B,2) f32 / probs (B,) f32 / episode_ids (B,) / seed0
      stats     dict             計時
    """
    mode = "command" if commands is not None else "random"
    if mode == "command":
        assert policy is not None and len(commands) == n_episodes
    else:
        assert random_probs is not None and len(random_probs) == n_episodes

    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    B = n_episodes
    norm_time = float(max(1, n_window))
    norm_nodes = float(max(n_on, n_cl))

    state = make_initial_state(np.broadcast_to(jobs, (B, T, 8)), n_on=n_on, n_cl=n_cl)
    h_max = int(jobs[:, 2].max())
    apply_jit = _apply_jit_cached(n_on, n_cl, h_max)
    obs_jit = _obs_jit_cached(n_window, norm_time, norm_nodes)

    jq = np.zeros((T + 1, JOB_QUEUE_SIZE), dtype=np.float64)
    jq[:T] = precompute_job_queue(jobs)                 # t=T 行は空（idx=T）
    jq_dev = jnp.asarray(jq)
    arr_arr = jobs[:, 0].astype(np.int64)

    episode_ids = np.arange(episode_id0, episode_id0 + B, dtype=np.int64)
    u = np.stack([episode_uniforms(seed0, int(e), T) for e in episode_ids])  # (B,T)

    if mode == "command":
        dr = np.stack([np.asarray(c[0], dtype=np.float32) for c in commands])
        hz = np.array([float(c[1]) for c in commands], dtype=np.float32)
        dr0 = dr.copy()
    else:
        probs = np.asarray(random_probs, dtype=np.float32)

    obs_all = np.zeros((B, T + 1, OBS_DIM), dtype=np.float32)
    actions_all = np.zeros((B, T), dtype=np.int8)
    rewards_all = np.zeros((B, T, 2), dtype=np.float64)

    # 資源別の有効イベント数（B,）をホストで追跡（alloc ビュー幅と prune 用）
    suffmin = np.minimum.accumulate(arr_arr[::-1])[::-1]  # 将来ジョブの最小 arrival
    ev_len_h = [np.zeros(B, dtype=np.int64), np.zeros(B, dtype=np.int64)]  # [on, cl]
    t_env = t_obs = t_nn = 0.0
    wall0 = time.perf_counter()
    for t in range(T):
        e_view = min(T, ((t // seg_len) + 1) * seg_len)
        if prune_seg and t and t % prune_seg == 0:
            # 生存イベント前詰め（結果不変; make_compact_fn 参照）→ ビュー幅が縮む
            for r in (0, 1):
                state, cnt = _compact_jit_cached(r)(state, jnp.int64(suffmin[t]))
                ev_len_h[r] = np.asarray(cnt, dtype=np.int64)
        # 資源別ビュー幅: 有効イベント数 max を覆う alloc_gran 刻み（結果不変・E²N項を分配）
        e_on = _bucket(int(ev_len_h[0].max()), T, alloc_gran)
        e_cl = _bucket(int(ev_len_h[1].max()), T, alloc_gran)
        t0 = time.perf_counter()
        s_on, nd_on = _alloc_res_jit_cached(h_max, kernel, 0, e_on)(state)
        s_cl, nd_cl = _alloc_res_jit_cached(h_max, kernel, 1, e_cl)(state)
        t1 = time.perf_counter()

        pw = jnp.maximum(0, s_on - int(arr_arr[t]))
        obs_dev = obs_jit(
            state.ev_obs[:, :e_view], jnp.int32(t), state.time, jq_dev[t], pw
        )
        obs_h = np.asarray(obs_dev)
        obs_all[:, t] = obs_h
        t2 = time.perf_counter()

        if mode == "command":
            p = policy.p_cloud(obs_h, dr, hz)
        else:
            p = probs
        acts = (u[:, t] < p).astype(np.int64)            # Bernoulli(p_cloud)
        t3 = time.perf_counter()

        state, out = apply_jit(
            state, jnp.asarray(acts, dtype=jnp.int32), s_on, nd_on, s_cl, nd_cl
        )
        rewards = np.asarray(out.rewards)                # (B,2) f64 整数値
        actions_all[:, t] = acts
        rewards_all[:, t] = rewards
        ev_len_h[0] += 1 - acts
        ev_len_h[1] += acts
        if mode == "command":
            # Actor._run_episode :1421 と同一演算列（f32 減算・クリップ無し）
            dr = (dr - rewards.astype(np.float32)).astype(np.float32)
            hz = np.maximum(hz - 1.0, 1.0).astype(np.float32)
        t4 = time.perf_counter()
        t_env += (t1 - t0) + (t4 - t3)
        t_obs += t2 - t1
        t_nn += t3 - t2

    # ---- 最終観測（本体 step :186 done 時の get_observation()）----
    obs_dev = obs_jit(
        state.ev_obs, jnp.int32(T), state.time, jq_dev[T],
        jnp.zeros((B,), dtype=jnp.int64),                # idx>=n_jobs → urgency 0
    )
    obs_all[:, T] = np.asarray(obs_dev)

    cost = np.asarray(state.cost_total, dtype=np.float64)
    avg_wait = np.asarray(state.waits.sum(axis=1), dtype=np.float64) / T
    makespan = np.asarray(jnp.max(state.ev_end, axis=(1, 2)), dtype=np.float64)
    out = {
        "obs": obs_all,
        "actions": actions_all,
        "rewards": rewards_all,
        "achieved": np.stack([cost, makespan, avg_wait], axis=1),
        "episode_ids": episode_ids,
        "seed0": seed0,
        "mode": mode,
        "stats": {
            "t_env": t_env, "t_obs": t_obs, "t_nn": t_nn,
            "wall": time.perf_counter() - wall0,
        },
    }
    if mode == "command":
        out["commands0"] = dr0
    else:
        out["probs"] = probs
    return out


# ---------------------------------------------------------------------------
# Transition 変換アダプタ
# ---------------------------------------------------------------------------
def episodes_to_transitions(res: dict) -> list:
    """run_factory 出力 → 本体 Actor._run_episode と同じ List[List[Transition]]。

    reward は per-step のまま格納する（累積化は Learner._add_episode :1928-1931 /
    ReplayBuffer 経由の学習取り込みが行う）。
    """
    from src.agents.pcn_agent import Transition

    obs = res["obs"]
    actions = res["actions"]
    rewards = res["rewards"]
    achieved = res["achieved"]
    B, T = actions.shape
    wall_per_ep = float(res["stats"]["wall"]) / max(1, B)
    episodes = []
    for b in range(B):
        eps = []
        for t in range(T):
            eps.append(
                Transition(
                    observation=obs[b, t].copy(),
                    action=int(actions[b, t]),
                    reward=np.float32(rewards[b, t]).copy(),
                    next_observation=obs[b, t + 1].copy(),
                    terminal=bool(t == T - 1),
                )
            )
        first = eps[0]
        # Actor :1441/1462: cost, makespan, avg_wait = env.calc_objective_values()
        first.objective_values = [
            int(achieved[b, 0]), int(achieved[b, 1]), float(achieved[b, 2]),
        ]
        if res["mode"] == "command":
            # Actor :1467-1471 と同じ動的属性（指令追従loss/可視化が読む）
            first.command_return = np.asarray(res["commands0"][b], dtype=np.float32)
            first.solution_execution_time = wall_per_ep
        else:
            # Actor run() :1143-1145 と同形式の uid（Learner 重複検出の高速経路）
            p = float(res["probs"][b])
            first._pcn_episode_uid = (
                f"gpufactory:{res['seed0']}:{int(res['episode_ids'][b])}:{p}"
            )
            first.random_action_prob = p
        episodes.append(eps)
    return episodes


def factory_chunks(
    jobs: np.ndarray,
    n_episodes: int,
    chunk_b: int = 256,
    *,
    commands: Sequence | None = None,
    random_probs: Sequence[float] | None = None,
    episode_id0: int = 0,
    **kw,
) -> Iterator[List[list]]:
    """チャンク送出: n_episodes を chunk_b ずつ生成し episodes(List[List[Transition]])
    を yield する。obs_all のピークを chunk_b×(T+1)×221×4B に抑える
    （B=1024/T=512 一括の ~464MB を回避）。episode_id はグローバル通し番号なので
    チャンク割りを変えても各エピソードの乱数列・出力は不変。
    """
    for i0 in range(0, n_episodes, chunk_b):
        i1 = min(n_episodes, i0 + chunk_b)
        res = run_factory(
            jobs,
            i1 - i0,
            commands=None if commands is None else commands[i0:i1],
            random_probs=None if random_probs is None else random_probs[i0:i1],
            episode_id0=episode_id0 + i0,
            **kw,
        )
        yield episodes_to_transitions(res)
