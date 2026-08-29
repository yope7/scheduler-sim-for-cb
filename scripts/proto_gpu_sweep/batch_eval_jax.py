"""Phase 2: 学習中eval（指令格子×greedy）の GPU バッチ実行。

「1つのモデル × 多数の指令 NCMD」の greedy 評価を B=NCMD 並列で回す:
  (a) env 遷移     = Phase 1 の JAX バッチ env（alloc/apply 分割; 全ビット一致検証済み）
  (b) 観測 221次元 = 本体と同一の C 関数 get_observation_event をホストで per-episode 呼び
                     （同一入力→同一ビット。イベント履歴はホスト側で追記）+ urgency 1次元
  (c) 方策 forward = PyTorch モデルをそのまま GPU バッチで（B,221）1発
  (d) 指令カウントダウン = pcn_agent._run_episode と同一の float32 演算列

■ 観測 221 = イベント30×6特徴(180) + job_queue 5ジョブ×8(40) + urgency(1)
  イベント6特徴・窓・正規化は scheduling_env_core.c:get_observation_event（:1104-1187）:
    window_start = max(0, time - n_window)
    filter: end >= window_start（全イベント履歴=追加順。event_buf は prune されない）
    qsort by start（:1085-1091 比較は start のみ。tie 順は qsort 実装依存 →
      JAX で再実装せず同一 C 関数を呼ぶことで tie 順までビット再現）
    末尾 n_take=min(nf,30) 件を出力行 0..n_take-1 へ、
    [clip01((s-ws)/norm_t), clip01((e-ws)/norm_t), clip01(d/norm_t), uc,
     clip01(sn/norm_n), clip01(jh/norm_n)]（clip01_scaled :1093-1098, f64 演算→f32 格納）
    norm_t = max(1, n_window), norm_n = max(n_on, n_cl)（event_c_env.py:69-70）
  job_queue 40 = jobs[idx:idx+5] を各行 np.roll(-1)（event_native_env.py:204-212 の
    _to_queue_job）で左1回転し flatten・不足ゼロ埋め（event_c_env.py:141-148）。
    2行動系では idx=t 全バッチ共通 → 全ステップ分を事前計算できる。
  urgency 1 = 現ジョブを onprem に置いた場合の予測待ち pw=max(0, start_on - arrival) の
    log 正規化 clip(log1p(pw)/16, 0, 1)（event_c_env.py:187-208 _front_job_urgency）。
    start_on は Phase 1 alloc の onprem 側出力（本体 _REUSE_URGENCY_ALLOC と同じ再利用）。

■ 方策 forward（pcn_agent.py:3782-3821 _act / :573-658 BasePCNModel.forward）
  greedy=argmax（決定的）。ただし GEMM は形状（バッチ数）とデバイスでビットが変わる
  （probe_forward_bitmatch.py 実測: CPU単発/CPUバッチ/GPU 全て互いに ~2.4e-5 ずれ、
  ビット一致無し）ため、バッチ argmax と参照（学習中evalのActor=CPU・B=1）の argmax は
  logit ギャップが極小の行だけ割れうる。→ ハイブリッド: バッチ forward 後、
  |logit0-logit1| < tie_eps(既定1e-3) の行のみ参照と同形状(B=1)・同デバイス(cpu)で
  再計算し argmax を参照ビットで確定。ずれ ~2.4e-5 ≪ eps/2 なので非再計算行の
  argmax は参照と同符号が保証され、行動列は参照と完全一致する。

■ カウントダウン（pcn_agent.py:3841-3901 _run_episode, eval_mode=True）
  desired_return ← float32((dr_f32 - reward_f64))  毎step・クリップ無し
  desired_horizon ← float32(max(hz-1, 1.0))        scheduled 時（2行動系は毎step）

usage（ライブラリ; verify_batch_eval.py / bench_batch_eval.py から使用）:
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. で import
"""
from __future__ import annotations

import functools
import os
import sys
import time
from typing import Sequence

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np

from batch_env_jax import (  # noqa: E402  (jax x64 有効化)
    make_alloc_fn,
    make_apply_fn,
    make_initial_state,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

N_EVENTS_OBS = 30
EVENT_FEATURES = 6
JOB_QUEUE_SIZE = 40  # 5ジョブ × 8属性
OBS_DIM = N_EVENTS_OBS * EVENT_FEATURES + JOB_QUEUE_SIZE + 1  # +urgency = 221


def _c_get_observation_event():
    from scheduling_env_core import get_observation_event

    return get_observation_event


@functools.lru_cache(maxsize=None)
def _alloc_jit_cached(h_max: int, kernel: str, e_view: int):
    """jit を関数オブジェクトごとキャッシュ（呼び直しでも再コンパイルしない）。"""
    return jax.jit(make_alloc_fn(h_max, kernel, e_view))


@functools.lru_cache(maxsize=None)
def _apply_jit_cached(n_on: int, n_cl: int, h_max: int):
    return jax.jit(make_apply_fn(n_on, n_cl, h_max), donate_argnums=0)


def precompute_job_queue(jobs: np.ndarray) -> np.ndarray:
    """全ステップ分の job_queue 観測 40 要素（f64）を事前計算する。

    step t の観測は jobs[t:t+5] の各行を _to_queue_job（np.roll(arr,-1) と同値;
    event_native_env.py:204-212）で回して flatten、不足ゼロ埋め
    （_pack_job_queue_f64; event_c_env.py:141-148）。defer 無し 2 行動系では
    idx_next_job = t が全エピソード共通なので (T,40) に閉じる。
    """
    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    out = np.zeros((T, JOB_QUEUE_SIZE), dtype=np.float64)
    rolled = np.concatenate([jobs[:, 1:], jobs[:, :1]], axis=1)  # 各行 roll(-1)
    for t in range(T):
        k = min(5, T - t)
        out[t, : 8 * k] = rolled[t : t + k].ravel()
    return out


class HybridGreedyPolicy:
    """バッチ greedy 方策: GPU バッチ forward + 近接タイ行の CPU(B=1) 再計算。

    参照（既存 eval = pcn_agent._act, モデルは CPU・バッチ1）と行動列を一致させる:
      - 全行を device(既定 cuda) でバッチ forward → argmax
      - |logit差| < tie_eps の行だけ、参照と同形状 (1,221)・同デバイス(ref_device) で
        再計算し argmax を上書き（forward のビットが参照と完全一致 → 同じ行動）
      - 非再計算行は「参照とのずれ ~2.4e-5 ≪ tie_eps/2」なので符号反転できない
    stats に再計算回数・バッチ↔参照 logit 最大偏差・最小非再計算ギャップを蓄積。
    """

    def __init__(
        self,
        ckpt_path: str,
        n_jobs: int,
        hidden_dim: int = 512,
        state_dim: int = OBS_DIM,
        device: str = "cuda",
        ref_device: str = "cpu",
        tie_eps: float = 1e-3,
    ):
        import torch as th

        from src.agents.pcn_agent import DiscreteActionsDefaultModel

        self.th = th
        self.tie_eps = float(tie_eps)
        self.device = device
        self.ref_device = ref_device
        st = th.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = st.get("model_state_dict", st)
        sf = np.array([1.0, 1.0, 1.0 / max(1, n_jobs)], dtype=np.float32)

        def build(dev):
            m = DiscreteActionsDefaultModel(state_dim, 2, 2, sf, hidden_dim=hidden_dim)
            m.load_state_dict(sd, strict=False)
            m.eval()
            return m.to(dev)

        self.model = build(device)
        self.model_ref = build(ref_device) if ref_device != device else self.model
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "n_recheck": 0,
            "n_flip": 0,
            "n_rows": 0,
            "max_dev": 0.0,          # 再計算行での |バッチlogit - 参照logit| 最大
            "min_gap_norecheck": float("inf"),
        }

    def act(self, obs: np.ndarray, dr: np.ndarray, hz: np.float32) -> np.ndarray:
        """obs(B,221) f32, dr(B,2) f32, hz f32 スカラ → actions(B,) int64（greedy）。"""
        th = self.th
        B = obs.shape[0]
        with th.no_grad():
            lp = self.model(
                th.tensor(obs, device=self.device),
                th.tensor(dr, device=self.device),
                th.full((B, 1), float(hz), dtype=th.float32, device=self.device),
            ).cpu().numpy()
        gap = np.abs(lp[:, 0] - lp[:, 1])
        actions = lp.argmax(axis=1)
        recheck = np.flatnonzero(gap < self.tie_eps)
        self.stats["n_rows"] += B
        self.stats["n_recheck"] += int(recheck.size)
        keep = gap >= self.tie_eps
        if keep.any():
            self.stats["min_gap_norecheck"] = min(
                self.stats["min_gap_norecheck"], float(gap[keep].min())
            )
        for b in recheck:
            # 参照 _act（pcn_agent.py:3782-3785, :3812-3814）と同一の演算列・形状・デバイス
            with th.no_grad():
                scores = self.model_ref(
                    th.tensor(np.array([obs[b]]), device=self.ref_device).float(),
                    th.tensor(np.array([dr[b]]), device=self.ref_device).float(),
                    th.tensor([[hz]], device=self.ref_device).float(),
                ).detach()[0]
            a_ref = int(th.argmax(scores).item())
            dev = float(np.abs(scores.cpu().numpy() - lp[b]).max())
            self.stats["max_dev"] = max(self.stats["max_dev"], dev)
            if a_ref != int(actions[b]):
                self.stats["n_flip"] += 1
            actions[b] = a_ref
        return actions


def run_batch_eval(
    jobs: np.ndarray,
    commands: Sequence[np.ndarray],
    policy: HybridGreedyPolicy,
    n_on: int = 256,
    n_cl: int = 1024,
    n_window: int = 100,
    kernel: str = "v1",
    seg_len: int = 64,
    record_obs: bool = False,
):
    """同一ジョブ列 jobs(NJ,8) × 指令 commands（NCMD 個の dr(2,) f32）を 1 バッチで評価。

    返り値 dict:
      actions   (B,T) int8   行動列
      achieved  (B,2) f64    [cost, avg_wait]（env.calc_objective_values と同値）
      e_return  (B,2) f32    達成 return（transitions[0].reward 相当の γ=1 累積; 参考）
      obs       (B,T,221) f32  record_obs=True のみ（検証用）
      stats     dict         タイ再計算などの方策統計 + 計時
    """
    get_obs = _c_get_observation_event()
    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    B = len(commands)
    norm_time = float(max(1, n_window))
    norm_nodes = float(max(n_on, n_cl))

    state = make_initial_state(np.broadcast_to(jobs, (B, T, 8)), n_on=n_on, n_cl=n_cl)
    h_max = int(jobs[:, 2].max())
    apply_jit = _apply_jit_cached(n_on, n_cl, h_max)

    def get_alloc(e_view: int):
        return _alloc_jit_cached(h_max, kernel, e_view)

    jq = precompute_job_queue(jobs)                    # (T,40) f64
    w_arr = jobs[:, 1].astype(np.int64)
    h_arr = jobs[:, 2].astype(np.int64)
    arr_arr = jobs[:, 0].astype(np.int64)

    dr = np.stack([np.asarray(c, dtype=np.float32) for c in commands])  # (B,2)
    hz = np.float32(T)                                 # _run_episode 呼び出し時の horizon
    events_host = np.zeros((B, T, 6), dtype=np.float64)
    actions_all = np.zeros((B, T), dtype=np.int8)
    e_return = np.zeros((B, 2), dtype=np.float64)      # f64 で積み、最後に f32 化はしない
    obs_all = np.zeros((B, T, OBS_DIM), dtype=np.float32) if record_obs else None
    obs = np.empty((B, OBS_DIM), dtype=np.float32)

    t_env = t_obs = t_nn = 0.0
    for t in range(T):
        e_view = min(T, ((t // seg_len) + 1) * seg_len)
        t0 = time.perf_counter()
        s_on, nd_on, s_cl, nd_cl = get_alloc(e_view)(state)
        s_on_h = np.asarray(s_on)                      # (B,) i64（urgency 用に取得）
        time_h = np.asarray(state.time)                # (B,) i64
        t1 = time.perf_counter()

        # ---- 観測（本体と同じ C 関数; イベント履歴は追加順そのまま）----
        # 注: 「end >= window_start の事前フィルタで C へ渡す行数を絞る」は C 側 filter と
        # 同一述語なのでビット安全だが、実測で mask 構築(B,t)+fancy copy のコストが C 側の
        # 短縮分を上回り逆効果（NJ512/372cmd で obs 4.4s→6.2s）だったため撤回した。
        ev_t = events_host[:, :t]
        for b in range(B):
            obs[b, : OBS_DIM - 1] = get_obs(
                ev_t[b], int(time_h[b]), n_window, norm_time, norm_nodes,
                N_EVENTS_OBS, EVENT_FEATURES, JOB_QUEUE_SIZE, jq[t],
            )
        # urgency（event_c_env.py:187-208 と同値。pw は int, log1p は f64 → f32 格納）
        pw = np.maximum(0, s_on_h - arr_arr[t]).astype(np.float64)
        obs[:, OBS_DIM - 1] = np.clip(np.log1p(pw) / 16.0, 0.0, 1.0).astype(np.float32)
        if record_obs:
            obs_all[:, t] = obs
        t2 = time.perf_counter()

        # ---- 方策（greedy; 近接タイのみ参照ビットで確定）----
        acts = policy.act(obs, dr, hz)
        actions_all[:, t] = acts
        t3 = time.perf_counter()

        # ---- 遷移 + カウントダウン（_run_episode :3894-3901 と同一演算）----
        state, out = apply_jit(
            state, jnp.asarray(acts, dtype=jnp.int32), s_on, nd_on, s_cl, nd_cl
        )
        rewards = np.asarray(out.rewards)              # (B,2) f64（整数値）
        start_h = np.asarray(out.start)
        node0_h = np.asarray(out.nodes[:, 0])
        events_host[:, t, 0] = start_h
        events_host[:, t, 1] = start_h + w_arr[t]
        events_host[:, t, 2] = w_arr[t]
        events_host[:, t, 3] = acts == 1
        events_host[:, t, 4] = node0_h
        events_host[:, t, 5] = h_arr[t]
        e_return += rewards
        dr = (dr - rewards).astype(np.float32)         # eval はクリップ無し
        hz = np.float32(max(hz - 1, 1.0))              # 2行動系は毎step scheduled
        t4 = time.perf_counter()
        t_env += (t1 - t0) + (t4 - t3)
        t_obs += t2 - t1
        t_nn += t3 - t2

    fs = state
    cost = np.asarray(fs.cost_total, dtype=np.float64)
    avg_wait = np.asarray(fs.waits.sum(axis=1), dtype=np.float64) / T
    result = {
        "actions": actions_all,
        "achieved": np.stack([cost, avg_wait], axis=1),
        "e_return": e_return,
        "stats": {**policy.stats, "t_env": t_env, "t_obs": t_obs, "t_nn": t_nn},
    }
    if record_obs:
        result["obs"] = obs_all
    return result
