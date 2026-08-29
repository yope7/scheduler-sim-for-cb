"""[PCN_GPU_FACTORY] 配列バックエンドのエピソード表現（Transition オブジェクト廃止）。

40960 級では 1 エピソード=40960 transition。従来経路は各 step を Python の Transition
オブジェクトに展開し（655万個）、Ray を跨いで直列化 → learner 取込していたため、
Ray put/get(22+15µs/tr) と _add_episode 逆cumsum(2.5µs/tr) が O(総transition) で律速
（40960 で learner 初期取込 >29 分。plasma 16GB 超で spill し超線形化）。

FactoryArrayEpisode は工場が既に出している配列(obs/action/reward)をそのまま保持し、
PCN 学習が必要とする教師 cache ブロック（observations / actions / desired_returns /
desired_horizons）を**ベクトル化**で前計算して先頭 head に `_pcn_training_block` として
メモ化する。これにより:
  - Ray 直列化 = 数百万オブジェクト → 数個の連続 numpy 配列（plasma zero-copy, 桁で軽い）
  - _encode_episode_training_block はメモを即返す（per-transition Python ループ皆無）
  - _add_episode は逆cumsum を回さない（rtg は前計算済）

**従来 Transition 経路との一致（replay snapshot 実測で確認）**:
学習ターゲット desired_returns は生 reward の「二重」逆cumsum（_add_episode が rtg 化 →
_encode がもう一度逆cumsum）。head[t].reward は「一重」rtg（archive/_choose_commands が読む）。
本クラスは gamma=1.0（learner 既定）で両者をベクトル計算して再現する。

list[Transition] の代替として振る舞う（__len__ / __getitem__ / __iter__）。ReplayBuffer /
Learner の hot path は先頭要素の uid・reward・objective_values・_pcn_training_block しか
参照しない（PCN_FAST_REPLAY_HASH + use_training_cache 経路）ので、[0] は完全な head を、
[t>0] は最小 head を返す。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

_DESIRED_RETURN_CLIP = float(os.environ.get("PCN_DESIRED_RETURN_CLIP", "1e12"))
# [PCN_LABEL_G] pcn_agent._encode_episode_training_block と同一規約で desired_returns を作る。
# LABEL_G=1(教師ラベル二重累積バグの修正版)では格納値 rtg(=G_t) をそのまま教師にする。
# ここを合わせないと _training_block のメモ化が使えず(label_g キー不一致で毎回ミス)、
# 1本 0.9 秒の per-step Python ループに落ちる。値はビット一致(下記 dr の分岐参照)。
_LABEL_G = os.environ.get("PCN_LABEL_G", "0") == "1"
_GAMMA = 1.0  # learner 既定（distributed_pcn.py:1695 self.gamma = 1.0）


def _reverse_cumsum(x: np.ndarray) -> np.ndarray:
    """逆方向累積和（末尾→先頭）。_add_episode(:2015-2018) と同じ順序で f32 累積。

    x[i] + gamma*x[i+1] + ... を各 i について返す（gamma=1）。np.cumsum(反転)[::反転] は
    末尾から順次加算＝_add_episode の逐次加算と同一順序（f32 丸めも一致）。
    """
    return np.cumsum(x[::-1], axis=0)[::-1]


class _Head:
    """1 transition 相当の軽量ビュー（Transition の代替; 参照のみ保持）。"""

    __slots__ = (
        "observation", "action", "reward", "next_observation", "terminal",
        "objective_values", "solution_execution_time", "command_return",
        "_pcn_episode_uid", "random_action_prob", "_pcn_training_block",
    )

    def __init__(self, observation, action, reward, next_observation, terminal):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.terminal = terminal


class FactoryArrayEpisode:
    """配列で 1 エピソードを保持し、list[Transition] のように振る舞う drop-in。"""

    # ダックタイピング用マーカー（hot path が isinstance でなくこれで判定＝import 結合回避）。
    _pcn_array_episode = True

    def __init__(
        self,
        observations: np.ndarray,   # (T, obs_dim) f32  ※既に nan_to_num 済み
        actions: np.ndarray,        # (T,) int64
        desired_returns: np.ndarray,  # (T, R) f32  二重逆cumsum・clip 済み（教師ターゲット）
        desired_horizons: np.ndarray,  # (T,) f32  = arange(T,0,-1)
        rtg: np.ndarray,            # (T, R) f32  一重 rtg（head[t].reward / archive 用）
        objective_values: Any,      # [cost, makespan, avg_wait]
        uid: Optional[str],
        command_return: Optional[np.ndarray] = None,
        solution_execution_time: Optional[float] = None,
        random_action_prob: Optional[float] = None,
    ):
        self.observations = observations
        self.actions = actions
        self.desired_returns = desired_returns
        self.desired_horizons = desired_horizons
        self.rtg = rtg
        self.objective_values = objective_values
        self.uid = uid
        self.command_return = command_return
        self.solution_execution_time = solution_execution_time
        self.random_action_prob = random_action_prob
        self._T = int(actions.shape[0])
        self._head0: Optional[_Head] = None  # 遅延生成・pickle しない

    # ---- list[Transition] 互換 API ----------------------------------------
    def __len__(self) -> int:
        return self._T

    def _training_block(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "desired_returns": self.desired_returns,
            "desired_horizons": self.desired_horizons,
            "episode_length": self._T,
            "label_g": _LABEL_G,  # メモ化キー: _encode 側の判定と揃える（フラグ違いの混入防止）
        }

    def _make_head(self, t: int) -> _Head:
        no = self.observations[t + 1] if (t + 1) < self._T else self.observations[t]
        h = _Head(self.observations[t], int(self.actions[t]), self.rtg[t], no,
                  bool(t == self._T - 1))
        if t == 0:
            h.objective_values = self.objective_values
            h._pcn_episode_uid = self.uid
            h.random_action_prob = self.random_action_prob
            if self.solution_execution_time is not None:
                h.solution_execution_time = self.solution_execution_time
            h.command_return = self.command_return
            # _encode_episode_training_block(:2124) がこれを即返す＝per-transition ループ回避
            h._pcn_training_block = self._training_block()
        return h

    def __getitem__(self, t: int) -> _Head:
        if t == 0 or t == -self._T:
            if self._head0 is None:
                self._head0 = self._make_head(0)
            return self._head0
        if t < 0:
            t += self._T
        return self._make_head(t)

    def __iter__(self):
        for t in range(self._T):
            yield self[t]

    # ---- Ray/pickle: 遅延 head は落とす（配列のみ直列化＝軽量） ----------
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_head0"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._head0 = None


def build_factory_array_episodes(res: Dict[str, Any], *, uids: List[Optional[str]],
                                 command_returns: Optional[List] = None,
                                 solution_execution_time: Optional[float] = None,
                                 random_action_probs: Optional[List] = None
                                 ) -> List[FactoryArrayEpisode]:
    """工場の res(obs/actions/rewards/achieved 配列) → FactoryArrayEpisode のリスト。

    per-transition Python ループ無し（全てベクトル演算）。desired_returns は生 reward の
    二重逆cumsum（従来 Transition 経路と一致）、horizon は arange(T,0,-1)。
    """
    obs = np.asarray(res["obs"], dtype=np.float32)        # (B, T+1, D)
    actions = np.asarray(res["actions"])                   # (B, T) int8
    rewards = np.asarray(res["rewards"], dtype=np.float32)  # (B, T, R)
    achieved = np.asarray(res["achieved"], dtype=np.float64)
    # [defer 工場] 可変長エピソード: res["lengths"](B,) があれば per-episode に切り詰める
    # (defer で長さ = n_jobs + defer 回数。scan は固定長で done 後は 0 パディング)。
    # 無ければ従来どおり全長 T(既定パス不変)。
    lengths = res.get("lengths")
    B, Tp1, D = obs.shape
    T = actions.shape[1]
    out: List[FactoryArrayEpisode] = []
    for b in range(B):
        L = int(lengths[b]) if lengths is not None else T
        # 一重 rtg（= _add_episode 後の stored reward。archive/head[t].reward が読む）
        raw = rewards[b, :L]                               # (L, R) f32
        rtg = _reverse_cumsum(raw).astype(np.float32)      # (L, R)
        # 二重（= _encode の desired_returns）: 一重 rtg を nan_to_num してもう一度逆cumsum
        rtg_clean = np.nan_to_num(rtg, nan=0.0, posinf=0.0, neginf=0.0)
        if _LABEL_G:
            # [PCN_LABEL_G=1] 再累積しない（_encode の `desired_returns[:] = rewards` と同一。
            # rewards[t] = nan_to_num(head.reward) = rtg_clean[t] なのでビット一致）。
            dr = rtg_clean
        else:
            dr = _reverse_cumsum(rtg_clean).astype(np.float32)
        if _DESIRED_RETURN_CLIP > 0:
            dr = np.clip(dr, -_DESIRED_RETURN_CLIP, _DESIRED_RETURN_CLIP)
        obs_block = np.nan_to_num(obs[b, :L], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        horizons = np.arange(L, 0, -1, dtype=np.float32)
        acts = actions[b, :L].astype(np.int64)
        objv = [int(achieved[b, 0]), int(achieved[b, 1]), float(achieved[b, 2])]
        cr = None
        if command_returns is not None:
            cr = np.asarray(command_returns[b], dtype=np.float32)
        rap = None
        if random_action_probs is not None:
            rap = float(random_action_probs[b])
        out.append(FactoryArrayEpisode(
            observations=obs_block, actions=acts, desired_returns=dr,
            desired_horizons=horizons, rtg=rtg, objective_values=objv,
            uid=uids[b], command_return=cr,
            solution_execution_time=solution_execution_time, random_action_prob=rap,
        ))
    return out
