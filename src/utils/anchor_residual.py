"""アンカー残差方策 (anchor-residual policy) のアンカー選択ロジック。

行動空間を「ゼロから配置」でなく「参照アンカー解からの逸脱(flip)」へ再定義するための
単一情報源。指令(desired_return)または達成値(cost, avg_wait)から、最も近い
アンカー解(NSGA-II PF点の遺伝子など)を決定論的に選ぶ。

設計の要点:
- 正規化定数は供給された pf 自身から固定する → 学習Actor / 学習中eval / evalスクリプトの
  3経路で「指令→アンカー」の対応が完全一致する(ずるなし: evalでも指令のみから決定)。
- AnchorSet は (pf, genes) 配列を直接受け取る = アンカー源(NSGA-II / ヒューリスティック /
  別方策)に依存しない。npz ローダーは from_npz の薄いファクトリ。
- gate: 環境変数 PCN_ANCHOR_RESIDUAL=<npzパス>(未設定なら get_anchor_set()=None で全経路OFF)。

指令⇔値の変換は src/utils/pf_command_eval.objectives_to_command と厳密に整合:
    desired_return = [-avg_wait * nj, -cost]
    → 逆変換: cost = -dr[1],  avg_wait = -dr[0] / nj
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-9


class AnchorSet:
    """アンカー解の集合と、指令/達成値からの最近傍選択。

    Parameters
    ----------
    pf : (M, 2) array  各アンカーの目的値 (cost, avg_wait)。
    genes : (M, NJ) int  各アンカーのジョブ順 0/1 割当(scheduled時のみ index 前進)。
    """

    def __init__(self, pf: np.ndarray, genes: np.ndarray):
        pf = np.asarray(pf, dtype=np.float64)
        genes = np.asarray(genes, dtype=np.int8)
        if pf.ndim != 2 or pf.shape[1] != 2:
            raise ValueError(f"pf は (M,2) を期待: {pf.shape}")
        if genes.ndim != 2 or genes.shape[0] != pf.shape[0]:
            raise ValueError(f"genes は (M,NJ) で pf と同数の行を期待: {genes.shape} vs {pf.shape}")

        # 遺伝子の行単位重複除去(先勝ち)→ pf と整合維持 → cost 昇順ソート
        # (PHASE1_NSGA の種まき規約と揃える)
        _, uidx = np.unique(genes, axis=0, return_index=True)
        uidx = np.sort(uidx)
        pf, genes = pf[uidx], genes[uidx]
        order = np.argsort(pf[:, 0], kind="stable")
        self.pf = pf[order]
        self.genes = genes[order]
        self.nj = int(self.genes.shape[1])

        # 正規化定数を pf 自身から固定(実行時データから作らない=3経路で一致)
        self._c_lo, self._c_hi = float(self.pf[:, 0].min()), float(self.pf[:, 0].max())
        self._w_lo, self._w_hi = float(self.pf[:, 1].min()), float(self.pf[:, 1].max())
        self._c_rng = max(self._c_hi - self._c_lo, _EPS)
        self._w_rng = max(self._w_hi - self._w_lo, _EPS)
        # 正規化済みアンカー座標(選択のたびに再計算しない)
        self._pf_cn = (self.pf[:, 0] - self._c_lo) / self._c_rng
        self._pf_wn = (self.pf[:, 1] - self._w_lo) / self._w_rng

    def _nearest(self, cost: float, avg_wait: float) -> Tuple[int, np.ndarray]:
        cn = (float(cost) - self._c_lo) / self._c_rng
        wn = (float(avg_wait) - self._w_lo) / self._w_rng
        d2 = (self._pf_cn - cn) ** 2 + (self._pf_wn - wn) ** 2
        i = int(np.argmin(d2))  # argmin=最小indexで決定的タイブレーク
        return i, self.genes[i]

    def select(self, desired_return) -> Tuple[int, np.ndarray]:
        """指令(desired_return=[-wait*nj, -cost])から最近傍アンカーを選ぶ。"""
        dr = np.asarray(desired_return, dtype=np.float64).ravel()
        cost = -float(dr[1])
        avg_wait = -float(dr[0]) / max(1, self.nj)
        return self._nearest(cost, avg_wait)

    def select_by_values(self, cost: float, avg_wait: float) -> Tuple[int, np.ndarray]:
        """達成値(cost, avg_wait)から最近傍アンカーを選ぶ(事後リアンカー用)。"""
        return self._nearest(cost, avg_wait)

    @classmethod
    def from_npz(cls, path: str) -> "AnchorSet":
        """NSGA-II 出力 npz (pf, chromosomes) からアンカー集合を構築。"""
        d = np.load(path, allow_pickle=True)
        return cls(pf=d["pf"], genes=d["chromosomes"])


# --- プロセス毎の遅延シングルトン ---
_AR: Optional[AnchorSet] = None
_AR_LOADED = False


def get_anchor_set() -> Optional[AnchorSet]:
    """PCN_ANCHOR_RESIDUAL が指す npz からアンカー集合を返す。未設定なら None(=OFF)。

    Ray worker / eval fork worker はモジュール再import時に環境変数を継承するため、
    各プロセスで初回呼び出し時に1回だけロードする。
    """
    global _AR, _AR_LOADED
    if _AR_LOADED:
        return _AR
    _AR_LOADED = True
    path = os.environ.get("PCN_ANCHOR_RESIDUAL", "").strip()
    if not path:
        _AR = None
    else:
        _AR = AnchorSet.from_npz(path)
    return _AR
