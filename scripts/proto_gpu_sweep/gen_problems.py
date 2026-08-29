"""GPU バッチ sweep-line 配置探索プロトタイプ用の問題ジェネレータ（seed 固定・再現可能）。

1問題 = 「既存イベント集合(時間区間+占有ノード集合) + 1ジョブ(width, height, arrival)」。
実 env の分布に厳密に合わせる必要はないが、
  - イベント数は E_max を上限に問題ごとに可変（[E_max/2, E_max]）→ パディング＋有効マスクの検証を兼ねる
  - 占有ノードは連続矩形/非連続集合の混在（実 env のオンプレ分散割当を模す）
  - 競合度は density で調整（イベント区間長・占有幅を伸ばすと候補時刻を深く舐める＝C 版が遅い側）
とし、C 版とのタイブレーク（最低位ノード・最早時刻）が効く同値ケースが自然に出る密度にする。

出力はバッチ構造体（dict of np.ndarray）:
  starts   (B, E_max) int64   無効イベントは start=+INF64/ends=-1（overlap/候補の両条件から自然に排除）
  ends     (B, E_max) int64
  occ      (B, E_max, N) bool 各イベントの占有ノード dense マスク（無効行は全 False）
  n_events (B,) int32
  width/height/arrival (B,) int64/int32/int64
  continuous_only: バッチ定数(bool)、n_nodes: バッチ定数(int)
CSR 形式（C 版入力）へは to_csr() で problem 単位に変換する。
"""
from __future__ import annotations

import numpy as np

INF64 = np.int64(1) << 60  # 無効イベント start（cand 上限 2^60 未満に収まる値域で生成する）
HORIZON = 1 << 20          # 時刻値域（int32 内; end = start+dur も余裕で int64 安全）


def gen_batch(
    B: int,
    E_max: int,
    n_nodes: int,
    continuous_only: bool,
    seed: int,
    density: float = 1.0,
    edge_cases: bool = False,
):
    """B 問題のバッチを生成。edge_cases=True で先頭に境界ケースを混ぜる。"""
    rng = np.random.default_rng(seed)
    starts = np.full((B, E_max), INF64, dtype=np.int64)
    ends = np.full((B, E_max), -1, dtype=np.int64)
    occ = np.zeros((B, E_max, n_nodes), dtype=bool)
    n_events = rng.integers(E_max // 2, E_max + 1, size=B).astype(np.int32)

    # ジョブ側: width/height/arrival
    width = rng.integers(1, HORIZON // 8, size=B).astype(np.int64)
    height = rng.integers(1, max(2, n_nodes // 4), size=B).astype(np.int32)
    arrival = rng.integers(0, int(HORIZON * 0.8), size=B).astype(np.int64)

    # イベント側: 区間と占有ノード
    # 区間長は density で伸縮（長い区間ほど同時刻の重なりが増え高競合）
    dur_hi = max(2, int(HORIZON // 4 * density))
    for b in range(B):
        ne = int(n_events[b])
        if ne == 0:
            continue
        s = rng.integers(0, HORIZON, size=ne)
        d = rng.integers(1, dur_hi, size=ne)
        starts[b, :ne] = s
        ends[b, :ne] = s + d
        # 占有: 連続矩形 or 非連続集合（半々）。幅は N/8 上限を density で伸縮。
        h_hi = max(2, int(n_nodes // 8 * density) + 1)
        hs = rng.integers(1, h_hi, size=ne)
        contig = rng.random(ne) < 0.5
        for e in range(ne):
            h = int(min(hs[e], n_nodes))
            if contig[e]:
                base = int(rng.integers(0, n_nodes - h + 1))
                occ[b, e, base : base + h] = True
            else:
                idx = rng.choice(n_nodes, size=h, replace=False)
                occ[b, e, idx] = True

    if edge_cases and B >= 8:
        # b=0: イベント 0 個
        n_events[0] = 0
        starts[0, :] = INF64
        ends[0, :] = -1
        occ[0, :, :] = False
        # b=1: height > n_nodes（フォールバック強制）
        height[1] = n_nodes + 3
        # b=2: arrival > 全 ends（候補が arrival のみ）
        ne = int(n_events[2])
        if ne > 0:
            arrival[2] = int(ends[2, :ne].max()) + 10
        # b=3: height == n_nodes（全ノード要求）
        height[3] = n_nodes
        # b=4: width=1 の最小ジョブ
        width[4] = 1
        height[4] = 1
        # b=5: 全ノードを常時占有するイベント1個（arrival では絶対置けない）
        n_events[5] = 1
        starts[5, :] = INF64
        ends[5, :] = -1
        occ[5, :, :] = False
        starts[5, 0] = 0
        ends[5, 0] = HORIZON
        occ[5, 0, :] = True
        arrival[5] = 100
        # b=6: 同一 end が多数（候補時刻の重複）
        ne = int(n_events[6])
        if ne > 4:
            ends[6, : ne // 2] = ends[6, 0]
        # b=7: arrival == あるイベントの end（境界タイ）
        ne = int(n_events[7])
        if ne > 0:
            arrival[7] = int(ends[7, 0])

    return {
        "starts": starts,
        "ends": ends,
        "occ": occ,
        "n_events": n_events,
        "width": width,
        "height": height,
        "arrival": arrival,
        "continuous_only": continuous_only,
        "n_nodes": n_nodes,
    }


def gen_batch_fast(
    B: int,
    E_max: int,
    n_nodes: int,
    continuous_only: bool,
    seed: int,
    density: float = 1.0,
):
    """ベンチ用の全ベクトル化ジェネレータ（gen_batch と同構造・同値域）。

    唯一の差: 占有ノード集合を「1〜2個の連続 run の和集合」で表す（choice の per-event
    Python ループを排除）。C 版の負荷（per-event ノード数に線形）・GPU 版の負荷
    （dense 積和はパターン非依存）とも代表性は保たれる。正しさの検証（任意集合）は
    verify.py + gen_batch 側で担保済み。
    """
    rng = np.random.default_rng(seed)
    n_events = rng.integers(E_max // 2, E_max + 1, size=B).astype(np.int32)
    ev_valid = np.arange(E_max)[None, :] < n_events[:, None]  # (B,E)

    s = rng.integers(0, HORIZON, size=(B, E_max)).astype(np.int64)
    dur_hi = max(2, int(HORIZON // 4 * density))
    d = rng.integers(1, dur_hi, size=(B, E_max)).astype(np.int64)
    starts = np.where(ev_valid, s, INF64)
    ends = np.where(ev_valid, s + d, np.int64(-1))

    h_hi = max(2, int(n_nodes // 8 * density) + 1)
    n = np.arange(n_nodes, dtype=np.int32)

    def runmask(base, h):
        return (n[None, None, :] >= base[:, :, None]) & (
            n[None, None, :] < (base + h)[:, :, None]
        )

    h1 = rng.integers(1, h_hi, size=(B, E_max)).astype(np.int32)
    b1 = rng.integers(0, n_nodes - h1 + 1).astype(np.int32)
    occ = runmask(b1, h1)
    two = rng.random((B, E_max)) < 0.5
    h2 = rng.integers(1, h_hi, size=(B, E_max)).astype(np.int32)
    b2 = rng.integers(0, n_nodes - h2 + 1).astype(np.int32)
    occ |= runmask(b2, h2) & two[:, :, None]
    occ &= ev_valid[:, :, None]

    width = rng.integers(1, HORIZON // 8, size=B).astype(np.int64)
    height = rng.integers(1, max(2, n_nodes // 4), size=B).astype(np.int32)
    arrival = rng.integers(0, int(HORIZON * 0.8), size=B).astype(np.int64)
    return {
        "starts": starts,
        "ends": ends,
        "occ": occ,
        "n_events": n_events,
        "width": width,
        "height": height,
        "arrival": arrival,
        "continuous_only": continuous_only,
        "n_nodes": n_nodes,
    }


def to_csr(batch, b: int):
    """問題 b を C 版 event_sweep_alloc の入力（CSR）へ変換（flatnonzero 1 回で構築）。"""
    ne = int(batch["n_events"][b])
    N = int(batch["n_nodes"])
    starts = np.ascontiguousarray(batch["starts"][b, :ne])
    ends = np.ascontiguousarray(batch["ends"][b, :ne])
    occ = batch["occ"][b, :ne]  # (ne, N)
    if ne:
        flat = np.flatnonzero(occ.reshape(-1))
        nodes_flat = (flat % N).astype(np.int32)
        node_off = np.zeros(ne + 1, dtype=np.int32)
        np.cumsum(occ.sum(axis=1, dtype=np.int32), out=node_off[1:])
    else:
        nodes_flat = np.empty(0, dtype=np.int32)
        node_off = np.zeros(1, dtype=np.int32)
    return dict(
        starts=starts,
        ends=ends,
        nodes_flat=nodes_flat,
        node_off=node_off,
        width=int(batch["width"][b]),
        height=int(batch["height"][b]),
        n_nodes=int(batch["n_nodes"]),
        continuous_only=bool(batch["continuous_only"]),
        arrival=int(batch["arrival"][b]),
    )
