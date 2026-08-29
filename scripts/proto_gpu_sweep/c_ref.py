"""C 版 event_sweep_alloc（scheduling_env_core, ビルド済み .so）の逐次呼びラッパ＋スループット計測。

GPU バッチプロトタイプとの直接比較軸:「同一問題インスタンス群を C 版で1問ずつ解いた alloc/sec」。
pybind11 呼び出しオーバーヘッド（~µs/call）込み＝実運用（env から都度呼ぶ）と同じ条件。
"""
from __future__ import annotations

import os
import sys
import time

# ビルド済み C 拡張の場所（worktree にはソースのみなので本体側 .so を使う; ソースは diff で同一確認済み）
_SO_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "envs", "scheduling_native"),
    "/home/noguchi/scheduler-sim-for-cb/src/envs/scheduling_native",
]
for _d in _SO_DIRS:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

from scheduling_env_core import event_sweep_alloc as _c_alloc  # noqa: E402


def solve_one(p):
    """問題 dict(gen_problems.to_csr の出力)を C 版で解く → (start, is_contig, nodes list)。"""
    return _c_alloc(
        p["starts"], p["ends"], p["nodes_flat"], p["node_off"],
        p["width"], p["height"], p["n_nodes"], p["continuous_only"], p["arrival"],
    )


def solve_batch(problems):
    return [solve_one(p) for p in problems]


def bench_throughput(problems, repeat: int = 3):
    """逐次スループット計測。CSR 変換はプリコンパイル済み前提（引数が CSR 済みリスト）。"""
    solve_batch(problems[: min(8, len(problems))])  # warm
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        solve_batch(problems)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return len(problems) / best, best
