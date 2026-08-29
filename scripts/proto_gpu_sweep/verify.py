"""C 版 event_sweep_alloc と GPU バッチ実装の全問一致検証。

E∈{64,256,1024} × (n_nodes, continuous_only)∈{(256,False),(1024,True)} の全組 × 各組
B 問（既定 200; 先頭 8 問は境界ケース: 空イベント/height>n_nodes/arrival>全end/全ノード
常時占有/候補重複/境界タイ等）で (start_time, is_contiguous, nodes[:height]) を突き合わせる。

usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/verify.py [B]
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import c_ref  # noqa: E402
import gen_problems  # noqa: E402
from gpu_sweep_jax import solve_batch_gpu  # noqa: E402


def verify_combo(B, E, n_nodes, cont_only, seed, density, kernel):
    batch = gen_problems.gen_batch(
        B, E, n_nodes, cont_only, seed=seed, density=density, edge_cases=True
    )
    g_start, g_contig, g_nodes = solve_batch_gpu(batch, kernel=kernel)
    n_mismatch = 0
    fallback = 0
    for b in range(B):
        c_start, c_contig, c_nodes = c_ref.solve_one(gen_problems.to_csr(batch, b))
        h = int(batch["height"][b])
        gn = g_nodes[b, :h].tolist()
        if not (
            int(g_start[b]) == int(c_start)
            and bool(g_contig[b]) == bool(c_contig)
            and gn == list(c_nodes)
        ):
            n_mismatch += 1
            if n_mismatch <= 3:
                print(
                    f"  MISMATCH b={b}: C=({c_start},{c_contig},{list(c_nodes)[:8]}...) "
                    f"GPU=({int(g_start[b])},{bool(g_contig[b])},{gn[:8]}...) h={h}"
                )
        # フォールバック発火数（カバレッジ確認用）: 全 free でも height>n_nodes 等
        if h > n_nodes:
            fallback += 1
    return n_mismatch, fallback


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    combos = [
        (E, n_nodes, cont)
        for E in (64, 256, 1024)
        for (n_nodes, cont) in ((256, False), (1024, True))
    ]
    total = 0
    bad = 0
    for kernel in ("v1", "v2"):
        for i, (E, n_nodes, cont) in enumerate(combos):
            for density in (0.5, 2.0):  # 低競合/高競合の両側
                nm, fb = verify_combo(
                    B, E, n_nodes, cont,
                    seed=1000 + i * 10 + int(density * 2), density=density, kernel=kernel,
                )
                total += B
                bad += nm
                print(
                    f"{kernel} E={E:5d} N={n_nodes:5d} cont={int(cont)} density={density}: "
                    f"{B - nm}/{B} match (fallback cases={fb})"
                )
    print(f"\nTOTAL: {total - bad}/{total} match -> {'ALL OK' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
