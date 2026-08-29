"""GPU バッチ sweep vs C 版逐次のマイクロベンチ（alloc/sec と倍率の B×E グリッド）。

- GPU: 定常スループット＝データ device 常駐でカーネルのみ（jit コンパイルは分離報告）。
  参考として H2D/D2H 転送込み(solve_batch_gpu 素通し)も 1 回測る。
- C 版: 同一問題インスタンス群を pybind 経由で 1 問ずつ逐次（1 コア）。B に依存しないので
  E×N 構成ごとに 1 回測り、全 B 行で共通の分母にする。
- density=1.0（中〜高競合）。構成は (N=256, cont=0=オンプレ相当) と (N=1024, cont=1=クラウド相当)。

usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/bench.py
env: BS="256,1024,4096" ES="64,256,1024" REPEAT=3 DENSITY=1.0
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import c_ref  # noqa: E402
import gen_problems  # noqa: E402
from gpu_sweep_jax import (  # noqa: E402
    _pad_chunk,
    _sweep_alloc_kernel,
    _sweep_alloc_kernel_v2,
    _sweep_alloc_kernel_v3,
    default_chunk,
    solve_batch_gpu,
)

import jax  # noqa: E402

BS = [int(x) for x in os.environ.get("BS", "256,1024,4096").split(",")]
ES = [int(x) for x in os.environ.get("ES", "64,256,1024").split(",")]
REPEAT = int(os.environ.get("REPEAT", "3"))
DENSITY = float(os.environ.get("DENSITY", "1.0"))
KERNELS = os.environ.get("KERNELS", "v1,v2").split(",")
CONFIGS = [(256, False), (1024, True)]  # (n_nodes, continuous_only)


def bench_gpu_resident(batch, h_max, kernel="v1"):
    """チャンク分割済み device 常駐データでカーネル定常スループットを測る。"""
    B = batch["starts"].shape[0]
    E = batch["starts"].shape[1]
    N = int(batch["n_nodes"])
    kfn = {"v1": _sweep_alloc_kernel, "v2": _sweep_alloc_kernel_v2,
           "v3": _sweep_alloc_kernel_v3}[kernel]
    chunk = min(default_chunk(E, N), B)
    cont = np.full(B, bool(batch["continuous_only"]))
    dev_chunks = []
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        args = _pad_chunk(batch, cont, lo, hi, chunk)
        for a in args:
            a.block_until_ready()
        dev_chunks.append(args)

    def run_all():
        outs = [kfn(*args, h_max=h_max) for args in dev_chunks]
        for o in outs:
            o[0].block_until_ready()

    t0 = time.perf_counter()
    run_all()  # warm = jit コンパイル込み
    t_compile = time.perf_counter() - t0
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        run_all()
        best = min(best, time.perf_counter() - t0)
    return B / best, best, t_compile, chunk


def main():
    print(f"# GPU batch sweep micro-benchmark  density={DENSITY} repeat={REPEAT}")
    print(f"# device: {jax.devices()}")
    rows = []
    c_cache = {}
    for n_nodes, cont in CONFIGS:
        for E in ES:
            # C 版逐次: 最大 B の問題群で 1 回（B 非依存・全行共通）
            key = (E, n_nodes, cont)
            batch_max = gen_problems.gen_batch_fast(
                max(BS), E, n_nodes, cont, seed=7000 + E + n_nodes, density=DENSITY
            )
            if key not in c_cache:
                n_c = min(2048, max(BS))  # C は線形なので上限 2048 問で十分
                probs = [gen_problems.to_csr(batch_max, b) for b in range(n_c)]
                t0 = time.perf_counter()
                c_ref.solve_batch(probs)
                dt = time.perf_counter() - t0
                c_cache[key] = n_c / dt
                print(f"C_SEQ  E={E:5d} N={n_nodes:5d} cont={int(cont)}: "
                      f"{c_cache[key]:10.0f} alloc/s ({dt/n_c*1e6:.1f} us/alloc, {n_c} probs)")
            c_rate = c_cache[key]
            h_max = int(batch_max["height"].max())
            for B in BS:
                batch = {
                    k: (v[:B] if isinstance(v, np.ndarray) else v) for k, v in batch_max.items()
                }
                for kernel in KERNELS:
                    g_rate, g_t, t_comp, chunk = bench_gpu_resident(batch, h_max, kernel)
                    # 参考: H2D/D2H 転送込み（ウォームアップ後 1 回）
                    solve_batch_gpu(batch, h_max=h_max, kernel=kernel)
                    t0 = time.perf_counter()
                    solve_batch_gpu(batch, h_max=h_max, kernel=kernel)
                    t_xfer = time.perf_counter() - t0
                    rows.append((kernel, n_nodes, cont, E, B, c_rate, g_rate,
                                 g_rate / c_rate, B / t_xfer, t_comp, chunk))
                    print(f"GPU-{kernel} E={E:5d} N={n_nodes:5d} cont={int(cont)} B={B:5d}: "
                          f"{g_rate:10.0f} alloc/s  x{g_rate/c_rate:7.1f} vs C  "
                          f"(steady {g_t*1e3:.1f} ms, xfer-incl {B/t_xfer:.0f} a/s, "
                          f"compile {t_comp:.2f}s, chunk {chunk})", flush=True)

    print("\n## summary (alloc/sec, xN = vs C 1-core sequential)")
    print("| N,cont | E | B | C alloc/s | GPU-v1 | xN | GPU-v2 | xN | v2(転送込) |")
    print("|---|---|---|---|---|---|---|---|---|")
    byk = {}
    for kernel, n_nodes, cont, E, B, c_rate, g_rate, ratio, x_rate, t_comp, chunk in rows:
        byk.setdefault((n_nodes, cont, E, B), {})[kernel] = (c_rate, g_rate, ratio, x_rate)
    for (n_nodes, cont, E, B), d in byk.items():
        c_rate = next(iter(d.values()))[0]
        v1 = d.get("v1", (0, 0, 0, 0))
        v2 = d.get("v2", (0, 0, 0, 0))
        print(f"| {n_nodes},{int(cont)} | {E} | {B} | {c_rate:.0f} "
              f"| {v1[1]:.0f} | x{v1[2]:.1f} | {v2[1]:.0f} | x{v2[2]:.1f} | {v2[3]:.0f} |")


if __name__ == "__main__":
    main()
