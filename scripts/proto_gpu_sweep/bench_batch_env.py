"""GPU バッチ env の steps/sec ベンチ（B×NJ グリッド）+ CPU 1 コア比・工場概算。

- GPU: rollout（jit + lax.scan, ys=rewards のみ）でエピソード全体を回し、
  steps/sec = B×NJ / 時間（ウォームアップ=コンパイル後、REPEAT 回の best）。
- CPU 1 コア: record_ref_traj.py が npz に記録した本体 env の per_step_us
  （PCN_FAST_ENV=1 = C 版 sweep ON の既定構成）から換算。
- ジョブ列は参照 npz のもの（synth NJ∈{128,512}）を B 本 broadcast。
  行動列は seed 固定 Bernoulli(0.5) の (T,B)。

usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/bench_batch_env.py
  env: BS="256,1024,4096" NJS="128,512" REPEAT=3 KERNEL=v1 N_ACTORS=16
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from batch_env_jax import make_initial_state, rollout, rollout_segmented  # noqa: E402

BS = [int(x) for x in os.environ.get("BS", "256,1024,4096").split(",")]
NJS = [int(x) for x in os.environ.get("NJS", "128,512").split(",")]
REPEAT = int(os.environ.get("REPEAT", "3"))
KERNEL = os.environ.get("KERNEL", "v1")
SEG_LEN = int(os.environ.get("SEG_LEN", "64"))  # 0 = 非セグメント(フル E)
N_ACTORS = int(os.environ.get("N_ACTORS", "16"))  # 現行 Ray 収集の actor 数
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def auto_chunk(B, NJ, n_on, n_cl):
    """v1 カーネル中間（(B,T,E)+(B,T,N) 系）が VRAM に収まる B チャンク幅。

    per-b 中間 ≈ T·(E·5 + (N_on+N_cl)·9) bytes（fp16 ov/einsum 入力 + f32 cnt +
    i16 cs/win 等の概算; A40 46GB・budget 20GB で実測較正）。B を割り切る 2 冪に丸める。
    """
    per_b = (NJ + 1) * (NJ * 5 + (n_on + n_cl) * 9)
    chunk = max(256, min(B, int((20 << 30) // per_b)))
    chunk = 1 << (chunk.bit_length() - 1)   # 2 冪へ切り下げ（B が 2 冪なら割り切る）
    while B % chunk:
        chunk //= 2
    return max(chunk, 1)


def bench_one(jobs, B, n_on, n_cl):
    NJ = jobs.shape[0]
    h_max = int(jobs[:, 2].max())
    chunk = int(os.environ.get("CHUNK", "0")) or auto_chunk(B, NJ, n_on, n_cl)
    rng = np.random.default_rng(123)
    actions = (rng.random((NJ, B)) < 0.5).astype(np.int32)
    acts_dev = [jnp.asarray(actions[:, lo : lo + chunk]) for lo in range(0, B, chunk)]

    def run():
        fs = None
        for a in acts_dev:  # VRAM 超過を避ける B 軸チャンク逐次（同形状 = jit 再利用）
            state = make_initial_state(
                np.broadcast_to(jobs, (chunk, NJ, 8)), n_on=n_on, n_cl=n_cl
            )
            if SEG_LEN > 0:
                fs, (rew, dn) = rollout_segmented(
                    state, a, n_on=n_on, n_cl=n_cl, h_max=h_max,
                    kernel=KERNEL, seg_len=SEG_LEN,
                )
            else:
                fs, rew, dn = rollout(state, a, n_on=n_on, n_cl=n_cl,
                                      h_max=h_max, kernel=KERNEL)
            rew.block_until_ready()
        return fs

    t0 = time.perf_counter()
    fs = run()  # warm = jit コンパイル + 初回実行
    t_warm = time.perf_counter() - t0
    assert bool(fs.done.all()), "episodes did not finish"
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t0)
    return B * NJ / best, best, t_warm, chunk


def main():
    print(f"# GPU batch env bench  kernel={KERNEL} seg_len={SEG_LEN} repeat={REPEAT}")
    print(f"# device: {jax.devices()}")
    rows = []
    for NJ in NJS:
        ref_path = os.path.join(DATA_DIR, f"ref_nj{NJ}_seed0_fast1.npz")
        ref = np.load(ref_path)
        jobs = ref["jobs"]
        n_on, n_cl = int(ref["n_on"]), int(ref["n_cl"])
        cpu_sps = 1e6 / float(ref["per_step_us"])   # 本体 env 1 コア steps/sec
        for B in BS:
            sps, t_ep, t_warm, chunk = bench_one(jobs, B, n_on, n_cl)
            rows.append((NJ, B, sps, t_ep, t_warm, cpu_sps))
            print(
                f"NJ={NJ:4d} B={B:5d}: {sps:10.0f} steps/s "
                f"(episode-batch {t_ep*1e3:.0f} ms, warm+compile {t_warm:.1f}s, "
                f"chunk {chunk})  "
                f"CPU1core {cpu_sps:.0f} steps/s -> x{sps/cpu_sps:.1f}",
                flush=True,
            )

    print("\n## summary")
    print("| NJ | B | GPU steps/s | CPU 1core steps/s | xCPU1core | x(Ray %d actor) |"
          % N_ACTORS)
    print("|---|---|---|---|---|---|")
    for NJ, B, sps, t_ep, t_warm, cpu_sps in rows:
        print(f"| {NJ} | {B} | {sps:.0f} | {cpu_sps:.0f} "
              f"| x{sps/cpu_sps:.1f} | x{sps/(cpu_sps*N_ACTORS):.1f} |")

    print(f"""
## rollout 工場概算（env 部分のみ; NN 側は含めない）
現行 Ray {N_ACTORS} actor CPU 収集の env スループット ≈ {N_ACTORS} × CPU1core。
GPU 工場倍率 = GPU steps/s ÷ ({N_ACTORS} × CPU1core steps/s) … 上表最終列。
ロールアウト全体では 学習時 rollout の内訳が NN 69% / env 30%（実測メモ）なので、
env を GPU 化しただけの全体倍率 ≈ 1 / (0.69 + 0.30/上表倍率) が上限目安
（NN も同バッチ GPU 化（Phase 2）で初めてフルに効く）。""")


if __name__ == "__main__":
    main()
