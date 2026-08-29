"""Phase 2 ベンチ: 学習中eval（指令格子 greedy）の GPU バッチ vs 既存 CPU 逐次。

同一タスク = 「1モデル × NCMD 指令 greedy 評価」（NCMD=40: eval_b2_compare 相当 /
NCMD=372: 学習中 eval_gap 格子相当）を
  - GPU バッチ（batch_eval_jax; B=NCMD 同時, コンパイル後の warm 2回目を計測）
  - CPU 逐次 1 コア（ag._run_episode; verify_batch_eval の参照キャッシュ実測から
    per-command 時間を採り NCMD へ外挿。REF_MEASURE=1 で今実測し直しも可）
で比較する。NPROC=16 並列換算 = 逐次/16（楽観上限。実際は fork 並列でメモリ帯域
飽和 ~3× 上限の実測経験あり → 表では両方を出す）。

usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/bench_batch_eval.py
  env: NJS=128,512 NCMDS=40,372 KERNEL=v1 SEED=1 TIE_EPS=1e-3 REPEAT=2
"""
from __future__ import annotations

import glob
import os
import sys
import time

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
os.environ["SCHEDULER_OBS_URGENCY"] = "1"
os.environ["SCHEDULER_OBS_OCCUPANCY"] = "0"
# Fourier/FILM は環境変数で切替可能に（未指定なら基本モデル=従来動作）。
os.environ.setdefault("PCN_FILM", "0")
os.environ.setdefault("PCN_FOURIER_CMD", "0")
os.environ.setdefault("PCN_FOURIER_BANDS", "4")
os.environ["PCN_OBS_LOG"] = "1"
os.environ.setdefault("PCN_HIDDEN_DIM", "512")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

MAIN_REPO = "/home/noguchi/scheduler-sim-for-cb"
NJS = [int(x) for x in os.environ.get("NJS", "128,512").split(",")]
NCMDS = [int(x) for x in os.environ.get("NCMDS", "40,372").split(",")]
KERNEL = os.environ.get("KERNEL", "v1")
SEED = int(os.environ.get("SEED", "1"))
TIE_EPS = float(os.environ.get("TIE_EPS", "1e-3"))
REPEAT = int(os.environ.get("REPEAT", "2"))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def default_ckpt(nj: int) -> str:
    # CKPT_PAT 環境変数で ckpt グロブを上書き可能（{nj} 展開）。既定は基本モデル nded。
    pat = os.environ.get(
        "CKPT_PAT",
        f"{MAIN_REPO}/experiments/distributed_pcn/"
        f"run_synth{{nj}}_nded{{nj}}_r1/*/iteration_100/model_iter_100.pth",
    ).replace("{nj}", str(nj))
    hits = sorted(glob.glob(pat))
    assert hits, pat
    return hits[-1]


def load_ref(nj: int, ckpt: str):
    """verify_batch_eval が書いた参照キャッシュを ckpt の tag に紐付けて読む（モデル取り違え防止）。"""
    tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))  # 例 20260606_020951
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"refeval_nj{nj}_seed{SEED}_ncmd40_{tag}.npz")))
    assert hits, f"run verify_batch_eval.py first (NJ={nj}, tag={tag})"
    return {k: v for k, v in np.load(hits[-1]).items()}


def expand_commands(cmds40: np.ndarray, n: int) -> np.ndarray:
    """40 指令の cost/wait 直線を n 本へ再サンプル（timing 用; 値の意味は同じ帯）。"""
    x = np.linspace(0, 1, len(cmds40))
    xi = np.linspace(0, 1, n)
    return np.stack(
        [np.interp(xi, x, cmds40[:, j]) for j in range(2)], axis=1
    ).astype(np.float32)


def main():
    import torch as th

    from batch_eval_jax import HybridGreedyPolicy, run_batch_eval

    print(f"# GPU batch eval bench kernel={KERNEL} tie_eps={TIE_EPS} repeat={REPEAT}")
    rows = []
    for nj in NJS:
        ckpt = default_ckpt(nj)
        ref = load_ref(nj, ckpt)
        jobs = ref["jobs"]
        n_on, n_cl, n_window = int(ref["n_on"]), int(ref["n_cl"]), int(ref["n_window"])
        cpu_per_cmd = float(ref["ref_wall_s"]) / len(ref["commands"])  # 逐次1コア実測
        policy = HybridGreedyPolicy(
            ckpt, nj,
            device="cuda" if th.cuda.is_available() else "cpu", tie_eps=TIE_EPS,
        )
        for ncmd in NCMDS:
            cmds = expand_commands(ref["commands"], ncmd)
            best, stats = None, None
            for rep in range(REPEAT + 1):  # +1 = コンパイル込み warm
                policy.reset_stats()
                t0 = time.perf_counter()
                res = run_batch_eval(
                    jobs, list(cmds), policy, n_on=n_on, n_cl=n_cl,
                    n_window=n_window, kernel=KERNEL,
                )
                dt = time.perf_counter() - t0
                if rep == 0:
                    warm = dt
                elif best is None or dt < best:
                    best, stats = dt, res["stats"]
            cpu_seq = cpu_per_cmd * ncmd
            rows.append((nj, ncmd, best, warm, cpu_seq, stats))
            print(
                f"NJ={nj:4d} NCMD={ncmd:4d}: GPU {best:7.2f}s (warm+compile {warm:.1f}s; "
                f"env {stats['t_env']:.1f}s obs {stats['t_obs']:.1f}s nn {stats['t_nn']:.1f}s) "
                f"CPU1core {cpu_seq:7.1f}s -> x{cpu_seq / best:5.1f} "
                f"(vs 16並列換算 x{cpu_seq / 16 / best:4.2f}) "
                f"recheck={stats['n_recheck']}/{stats['n_rows']}",
                flush=True,
            )

    print("\n## summary（wall秒; CPU=逐次1コア実測からの外挿, 16par=/16 の楽観換算）")
    print("| NJ | NCMD | GPU batch | CPU 1core | x1core | CPU 16par | x16par |")
    print("|---|---|---|---|---|---|---|")
    for nj, ncmd, best, warm, cpu_seq, stats in rows:
        print(
            f"| {nj} | {ncmd} | {best:.1f}s | {cpu_seq:.1f}s | x{cpu_seq / best:.1f} "
            f"| {cpu_seq / 16:.1f}s | x{cpu_seq / 16 / best:.2f} |"
        )


if __name__ == "__main__":
    main()
