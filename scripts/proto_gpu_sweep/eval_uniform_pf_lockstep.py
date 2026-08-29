"""一様指令グリッドの到達PF評価を lockstep(GPU) で行う高速版。

eval_uniform_command_pf.py と同じグリッド(r0×r1 直積、build_r1_grid 使用)を、
CPU逐次 _run_episode(≈7分/本)の代わりに lockstep greedy(B本を1チャンク定額≈11分)で回す。
greedy の CPU⇔lockstep 一致は verify_lockstep_nn.py で全ステップ一致検証済み。

usage:
  CUDA_HOME=$PWD/tools/nvcc122/nvidia/cuda_nvcc CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:scripts/proto_gpu_sweep \
    .venv/bin/python scripts/proto_gpu_sweep/eval_uniform_pf_lockstep.py \
      --checkpoint <ckpt.pth> --output <dir> --grid 16 [--label v9]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
# [2026-08-30] この "1" は CPU 側 env(ジョブ列の取り出しにしか使わない)向けで、lockstep の
# 正確性には影響しない: 観測は kernel が全幅で構築し、モデルは checkpoint の入力次元
# (s_emb.0.weight)ぶんだけ先頭から読む(lockstep_nn の R5 スライス)。221次元 checkpoint も
# フラグ変更なしでそのまま評価できる。
os.environ.setdefault("SCHEDULER_OBS_EFFICIENCY", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_NATIVE", "1")
os.environ.setdefault("PCN_OBS_LOG", "1")
os.environ.setdefault("PCN_FOURIER_CMD", "1")
os.environ.setdefault("PCN_FC_DEPTH", "4")

# [2026-08-28] 条件付けの前処理フラグは pcn_agent のモジュール定数として import 時に焼き込まれ、
# 未設定だと checkpoint の command_balance が適用されない(cost 側で10倍・wait 側で6倍ずれ、
# Fourier がエイリアスを起こして指令がノイズになる)。実際にこれで測定を1本無駄にした。
# 既定値で黙って走らせず、学習時の値を明示させる。学習runの v9_env_export.sh を source すれば通る。
_REQUIRED = ("PCN_COMMAND_BALANCE", "PCN_COND_WAIT_ROBUST", "PCN_COND_WAIT_Z0")
_missing = [k for k in _REQUIRED if k not in os.environ]
if _missing and os.environ.get("PCN_EVAL_ALLOW_DEFAULTS") != "1":
    raise SystemExit(
        f"[eval_uniform_pf_lockstep] 必須フラグが未設定: {', '.join(_missing)}\n"
        "  学習時と条件付けが食い違うと指令がノイズになり、測定が無効になります。\n"
        "  例: set -a; source experiments/distributed_pcn/<run>/v9_env_export.sh; set +a\n"
        "  意図的に既定値で走らせるなら PCN_EVAL_ALLOW_DEFAULTS=1")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--label", default="lockstep")
    ap.add_argument("--emax", type=int, default=8192)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--commands-npz", default="",
                    help="(archive_nd.npz等) archive_nd の (cost, avg_wait) を指令に使う"
                         "自己再現テスト(論文の eval プロトコル相当)。指定時は grid 無視")
    args = ap.parse_args()

    from scripts.pcn_replay_snapshot import load_config, create_eval_env  # noqa: E402
    from src.utils.pf_uniform_plot import build_r1_grid  # noqa: E402
    from lockstep_nn import load_policy_model, run_lockstep_greedy  # noqa: E402

    cfg = load_config(
        "experiments/distributed_pcn/job_trace_weekB_head50000_cap48000_pcn.yml")
    env = create_eval_env(cfg, job_seed=0, n_jobs=50000)
    env.reset()
    jobs = np.ascontiguousarray(np.asarray(env.jobs, dtype=np.float64))
    n_on, n_cl = int(env.n_on_premise_node), int(env.n_cloud_node)
    n_jobs = jobs.shape[0]

    # 指令レンジは checkpoint の正規化バッファから(eval_uniform_command_pf と同規約)
    st = th.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    scale = sd["desired_return_scale"].numpy().astype(np.float64)
    cost_max = float(scale[1])
    wait_max = float(scale[0]) / max(1, n_jobs)
    print(f"[range] cost_max={cost_max:.3g} avgwait_max={wait_max:.1f} n_jobs={n_jobs}",
          flush=True)

    if args.commands_npz:
        _pts = np.load(args.commands_npz)["archive_nd"]  # (N,2)=[total_cost, avg_wait]
        commands = np.array([[-w * n_jobs, -c] for c, w in _pts], dtype=np.float32)
        print(f"[commands] archive自己再現モード: {len(commands)}指令", flush=True)
    else:
        r1_grid = build_r1_grid(cost_max, 1.0, args.grid)
        r0_grid = np.linspace(0.0, -wait_max * n_jobs, args.grid)
        commands = np.array([[r0, r1] for r0 in r0_grid for r1 in r1_grid], dtype=np.float32)
    B = len(commands)
    hz = np.full(B, float(n_jobs), dtype=np.float32)
    print(f"[grid] {args.grid}x{args.grid} = {B} commands", flush=True)

    model = load_policy_model(args.checkpoint, n_jobs, device="cuda")
    # warmup(JIT)
    run_lockstep_greedy(jobs[:256], model, commands[:1], n_on, n_cl, n_window=100,
                        horizons=hz[:1], e_max=2048, k=args.k, tpb=1,
                        mode="greedy", return_obs=False)
    t0 = time.time()
    out = run_lockstep_greedy(jobs, model, commands, n_on, n_cl, n_window=100,
                              horizons=hz, e_max=args.emax, k=args.k, tpb=1,
                              mode="greedy", return_obs=False)
    dt = time.time() - t0
    ovf = int(np.asarray(out["ovf"]).sum())
    waits = np.asarray(out["waits"].cpu() if hasattr(out["waits"], "cpu") else out["waits"])
    costs = np.asarray(out["costs"].cpu() if hasattr(out["costs"], "cpu") else out["costs"])
    total_cost = costs.sum(axis=1).astype(np.float64)
    avg_wait = waits.sum(axis=1).astype(np.float64) / n_jobs
    pts = np.stack([total_cost, avg_wait], axis=1)
    print(f"[rollout] {B}本 {dt:.1f}s ovf={ovf}", flush=True)

    # 非支配集合(最小化)
    nd = np.ones(B, dtype=bool)
    for i in range(B):
        if nd[i]:
            dominated = (pts[:, 0] >= pts[i, 0]) & (pts[:, 1] >= pts[i, 1]) & \
                        ((pts[:, 0] > pts[i, 0]) | (pts[:, 1] > pts[i, 1]))
            nd[dominated] = False
    uniq = np.unique(pts[nd], axis=0)
    print(f"[PF] 到達点 {B} → 非支配 {int(nd.sum())} → ユニーク {len(uniq)}", flush=True)

    os.makedirs(args.output, exist_ok=True)
    np.savez(os.path.join(args.output, f"uniform_pf_{args.label}.npz"),
             commands=commands, points=pts, nd_mask=nd)

    import matplotlib
    matplotlib.use("Agg")
    # 日本語ラベルの文字化け対策(Noto CJKをmatplotlibへ登録)
    try:
        from matplotlib import font_manager as _fm
        _fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=14, c="#9ecae1", label=f"到達点 ({B}指令)")
    order = np.argsort(uniq[:, 0])
    ax.plot(uniq[order, 0], uniq[order, 1], "o-", c="#d62728", ms=5, lw=1.5,
            label=f"非支配 ({len(uniq)}点)")
    ax.set_xlabel("total cost")
    ax.set_ylabel("avg wait [s]")
    ax.set_title(f"uniform-command reachable PF — {args.label} "
                 f"(grid {args.grid}x{args.grid}, greedy/lockstep)")
    ax.legend()
    fig_path = os.path.join(args.output, f"uniform_pf_{args.label}.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"[fig] {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
