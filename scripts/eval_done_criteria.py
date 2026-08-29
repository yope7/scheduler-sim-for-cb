#!/usr/bin/env python3
"""ゴール判定器: モデルの一発再現PFが「端点を完全再現・中央は真PF近く」を満たすか。

真PF = 厳密端点(全onprem=cost0/最大wait, 全cloud=最大cost/最小wait) + NSGA-II参照(中央)。
入力 = run の oneshot_pf_cache.npz (final iter の到達点; uniform格子一発再現)。

指標(すべて真PFの cost/wait レンジで正規化):
  - 端点再現: 各真端点への最近傍到達点の正規化距離 (小=良, <0.03目安で「完全」)
  - 中央gap : 各NSGA参照点への最近傍到達点距離の平均 (小=真PF近く)
  - HV比    : 到達HV / NSGA-HV
使い方:
  PCN_FILM=1 ... (学習一致フラグ) PYTHONPATH=. uv run python scripts/eval_done_criteria.py \
    --cache <run>/.../oneshot_pf_cache.npz --config <cfg> --nsga results/eval_pf/nsga2_trace256_s0.npz
"""
from __future__ import annotations
import argparse, os
import numpy as np

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
from src.agents.pcn_agent import get_non_dominated_inds_minimize as nd
from scripts.pcn_replay_snapshot import create_eval_env, load_config


def true_endpoints(config_path, n_jobs):
    env = create_eval_env(load_config(config_path), job_seed=0, n_jobs=n_jobs)
    pts = {}
    for a in range(env.action_space.n):
        env.reset(); done = False; cost = 0.0; wt = 0.0
        while not done:
            _, r, _, wt_step, done = env.step(a)
            cost += -float(np.asarray(r)[1]); wt += wt_step
        pts[a] = (cost, wt / n_jobs)
    cheap = min(pts.values(), key=lambda p: p[0])   # cost最小(=0)端 (高wait)
    exp = max(pts.values(), key=lambda p: p[0])     # cost最大端 (低wait)
    return np.array(cheap), np.array(exp)


def hv2d(pf, ref):
    pf = pf[(pf[:, 0] <= ref[0]) & (pf[:, 1] <= ref[1])]
    if pf.size == 0:
        return 0.0
    pf = pf[np.argsort(pf[:, 0])]; h = 0.0; pw = ref[1]
    for c, w in pf:
        h += (ref[0] - c) * (pw - w); pw = w
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    ap.add_argument("--run-dir", default="", help="指定時: 最終ckptのみ直接評価(高速)")
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nsga", default="results/eval_pf/nsga2_trace256_s0.npz")
    ap.add_argument("--n-jobs", type=int, default=256)
    ap.add_argument("--iter", default="last")
    args = ap.parse_args()

    if args.run_dir:
        import glob as _g
        from scripts.eval_uniform_command_pf import run as eval_run
        ck = sorted(_g.glob(args.run_dir + "/iteration_*/model_iter_*.pth"))[-1]
        snap = args.run_dir + "/learner_replay_snapshot.pkl.gz"
        import tempfile
        stats = eval_run(__import__("pathlib").Path(ck), __import__("pathlib").Path(snap),
                         __import__("pathlib").Path(tempfile.mkdtemp()), "done", command_mode="uniform",
                         grid=args.grid, config_path=args.config)
        ach = np.asarray(stats["achieved_points"], dtype=np.float64)
        it = int(ck.split("iter_")[-1].split(".")[0])
    else:
        z = np.load(args.cache)
        its = sorted(int(k[5:]) for k in z.files if k.startswith("iter_"))
        it = its[-1] if args.iter == "last" else int(args.iter)
        ach = z[f"iter_{it}"]
    ach_pf = ach[nd(ach)]

    ep_cheap, ep_cloud = true_endpoints(args.config, args.n_jobs)  # 厳密端(全onprem/全cloud)
    zc = np.load(args.nsga)
    ref = zc["pf"] if "pf" in zc.files else (zc["pareto_front"] if "pareto_front" in zc.files else zc[zc.files[0]])
    ref = ref[nd(ref)]
    # 真PF = NSGA + 厳密な全onprem端(cost0)。全cloudは支配されるので端点に使わない。
    truepf = np.vstack([ref, ep_cheap[None]])
    truepf = truepf[nd(truepf)]
    # 真PFの両端 = min-cost点(安・遅) と min-wait点(高速・NSGAの効率端)
    cheap = truepf[truepf[:, 0].argmin()]
    fast = truepf[truepf[:, 1].argmin()]

    # 正規化レンジ(真PFの張る範囲)
    cr = max(truepf[:, 0].max() - truepf[:, 0].min(), 1.0)
    wr = max(truepf[:, 1].max() - truepf[:, 1].min(), 1.0)

    def ndist_to_ach(p):
        d = np.sqrt(((ach[:, 0] - p[0]) / cr) ** 2 + ((ach[:, 1] - p[1]) / wr) ** 2)
        return float(d.min())

    d_cheap = ndist_to_ach(cheap)
    d_exp = ndist_to_ach(fast)
    # 中央gap: NSGA各点への最近傍到達点距離
    gaps = np.array([ndist_to_ach(p) for p in ref])
    ref_hv = hv2d(ref, np.array([truepf[:, 0].max() * 1.02, truepf[:, 1].max() * 1.02]))
    ach_hv = hv2d(ach_pf, np.array([truepf[:, 0].max() * 1.02, truepf[:, 1].max() * 1.02]))

    print(f"[done-eval] iter{it}  n_jobs={args.n_jobs}")
    print(f"  真端点  cheap=(cost{cheap[0]:.2e}, wait{cheap[1]:.0f})  fast=(cost{fast[0]:.2e}, wait{fast[1]:.0f})")
    print(f"  到達PF点数={len(ach_pf)}  cost[{ach[:,0].min():.2e},{ach[:,0].max():.2e}] wait[{ach[:,1].min():.0f},{ach[:,1].max():.0f}]")
    print(f"  ★端点再現(正規化距離): cheap={d_cheap:.3f}  exp={d_exp:.3f}   (<0.03=完全再現目安)")
    print(f"  ★中央gap(NSGA{len(ref)}点への最近傍平均)={gaps.mean():.3f} 中央値={np.median(gaps):.3f}")
    print(f"  ★HV比(到達/NSGA)={ach_hv/max(ref_hv,1e-9):.3f}")
    done_ep = d_cheap < 0.03 and d_exp < 0.03
    done_mid = gaps.mean() < 0.05
    print(f"  判定: 端点完全={'OK' if done_ep else 'NG'}  中央近い={'OK' if done_mid else 'NG'}  → {'DONE' if (done_ep and done_mid) else '未達'}")


if __name__ == "__main__":
    main()
