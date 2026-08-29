#!/usr/bin/env python3
"""実運用に即した「一発再現」評価で Eval PF 進化 GIF を作る。

実運用 = 最終モデル1個に「PFの各点を出して」と命令(desired_return)を1回ずつ入れ、
その1回で各点を再現する。だから評価も union やポートフォリオ回収ではなく、
command_mode=pf_archive（archive PF点を命令化して各点1回）で測る。

各 iter のチェックポイントを同じ命令セット(=固定 snapshot の archive PF)・同じ条件で
一発再現させ、「注文(緑)に赤がどれだけ一発で乗るか」の変化を GIF 化する。

注意: 評価の env フラグ(FILM 等)は学習と一致必須。この run は PCN_FILM=0
（FILM=1 だと命令無視で1点に潰れる）。呼び出し側で env を合わせること。

使い方:
    PCN_FILM=0 PYTHONPATH=. uv run python scripts/make_oneshot_pf_gif.py \
        --run-dir experiments/distributed_pcn/trace24_scratch_20260603_163503/20260603_163507 \
        --config experiments/distributed_pcn/job_trace_24_scratch_pass.yml \
        --out docs/eval_pf_oneshot.gif
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")

from scripts.analyze_pf_retention import collect_iters, nd_min, setup_jp_font


def norm_dist(ach: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    cr = max(tgt[:, 0].max() - tgt[:, 0].min(), 1.0)
    wr = max(tgt[:, 1].max() - tgt[:, 1].min(), 1.0)
    return np.sqrt(((ach[:, 0] - tgt[:, 0]) / cr) ** 2 + ((ach[:, 1] - tgt[:, 1]) / wr) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--out", default="docs/eval_pf_oneshot.gif")
    ap.add_argument("--cache", default="")
    ap.add_argument("--ms", type=int, default=280)
    ap.add_argument("--command-mode", default="pf_archive",
                    help="uniform=目的空間の均等格子(密), pf_archive=snapshot archive PF点")
    ap.add_argument("--grid", type=int, default=12, help="command-mode=uniform の格子数")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    snap = Path(args.snapshot) if args.snapshot else (run_dir / "learner_replay_snapshot.pkl.gz")
    snap = snap if snap.is_file() else None
    cache = Path(args.cache) if args.cache else (run_dir / "oneshot_pf_cache.npz")
    iters = collect_iters(run_dir)
    if not iters:
        raise SystemExit(f"no checkpoints under {run_dir}")

    per_iter: dict[int, np.ndarray] = {}
    targets = None
    if cache.is_file() and not args.force:
        z = np.load(cache, allow_pickle=True)
        targets = z["targets"] if "targets" in z.files else None
        for k in z.files:
            if k.startswith("iter_"):
                per_iter[int(k[5:])] = z[k]
        print(f"[oneshot] cache {cache} ({len(per_iter)} iters)")
    if len(per_iter) < len(iters) or targets is None:
        from scripts.eval_uniform_command_pf import run as eval_run
        scratch = Path(os.environ.get("ONESHOT_SCRATCH", "/tmp/oneshot_reval"))
        scratch.mkdir(parents=True, exist_ok=True)
        for it, ck in iters:
            if it in per_iter:
                continue
            stats = eval_run(ck, snap, scratch, f"os{it:03d}", command_mode=args.command_mode,
                             grid=args.grid, config_path=args.config)
            per_iter[it] = np.asarray(stats["achieved_points"], dtype=np.float64)
            if targets is None:
                targets = np.asarray(stats["command_targets"], dtype=np.float64)
        np.savez_compressed(cache, targets=targets, **{f"iter_{it}": p for it, p in per_iter.items()})
        print(f"[oneshot] cached → {cache}")

    its = sorted(per_iter)
    tgt = np.asarray(targets, dtype=np.float64)
    to = np.argsort(tgt[:, 0])
    # 固定軸
    allpts = np.vstack([per_iter[it] for it in its] + [tgt])
    xlim = (allpts[:, 0].min() * 0.98 - 1e4, allpts[:, 0].max() * 1.02)
    ylim = (max(0.0, allpts[:, 1].min() - 30), allpts[:, 1].max() * 1.05)
    # per-iter 指標
    mean_d = np.array([norm_dist(per_iter[it], tgt).mean() for it in its])
    cost_floor = np.array([per_iter[it][:, 0].min() for it in its])

    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    setup_jp_font()

    frames = []
    for i, it in enumerate(its):
        ach = per_iter[it]
        pf = ach[nd_min(ach)]
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
        # 左: 注文(緑) と 一発到達(青) と 再現PF(赤)
        axL.plot(tgt[to, 0], tgt[to, 1], "-", c="#16a34a", lw=1.3, alpha=0.5,
                 label=f"注文リスト archive PF ({len(tgt)})")
        axL.scatter(tgt[:, 0], tgt[:, 1], c="#16a34a", s=12, alpha=0.35)
        axL.scatter(ach[:, 0], ach[:, 1], c="#2563eb", s=20, alpha=0.4, label="一発到達点")
        po = np.argsort(pf[:, 0])
        axL.plot(pf[po, 0], pf[po, 1], "r-o", lw=1.5, ms=4, label=f"再現PF ({len(pf)})")
        axL.set_xlim(*xlim); axL.set_ylim(*ylim)
        axL.set_xlabel("Cost"); axL.set_ylabel("Average Waiting Time")
        axL.set_title(f"iter {it} — 注文(緑)を1発で再現できるか  平均ズレ={mean_d[i]:.2f}")
        axL.legend(fontsize=8, loc="upper right"); axL.grid(True, alpha=0.3)
        # 右: 一発再現ズレ と 到達できた最安cost(=左端の限界) の推移
        axR.plot(its, mean_d, "-", c="#dc2626", lw=1.2, label="一発再現ズレ(正規化)")
        axR.scatter(its[: i + 1], mean_d[: i + 1], c="#dc2626", s=12)
        axR.scatter([it], [mean_d[i]], c="#f59e0b", s=110, zorder=5)
        axR.set_xlabel("iteration"); axR.set_ylabel("一発再現ズレ（小さいほど良い）", color="#dc2626")
        axR.tick_params(axis="y", labelcolor="#dc2626")
        axR.set_xlim(its[0] - 3, its[-1] + 3)
        ax2 = axR.twinx()
        ax2.plot(its, cost_floor, "-", c="#2563eb", lw=1.0, alpha=0.7)
        ax2.scatter([it], [cost_floor[i]], c="#2563eb", s=70, zorder=5)
        ax2.set_ylabel("届いた最安cost（左端の限界・低いほど良い）", color="#2563eb")
        ax2.tick_params(axis="y", labelcolor="#2563eb")
        axR.set_title("学習が進むほどズレ↑・最安cost↑＝安い端へ届かなくなる")
        axR.grid(True, alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    w = max(f.width for f in frames); h = max(f.height for f in frames)
    def pad(im):
        if im.size == (w, h):
            return im
        c = Image.new("RGB", (w, h), (255, 255, 255))
        c.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
        return c
    frames = [pad(f) for f in frames]
    dur = [args.ms] * len(frames); dur[-1] = args.ms * 6
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=dur, loop=0,
                   optimize=True, disposal=2)
    print(f"[oneshot] {len(frames)} frames → {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
