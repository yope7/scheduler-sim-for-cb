#!/usr/bin/env python3
"""LIVE計装 npz(uniform_cmd_pts_iter_*.npz)から PF進化 GIF + 最終PF図を作る(再Eval不要)。
真PF(NSGA)があれば緑で重ねる。固定軸。使い方:
  uv run python scripts/make_scaled_pf_gif.py --run-dir <exec> --nsga <npz|""> --label 512 --out-dir docs/figures
"""
from __future__ import annotations
import argparse, glob, io, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from src.agents.pcn_agent import get_non_dominated_inds_minimize as nd
from scripts.analyze_pf_retention import setup_jp_font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--nsga", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--ms", type=int, default=320)
    args = ap.parse_args()
    setup_jp_font()

    fs = sorted(glob.glob(args.run_dir + "/uniform_cmd_pts_iter_*.npz"))
    if not fs:
        raise SystemExit("no LIVE npz")
    its = [int(re.search(r"iter_(\d+)", f).group(1)) for f in fs]
    achs = [np.asarray(np.load(f)["achieved"], dtype=np.float64) for f in fs]
    ref = None
    if args.nsga:
        z = np.load(args.nsga)
        ref = np.asarray(z["pf"], dtype=np.float64)
        ref = ref[nd(ref)]
        ref = ref[np.argsort(ref[:, 0])]

    allp = np.vstack(achs + ([ref] if ref is not None else []))
    xlim = (allp[:, 0].min() - allp[:, 0].ptp() * 0.03, allp[:, 0].max() * 1.03)
    ylim = (max(0, allp[:, 1].min() - allp[:, 1].ptp() * 0.03), allp[:, 1].max() * 1.03)

    frames = []
    for it, a in zip(its, achs):
        pf = a[nd(a)]; o = np.argsort(pf[:, 0])
        fig, ax = plt.subplots(figsize=(8, 6))
        if ref is not None:
            ax.plot(ref[:, 0], ref[:, 1], "-", c="#16a34a", lw=2.2, label="真PF (NSGA)", zorder=2)
        ax.scatter(a[:, 0], a[:, 1], c="#93c5fd", s=18, alpha=0.45, label="到達点", zorder=1)
        ax.plot(pf[o, 0], pf[o, 1], "r-o", lw=1.7, ms=4, label=f"再現PF ({len(pf)})", zorder=3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("Cost"); ax.set_ylabel("Average Waiting Time")
        ax.set_title(f"{args.label}ジョブ 勝ちレシピ — iter {it}")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
        buf.seek(0); frames.append(Image.open(buf).convert("RGB"))
        if it == its[-1]:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            fig2, ax2 = plt.subplots(figsize=(8.5, 6))
            if ref is not None:
                ax2.plot(ref[:, 0], ref[:, 1], "-", c="#16a34a", lw=2.4, label="真PF (NSGA)")
            ax2.scatter(a[:, 0], a[:, 1], c="#93c5fd", s=20, alpha=0.5, label="到達点")
            ax2.plot(pf[o, 0], pf[o, 1], "r-o", lw=1.8, ms=5, label=f"再現PF ({len(pf)})")
            ax2.set_xlabel("Cost"); ax2.set_ylabel("Average Waiting Time")
            ax2.set_title(f"{args.label}ジョブ 勝ちレシピ 最終PF vs 真PF")
            ax2.legend(fontsize=9); ax2.grid(alpha=0.3); fig2.tight_layout()
            fig2.savefig(f"{args.out_dir}/pf_{args.label}_final.png", dpi=130, bbox_inches="tight")
            plt.close(fig2)

    w = max(f.width for f in frames); h = max(f.height for f in frames)
    def pad(im):
        if im.size == (w, h):
            return im
        c = Image.new("RGB", (w, h), (255, 255, 255)); c.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); return c
    frames = [pad(f) for f in frames]
    dur = [args.ms] * len(frames); dur[-1] = args.ms * 6
    out = f"{args.out_dir}/pf_evolution_{args.label}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True, disposal=2)
    print(f"[scaled] {args.label}: {len(frames)} frames → {out} + pf_{args.label}_final.png")


if __name__ == "__main__":
    main()
