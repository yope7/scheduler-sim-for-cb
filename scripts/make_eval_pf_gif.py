#!/usr/bin/env python3
"""学習中の Eval PF (uniform_cmd_pf_iter_XXX.png) をそのまま繋いで GIF 化する。

pareto_animation_pcn.py のように配列を手打ちしない。実験ディレクトリに
学習中保存された per-iter 図（DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1 で出る
`uniform_cmd_pf_iter_XXX.png`）を iter 順に集めて 1 本の GIF にする。

使い方:
    uv run python scripts/make_eval_pf_gif.py \
        --dir experiments/distributed_pcn/trace24_scratch_20260603_163503/20260603_163507 \
        --out docs/eval_pf_evolution.gif --ms 300 --hold 5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

ITER_RE = re.compile(r"uniform_cmd_pf_iter_(\d+)\.png$")


def collect_frames(exec_dir: Path) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for p in exec_dir.glob("uniform_cmd_pf_iter_*.png"):
        m = ITER_RE.search(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda t: t[0])
    return frames


def load_padded(paths: list[Path]) -> list[Image.Image]:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    out = []
    for im in imgs:
        if im.size == (w, h):
            out.append(im)
        else:  # bbox_inches="tight" でサイズが揺れる分を白背景でパディング
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
            out.append(canvas)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="uniform_cmd_pf_iter_*.png を含む実験実行ディレクトリ")
    ap.add_argument("--out", default=None, help="出力 GIF パス（既定: <dir>/eval_pf_evolution.gif）")
    ap.add_argument("--ms", type=int, default=300, help="1フレームの表示ミリ秒")
    ap.add_argument("--hold", type=int, default=6, help="最終フレームを何倍長く止めるか")
    args = ap.parse_args()

    exec_dir = Path(args.dir)
    frames = collect_frames(exec_dir)
    if not frames:
        raise SystemExit(f"no uniform_cmd_pf_iter_*.png under {exec_dir}")

    iters = [it for it, _ in frames]
    print(f"[gif] {len(frames)} frames  iter {iters[0]}..{iters[-1]}  from {exec_dir}")

    imgs = load_padded([p for _, p in frames])
    durations = [args.ms] * len(imgs)
    durations[-1] = args.ms * max(1, args.hold)  # 最後で一拍止める

    out = Path(args.out) if args.out else exec_dir / "eval_pf_evolution.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        out,
        save_all=True,
        append_images=imgs[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"[gif] saved {out}  ({size_mb:.1f} MB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
