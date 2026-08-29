#!/usr/bin/env python3
"""「学習途中で出たEval点が、後半で失われる」損失を数値化する（ずるくない範囲）。

やること: 実験ディレクトリの per-iter チェックポイント model_iter_XXX.pth を
すべて同一 env / 同一 command 格子で再Evalし、
  - union PF  = 全 iter の到達点の和集合の非支配フロント（＝一度でも到達できた能力）
  - final PF  = 最終 iter の到達点
を比べ、「一度出せたのに最後に出せない点（＝忘れた点）」を帯ごとに数える。

すべて学習インスタンス(job_seed=0)上の測定のみ。未知データを覗いていない＝ずるくない。

使い方:
    PYTHONPATH=. uv run python scripts/analyze_pf_retention.py \
        --run-dir experiments/distributed_pcn/trace24_scratch_20260603_163503/20260603_163507 \
        --config experiments/distributed_pcn/job_trace_24_scratch_pass.yml \
        --out docs/figures/pf_retention_trace24.png
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")

ITER_RE = re.compile(r"iteration_(\d+)$")


def setup_jp_font() -> None:
    """日本語ラベルが豆腐(□)化しないよう Noto Sans CJK JP を登録。"""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    import matplotlib.pyplot as plt
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              *sorted(glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc"))):
        try:
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def nd_min(pts: np.ndarray) -> np.ndarray:
    """2D 最小化の非支配マスク。"""
    from src.agents.pcn_agent import get_non_dominated_inds_minimize
    if pts.size == 0:
        return np.array([], dtype=int)
    return get_non_dominated_inds_minimize(pts)


def hv2d_min(pf: np.ndarray, ref: np.ndarray) -> float:
    """2D 最小化の hypervolume（ref=最悪角）。"""
    if pf.size == 0:
        return 0.0
    pf = pf[(pf[:, 0] <= ref[0]) & (pf[:, 1] <= ref[1])]
    if pf.size == 0:
        return 0.0
    order = np.argsort(pf[:, 0])  # cost 昇順 → wait 降順
    pf = pf[order]
    hv = 0.0
    prev_wait = ref[1]
    for c, w in pf:
        hv += (ref[0] - c) * (prev_wait - w)
        prev_wait = w
    return float(hv)


def weakly_dominated_by_any(u: np.ndarray, pool: np.ndarray, rtol: float) -> bool:
    """pool のどれかが u を（相対 rtol の緩みで）弱支配するか。"""
    if pool.size == 0:
        return False
    tol_c = abs(u[0]) * rtol + 1e-9
    tol_w = abs(u[1]) * rtol + 1e-9
    return bool(np.any((pool[:, 0] <= u[0] + tol_c) & (pool[:, 1] <= u[1] + tol_w)))


def collect_iters(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for d in run_dir.glob("iteration_*"):
        m = ITER_RE.search(d.name)
        if not m:
            continue
        ck = d / f"model_iter_{int(m.group(1)):03d}.pth"
        if ck.is_file():
            out.append((int(m.group(1)), ck))
    out.sort(key=lambda t: t[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", default="", help="未指定なら run-dir の learner_replay_snapshot.pkl.gz")
    ap.add_argument("--out", default="docs/figures/pf_retention.png")
    ap.add_argument("--cache", default="", help="per-iter 到達点キャッシュ npz（既定 <run-dir>/pf_retention_cache.npz）")
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--low-tail-frac", type=float, default=0.18)
    ap.add_argument("--low-tail-extra", type=int, default=16)
    ap.add_argument("--rtol", type=float, default=0.02, help="忘却判定の相対緩み（final がこの緩みで届けば保持）")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="キャッシュを無視して再Eval")
    ap.add_argument("--gif", default="", help="固定軸の進化GIFも出す（例 docs/eval_pf_evolution_fixed.gif）")
    ap.add_argument("--gif-ms", type=int, default=280)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    snap = Path(args.snapshot) if args.snapshot else (run_dir / "learner_replay_snapshot.pkl.gz")
    snap = snap if snap.is_file() else None
    cache = Path(args.cache) if args.cache else (run_dir / "pf_retention_cache.npz")
    iters = collect_iters(run_dir)
    if not iters:
        raise SystemExit(f"no iteration_*/model_iter_*.pth under {run_dir}")
    print(f"[retention] {len(iters)} checkpoints iter {iters[0][0]}..{iters[-1][0]}")

    # --- 各チェックポイントを同条件で再Eval（キャッシュがあれば読む） ---
    per_iter: dict[int, np.ndarray] = {}
    if cache.is_file() and not args.force:
        z = np.load(cache, allow_pickle=True)
        for k in z.files:
            if k.startswith("iter_"):
                per_iter[int(k[5:])] = z[k]
        print(f"[retention] loaded cache {cache} ({len(per_iter)} iters)")
    if len(per_iter) < len(iters):
        from scripts.eval_uniform_command_pf import run as eval_run
        scratch = Path(os.environ.get("PF_RET_SCRATCH", "/tmp/pf_retention_reval"))
        scratch.mkdir(parents=True, exist_ok=True)
        for it, ck in iters:
            if it in per_iter:
                continue
            stats = eval_run(
                ck, snap, scratch, f"it{it:03d}",
                grid=args.grid, low_tail_frac=args.low_tail_frac, low_tail_extra=args.low_tail_extra,
                config_path=args.config,
            )
            per_iter[it] = np.asarray(stats["achieved_points"], dtype=np.float64)
        np.savez_compressed(cache, **{f"iter_{it}": pts for it, pts in per_iter.items()})
        print(f"[retention] cached → {cache}")

    its = sorted(per_iter)
    final_it = its[-1]
    final_pts = per_iter[final_it]
    union_pts = np.vstack([per_iter[it] for it in its])
    union_pf = union_pts[nd_min(union_pts)]
    final_pf = final_pts[nd_min(final_pts)]

    # 忘却判定: union PF 各点を final の到達点が rtol の緩みで届くか
    lost_mask = np.array([not weakly_dominated_by_any(u, final_pts, args.rtol) for u in union_pf])
    lost = union_pf[lost_mask]
    kept = union_pf[~lost_mask]

    # HV（ref=union の最悪角に少し余裕）
    ref = np.array([union_pts[:, 0].max() * 1.02, union_pts[:, 1].max() * 1.02])
    hv_union = hv2d_min(union_pf, ref)
    hv_final = hv2d_min(final_pf, ref)
    hv_per_iter = np.array([hv2d_min(per_iter[it][nd_min(per_iter[it])], ref) for it in its])
    peak_i = int(np.argmax(hv_per_iter))
    peak_it = its[peak_i]

    # 帯ごとの忘却数（cost 軸）
    cmin, cmax = union_pf[:, 0].min(), union_pf[:, 0].max()
    edges = np.linspace(cmin, cmax, args.bands + 1)
    print("\n=== 忘却の数値化（union PF = 一度でも出せた能力） ===")
    print(f"union PF 点数 : {len(union_pf)}   final PF 点数 : {len(final_pf)}")
    print(f"忘れた点(lost): {int(lost_mask.sum())} / {len(union_pf)}  "
          f"({100*lost_mask.mean():.0f}%)  [final が {args.rtol*100:.0f}% 緩めても届かない union PF 点]")
    print(f"HV  union={hv_union:.3e}  final={hv_final:.3e}  "
          f"final/union={100*hv_final/max(hv_union,1e-12):.1f}%   "
          f"peak@iter{peak_it}={hv_per_iter[peak_i]:.3e} (final/peak={100*hv_final/max(hv_per_iter[peak_i],1e-12):.1f}%)")
    print("\ncost帯ごとの忘却（左=安い端＝“左上の激安プラン”）:")
    for b in range(args.bands):
        lo, hi = edges[b], edges[b + 1]
        in_b = (union_pf[:, 0] >= lo) & (union_pf[:, 0] <= hi if b == args.bands - 1 else union_pf[:, 0] < hi)
        n_b = int(in_b.sum())
        n_lost_b = int((in_b & lost_mask).sum())
        bar = "█" * n_lost_b + "·" * (n_b - n_lost_b)
        print(f"  cost[{lo:9.0f},{hi:9.0f}]  union={n_b:2d}  lost={n_lost_b:2d}  {bar}")

    # --- 図 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_jp_font()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    # 左: union vs final
    axL.scatter(final_pts[:, 0], final_pts[:, 1], c="#9ca3af", s=18, alpha=0.5,
                label=f"final 到達点 (iter{final_it})", zorder=2)
    uo = np.argsort(union_pf[:, 0])
    axL.plot(union_pf[uo, 0], union_pf[uo, 1], "-", c="#16a34a", lw=1.8,
             label=f"union PF 一度でも到達 ({len(union_pf)})", zorder=3)
    fo = np.argsort(final_pf[:, 0])
    axL.plot(final_pf[fo, 0], final_pf[fo, 1], "r-o", lw=1.4, ms=4,
             label=f"final PF ({len(final_pf)})", zorder=4)
    if lost.size:
        axL.scatter(lost[:, 0], lost[:, 1], marker="x", c="#dc2626", s=90, lw=2.2,
                    label=f"忘れた点 lost ({len(lost)})", zorder=5)
    axL.set_xlabel("Cost"); axL.set_ylabel("Average Waiting Time")
    axL.set_title(f"一度出せた能力(緑) vs 最後に出せる点(赤)\n忘却 {int(lost_mask.sum())}/{len(union_pf)} 点")
    axL.legend(fontsize=8); axL.grid(True, alpha=0.3)

    # 右: HV の推移（登って落ちる）
    axR.plot(its, hv_per_iter, "-o", c="#2563eb", ms=4)
    axR.axhline(hv_union, ls="--", c="#16a34a", lw=1.2, label=f"union HV (歴代包絡)")
    axR.scatter([peak_it], [hv_per_iter[peak_i]], c="#f59e0b", s=110, zorder=5,
                label=f"peak iter{peak_it}")
    axR.scatter([final_it], [hv_final], c="#dc2626", s=90, zorder=5,
                label=f"final iter{final_it} ({100*hv_final/max(hv_per_iter[peak_i],1e-12):.0f}% of peak)")
    axR.set_xlabel("iteration"); axR.set_ylabel("Eval PF hypervolume")
    axR.set_title("Eval PF の質(HV)は途中で頂点→その後に低下")
    axR.legend(fontsize=8); axR.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[retention] figure → {out}")

    if args.gif:
        render_evolution_gif(
            per_iter, its, union_pf, hv_per_iter, ref, peak_it, final_it,
            Path(args.gif), ms=args.gif_ms,
        )


def render_evolution_gif(per_iter, its, union_pf, hv_per_iter, ref, peak_it, final_it,
                         out: Path, ms: int = 280) -> None:
    """固定軸で Eval PF の iter 進化を GIF 化（歴代包絡=緑を背景に、右に HV 進捗）。"""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    setup_jp_font()

    allpts = np.vstack([per_iter[it] for it in its])
    xlim = (allpts[:, 0].min() * 0.98, allpts[:, 0].max() * 1.02)
    ylim = (max(0.0, allpts[:, 1].min() - 20), allpts[:, 1].max() * 1.05)
    uo = np.argsort(union_pf[:, 0])
    hv_peak = hv_per_iter.max()
    frames = []
    for i, it in enumerate(its):
        pts = per_iter[it]
        pf = pts[nd_min(pts)]
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
        # 左: 固定軸 + 歴代包絡(緑) + 現iter到達点(青) + 現iter Eval PF(赤)
        axL.plot(union_pf[uo, 0], union_pf[uo, 1], "-", c="#16a34a", lw=1.4, alpha=0.55,
                 label="歴代ベスト包絡 union PF")
        axL.scatter(pts[:, 0], pts[:, 1], c="#2563eb", s=20, alpha=0.35, label="到達点")
        po = np.argsort(pf[:, 0])
        axL.plot(pf[po, 0], pf[po, 1], "r-o", lw=1.6, ms=4, label=f"Eval PF ({len(pf)})")
        axL.set_xlim(*xlim); axL.set_ylim(*ylim)
        axL.set_xlabel("Cost"); axL.set_ylabel("Average Waiting Time")
        axL.set_title(f"iter {it}  —  赤(今)が緑(歴代best)にどれだけ届くか")
        axL.legend(fontsize=8, loc="upper right"); axL.grid(True, alpha=0.3)
        # 右: HV 進捗
        axR.plot(its, hv_per_iter, "-", c="#93c5fd", lw=1.2)
        axR.scatter(its[: i + 1], hv_per_iter[: i + 1], c="#2563eb", s=16)
        axR.scatter([it], [hv_per_iter[i]], c="#f59e0b", s=120, zorder=5)
        axR.axhline(hv_peak, ls="--", c="#16a34a", lw=1.0, alpha=0.6)
        axR.set_xlim(its[0] - 3, its[-1] + 3)
        axR.set_xlabel("iteration"); axR.set_ylabel("Eval PF hypervolume")
        axR.set_title(f"質(HV) = {100*hv_per_iter[i]/max(hv_peak,1e-12):.0f}% of peak")
        axR.grid(True, alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    w = max(f.width for f in frames); h = max(f.height for f in frames)
    frames = [f if f.size == (w, h) else _pad(f, w, h) for f in frames]
    durations = [ms] * len(frames)
    durations[-1] = ms * 6
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    print(f"[gif] {len(frames)} frames (fixed axes) → {out}  ({out.stat().st_size/1e6:.1f} MB)")


def _pad(im, w, h):
    from PIL import Image
    c = Image.new("RGB", (w, h), (255, 255, 255))
    c.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return c


if __name__ == "__main__":
    main()
