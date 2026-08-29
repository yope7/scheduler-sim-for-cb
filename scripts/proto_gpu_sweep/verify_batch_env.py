"""GPU バッチ env（batch_env_jax）と本体 env 参照軌跡の全ステップ・全バッチ一致検証。

record_ref_traj.py が書いた npz を読み、同一ジョブ列（broadcast）×同一行動列 B 本を
step 単発（jit）の Python ループで回して毎ステップ突き合わせる:
  - 配置: start_time / nodes[:height] = 完全一致
  - 報酬: rewards / wait / cost = 完全一致（全て整数演算由来の f64。ズレたら atol 報告）
  - フラグ: scheduled / done = 完全一致
  - 最終: cost_total / avgwait = 完全一致
fast0（PCN_FAST_ENV=0 = prune 無し・legacy 候補ループ）参照があれば fast1 と
突き合わせ、prune + sweep/C 経路の結果不変も白黒つける。

usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/verify_batch_env.py \
      scripts/proto_gpu_sweep/data/ref_nj128_seed0_fast1.npz [...]
  env: KERNEL=v1
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import jax  # noqa: E402

from batch_env_jax import (  # noqa: E402
    make_initial_state,
    make_step,
    rollout_segmented,
)

KERNEL = os.environ.get("KERNEL", "v1")
SEG_LEN = int(os.environ.get("SEG_LEN", "64"))


def verify_against_ref(path: str) -> tuple[int, int]:
    ref = {k: v for k, v in np.load(path).items()}   # npz 遅延解凍を避け全展開
    jobs = ref["jobs"]                    # (NJ,8)
    actions = ref["actions"]              # (B,NJ)
    B, NJ = actions.shape
    n_on, n_cl = int(ref["n_on"]), int(ref["n_cl"])
    heights = jobs[:, 2].astype(np.int64)

    state = make_initial_state(np.broadcast_to(jobs, (B, NJ, 8)), n_on=n_on, n_cl=n_cl)
    h_max = int(jobs[:, 2].max())
    step = jax.jit(make_step(n_on, n_cl, h_max, KERNEL))

    n_bad = 0
    n_cmp = 0
    import jax.numpy as jnp

    acts_dev = jnp.asarray(actions.T, dtype=jnp.int32)   # (T,B)
    for t in range(NJ):
        state, out = step(state, acts_dev[t])
        g_start = np.asarray(out.start)
        g_nodes = np.asarray(out.nodes)
        g_wait = np.asarray(out.wait)
        g_cost = np.asarray(out.cost)
        g_rew = np.asarray(out.rewards)
        g_sched = np.asarray(out.scheduled)
        g_done = np.asarray(out.done)
        h = int(heights[t])               # 同一ジョブ列 broadcast なので t で共通
        ok_vec = (
            (g_start == ref["start"][:, t])
            & (g_nodes[:, :h] == ref["nodes"][:, t, :h]).all(axis=1)
            & (g_wait == ref["wait"][:, t])
            & (g_cost == ref["cost"][:, t])
            & (g_rew == ref["rew"][:, t]).all(axis=1)
            & (g_sched == ref["sched"][:, t])
            & (g_done == ref["done"][:, t])
        )
        n_cmp += B
        for b in np.flatnonzero(~ok_vec):
            n_bad += 1
            if n_bad <= 5:
                print(
                    f"  MISMATCH b={b} t={t}: "
                    f"ref(start={ref['start'][b,t]}, wait={ref['wait'][b,t]}, "
                    f"cost={ref['cost'][b,t]}, nodes={ref['nodes'][b,t,:h][:8]}) "
                    f"gpu(start={g_start[b]}, wait={g_wait[b]}, "
                    f"cost={g_cost[b]}, nodes={g_nodes[b,:h][:8]}) "
                    f"rew_diff={g_rew[b] - ref['rew'][b,t]}"
                )

    # 最終値: cost は整数、avgwait は「整数和 / n_jobs」の f64（順序不問で厳密）
    g_final_cost = np.asarray(state.cost_total, dtype=np.float64)
    g_avgwait = np.asarray(state.waits.sum(axis=1), dtype=np.float64) / NJ
    fc_ok = np.array_equal(g_final_cost, ref["final_cost"])
    aw_ok = np.array_equal(g_avgwait, ref["final_avgwait"])
    if not fc_ok or not aw_ok:
        n_bad += 1
        print(
            f"  FINAL MISMATCH cost_ok={fc_ok} avgwait_ok={aw_ok} "
            f"max|dc|={np.abs(g_final_cost - ref['final_cost']).max()} "
            f"max|dw|={np.abs(g_avgwait - ref['final_avgwait']).max()}"
        )
    tag = "OK " if n_bad == 0 else "FAIL"
    print(
        f"{tag} {os.path.basename(path)}: {n_cmp - n_bad}/{n_cmp} step-cells match "
        f"(B={B} T={NJ} kernel={KERNEL}, final cost/avgwait "
        f"{'exact' if fc_ok and aw_ok else 'MISMATCH'})"
    )
    return n_bad, n_cmp


def verify_segmented(path: str) -> tuple[int, int]:
    """rollout_segmented（E 段階拡大 + lax.scan）でも ref と全項目一致するか。"""
    ref = {k: v for k, v in np.load(path).items()}
    jobs = ref["jobs"]
    actions = ref["actions"]
    B, NJ = actions.shape
    n_on, n_cl = int(ref["n_on"]), int(ref["n_cl"])
    heights = jobs[:, 2].astype(np.int64)

    state = make_initial_state(np.broadcast_to(jobs, (B, NJ, 8)), n_on=n_on, n_cl=n_cl)
    fs, out = rollout_segmented(
        state, actions.T, n_on=n_on, n_cl=n_cl, kernel=KERNEL,
        seg_len=SEG_LEN, record=True,
    )
    g_nodes = np.asarray(out.nodes)                     # (T,B,h_max)
    hk = np.arange(g_nodes.shape[2])[None, None, :] < heights[:, None, None]  # (T,1,K)
    nodes_ok = (
        np.where(hk, g_nodes, -1)
        == np.where(hk, np.transpose(ref["nodes"], (1, 0, 2))[:, :, : g_nodes.shape[2]], -1)
    ).all(axis=2)
    ok = (
        (np.asarray(out.start) == ref["start"].T)
        & nodes_ok
        & (np.asarray(out.wait) == ref["wait"].T)
        & (np.asarray(out.cost) == ref["cost"].T)
        & (np.asarray(out.rewards) == np.transpose(ref["rew"], (1, 0, 2))).all(axis=2)
        & (np.asarray(out.scheduled) == ref["sched"].T)
        & (np.asarray(out.done) == ref["done"].T)
    )
    fc_ok = np.array_equal(
        np.asarray(fs.cost_total, dtype=np.float64), ref["final_cost"]
    )
    aw_ok = np.array_equal(
        np.asarray(fs.waits.sum(axis=1), dtype=np.float64) / NJ, ref["final_avgwait"]
    )
    n_bad = int((~ok).sum()) + int(not (fc_ok and aw_ok))
    tag = "OK " if n_bad == 0 else "FAIL"
    print(
        f"{tag} segmented(seg_len={SEG_LEN}) {os.path.basename(path)}: "
        f"{ok.sum()}/{ok.size} step-cells match, final "
        f"{'exact' if fc_ok and aw_ok else 'MISMATCH'}"
    )
    return n_bad, int(ok.size)


def _compare_block(out, fs_cost, fs_waits, ref, sl) -> tuple[int, int]:
    """rollout(record=True) の出力スライス sl を参照 ref と全項目比較。"""
    B_r, NJ = ref["actions"].shape
    heights = ref["jobs"][:, 2].astype(np.int64)
    g_nodes = np.asarray(out.nodes)[:, sl]              # (T,B_r,K)
    K = g_nodes.shape[2]
    hk = np.arange(K)[None, None, :] < heights[:, None, None]   # (T,1,K)
    ref_nodes = np.transpose(ref["nodes"], (1, 0, 2))   # (T,B_r,h_pad)
    if ref_nodes.shape[2] < K:
        pad = np.full((NJ, B_r, K - ref_nodes.shape[2]), -1, ref_nodes.dtype)
        ref_nodes = np.concatenate([ref_nodes, pad], axis=2)
    nodes_ok = (
        np.where(hk, g_nodes, -1) == np.where(hk, ref_nodes[:, :, :K], -1)
    ).all(axis=2)
    ok = (
        (np.asarray(out.start)[:, sl] == ref["start"].T)
        & nodes_ok
        & (np.asarray(out.wait)[:, sl] == ref["wait"].T)
        & (np.asarray(out.cost)[:, sl] == ref["cost"].T)
        & (np.asarray(out.rewards)[:, sl] == np.transpose(ref["rew"], (1, 0, 2))).all(axis=2)
        & (np.asarray(out.scheduled)[:, sl] == ref["sched"].T)
        & (np.asarray(out.done)[:, sl] == ref["done"].T)
    )
    fc_ok = np.array_equal(fs_cost[sl], ref["final_cost"])
    aw_ok = np.array_equal(fs_waits[sl], ref["final_avgwait"])
    return int((~ok).sum()) + int(not (fc_ok and aw_ok)), int(ok.size)


def verify_mixed(paths: list[str]) -> tuple[int, int]:
    """工場ユース検証: 異なるジョブ列（別 job_seed）を 1 バッチに混在させても
    各エピソードがそれぞれの参照と一致するか（B 次元の独立性）。"""
    refs = [{k: v for k, v in np.load(p).items()} for p in paths]
    NJ = refs[0]["actions"].shape[1]
    assert all(r["actions"].shape[1] == NJ for r in refs)
    n_on, n_cl = int(refs[0]["n_on"]), int(refs[0]["n_cl"])
    jobs_all = np.concatenate(
        [np.broadcast_to(r["jobs"], (r["actions"].shape[0], NJ, 8)) for r in refs]
    )
    actions_all = np.concatenate([r["actions"] for r in refs])   # (B,NJ)

    state = make_initial_state(jobs_all, n_on=n_on, n_cl=n_cl)
    fs, out = rollout_segmented(
        state, actions_all.T, n_on=n_on, n_cl=n_cl, kernel=KERNEL,
        seg_len=SEG_LEN, record=True,
    )
    fs_cost = np.asarray(fs.cost_total, dtype=np.float64)
    fs_waits = np.asarray(fs.waits.sum(axis=1), dtype=np.float64) / NJ
    n_bad = 0
    n_cmp = 0
    off = 0
    for p, r in zip(paths, refs):
        B_r = r["actions"].shape[0]
        bad, cmp_ = _compare_block(out, fs_cost, fs_waits, r, slice(off, off + B_r))
        n_bad += bad
        n_cmp += cmp_
        off += B_r
    tag = "OK " if n_bad == 0 else "FAIL"
    print(
        f"{tag} mixed-jobs batch ({' + '.join(os.path.basename(p) for p in paths)}): "
        f"{n_cmp - n_bad}/{n_cmp} step-cells match"
    )
    return n_bad, n_cmp


def compare_fast01(path1: str) -> None:
    """fast1 参照と fast0 参照（prune 無し legacy）を突き合わせ、prune の白黒をつける。"""
    path0 = path1.replace("_fast1.npz", "_fast0.npz")
    if not os.path.exists(path0):
        print(f"(prune check skipped: {os.path.basename(path0)} not found)")
        return
    a, b = np.load(path1), np.load(path0)
    keys = ["start", "nodes", "wait", "cost", "rew", "sched", "done",
            "final_cost", "final_avgwait"]
    bad = [k for k in keys if not np.array_equal(a[k], b[k])]
    if bad:
        print(f"PRUNE CHECK FAIL {os.path.basename(path1)} vs fast0: mismatch in {bad}")
    else:
        print(
            f"PRUNE CHECK OK {os.path.basename(path1)} == fast0 "
            f"(prune+sweep/C 経路は全ステップ結果不変)"
        )


def main():
    paths = sys.argv[1:]
    if not paths:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("ref_") and f.endswith("_fast1.npz")
        )
    total_bad = 0
    for p in paths:
        if p.endswith("_fast1.npz"):
            compare_fast01(p)
        bad, _ = verify_against_ref(p)
        total_bad += bad
        bad_seg, _ = verify_segmented(p)
        total_bad += bad_seg
    # 工場ユース: 別 job_seed の参照が同 NJ で複数あれば混在バッチでも検証
    by_nj: dict[int, list[str]] = {}
    for p in paths:
        nj = int(np.load(p)["actions"].shape[1])
        by_nj.setdefault(nj, []).append(p)
    for nj, ps in by_nj.items():
        if len(ps) >= 2:
            bad, _ = verify_mixed(ps)
            total_bad += bad
    print(f"\nTOTAL: {'ALL OK' if total_bad == 0 else f'FAIL ({total_bad} mismatches)'}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
