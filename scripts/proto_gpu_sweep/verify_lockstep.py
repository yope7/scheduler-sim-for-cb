"""lockstep_kernel の検証器(設計書 B-1/B-2)。

(a) B-1検証: 同一 jobs/actions で raw_rollout_kernel.run_raw_rollout と
    (start_time, waiting_time, cost) が全件一致(ビット)することを確認する。
    128ジョブ・B=4(verify_raw_rollout.py と同じ生成条件)と、
    実トレース weekB 先頭5万ジョブ・B=2(verify_raw_rollout.py --big と同じ条件)で行う。

(b) B-2検証: 同一軌道(行動列既知)で replay_obs_builder の obs と全ステップ一致
    (np.array_equal)することを確認する。128ジョブ・B=4、および 5万ジョブ・B=1(1本)。

(c) 性能: 5万ジョブ・B=64 のロックステップ実行時間(発行税込み)を実測し、
    既存 run_raw_rollout(全周カーネル内)との比を報告する。
    step_kernel のみ(collect_obs=False)と、step+obs 両方(collect_obs=True)を
    分けて計測する(観測構築は末尾に近い step ほど過去イベント数 j に比例して
    重くなるため=SchedulingEventBuffer が一切 prune されない設計。design doc の
    見積りとの差分として報告する)。

usage: CUDA_HOME=$PWD/tools/nvcc122/nvidia/cuda_nvcc CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
    scripts/proto_gpu_sweep/verify_lockstep.py [--skip-big] [--skip-perf]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 観測環境変数: obs_kernel が実装している固定構成と揃える(build_env が読む)。
os.environ.setdefault("SCHEDULER_OBS_URGENCY", "1")
os.environ.setdefault("SCHEDULER_OBS_EFFICIENCY", "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")

from lockstep_kernel import run_lockstep_rollout  # noqa: E402
from raw_rollout_kernel import run_raw_rollout  # noqa: E402
from replay_obs_builder import build_env, build_replay_dataset  # noqa: E402
from verify_raw_rollout import (  # noqa: E402
    B, BIG_CFG, BIG_N_JOBS, BIG_P_LIST, N_CLOUD, N_JOBS, N_ON, P_LIST, SEED,
    gen_actions, gen_jobs,
)

N_WINDOW = 16  # replay_obs_builder.build_env の既定値と一致させる


# ---------------------------------------------------------------------------
# (a) B-1: step_kernel vs raw_rollout_kernel
# ---------------------------------------------------------------------------
def check_b1_small() -> bool:
    jobs = gen_jobs(N_JOBS, SEED)
    actions = gen_actions(N_JOBS, P_LIST, SEED + 1)

    print(f"[b1-small] N_JOBS={N_JOBS} B={B} n_on={N_ON} n_cloud={N_CLOUD} p_list={P_LIST}")
    ref = run_raw_rollout(jobs, actions, N_ON, N_CLOUD)
    assert np.all(ref["ovf"] == 0), f"ref ovf set: {ref['ovf'].tolist()}"
    ok = True
    # collect_obs=False(素のB-1経路)と True(obs probe再利用経路)の両方を raw と突き合わせる。
    for co in (False, True):
        got = run_lockstep_rollout(
            jobs, actions, N_ON, N_CLOUD, e_max=8192, k=16, n_window=N_WINDOW, tpb=1,
            collect_obs=co,
        )
        assert np.all(got["ovf"] == 0), f"lockstep ovf set (obs={co}): {got['ovf'].tolist()}"
        mism_start = ref["start_times"] != got["start_times"]
        mism_wait = ref["waits"] != got["waits"]
        mism_cost = ref["costs"] != got["costs"]
        n_mismatch = int(mism_start.sum() + mism_wait.sum() + mism_cost.sum())
        if n_mismatch:
            print(f"[b1-small] FAIL (collect_obs={co}): {n_mismatch} field mismatches")
            for b in range(B):
                idx = np.flatnonzero(mism_start[b] | mism_wait[b] | mism_cost[b])
                for j in idx[:10]:
                    print(
                        f"  row={b} job={j}: ref=(start={ref['start_times'][b,j]},"
                        f"wait={ref['waits'][b,j]},cost={ref['costs'][b,j]}) "
                        f"lockstep=(start={got['start_times'][b,j]},wait={got['waits'][b,j]},"
                        f"cost={got['costs'][b,j]})"
                    )
            ok = False
            continue
        total = B * N_JOBS * 3
        print(f"[b1-small] PASS (collect_obs={co}): {total}/{total} fields match "
              f"(B={B} x N_JOBS={N_JOBS} x 3)")
    return ok


def check_b1_big() -> bool:
    from scripts.pcn_replay_snapshot import create_eval_env, load_config

    cfg = load_config(BIG_CFG)
    env = create_eval_env(cfg, job_seed=0, n_jobs=BIG_N_JOBS)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    n_on = int(env.n_on_premise_node)
    n_cl = int(env.n_cloud_node)
    n_jobs = jobs.shape[0]
    p_list = BIG_P_LIST[:2]  # B=2
    print(f"[b1-big] trace={BIG_CFG} n_jobs={n_jobs} n_on={n_on} n_cl={n_cl} p_list={p_list}")

    def gen_actions_local(n_jobs, p_list, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for p in p_list:
            rows.append((rng.random(n_jobs) < p).astype(np.int8))
        return np.stack(rows, axis=0)

    actions = gen_actions_local(n_jobs, p_list, SEED + 1)

    e_max, kk = 16384, 128
    ref = None
    for _ in range(4):
        t0 = time.time()
        ref = run_raw_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=kk)
        print(f"[b1-big ref] E_MAX={e_max} K={kk}: {time.time()-t0:.1f}s "
              f"ovf={ref['ovf'].tolist()}", flush=True)
        if not ref["ovf"].any():
            break
        if (ref["ovf"] == 1).any():
            e_max = 65536
        if (ref["ovf"] == 2).any():
            kk *= 4
    if ref["ovf"].any():
        print(f"[b1-big] FAIL: ref ovf still set (E_MAX={e_max}, K={kk})")
        return False

    t0 = time.time()
    got = run_lockstep_rollout(
        jobs, actions, n_on, n_cl, e_max=e_max, k=kk, n_window=N_WINDOW, tpb=1,
        collect_obs=False,
    )
    print(f"[b1-big lockstep] {time.time()-t0:.1f}s ovf={got['ovf'].tolist()}", flush=True)
    if got["ovf"].any():
        print(f"[b1-big] FAIL: lockstep ovf set {got['ovf'].tolist()}")
        return False

    mism = (
        (ref["start_times"] != got["start_times"])
        | (ref["waits"] != got["waits"])
        | (ref["costs"] != got["costs"])
    )
    n_mismatch = int(mism.sum())
    if n_mismatch:
        print(f"[b1-big] FAIL: {n_mismatch} field mismatches")
        for b in range(len(p_list)):
            idx = np.flatnonzero(mism[b])
            for j in idx[:10]:
                print(
                    f"  row={b} job={j}: ref=(start={ref['start_times'][b,j]},"
                    f"wait={ref['waits'][b,j]},cost={ref['costs'][b,j]}) "
                    f"lockstep=(start={got['start_times'][b,j]},wait={got['waits'][b,j]},"
                    f"cost={got['costs'][b,j]})"
                )
        return False
    total = len(p_list) * n_jobs * 3
    print(f"[b1-big] PASS: {total}/{total} fields match (B={len(p_list)} x N_JOBS={n_jobs} x 3)")
    return True, (jobs, actions, n_on, n_cl, e_max, kk, ref)


# ---------------------------------------------------------------------------
# (b) B-2: obs_kernel vs replay_obs_builder
# ---------------------------------------------------------------------------
def check_b2_small() -> bool:
    jobs = gen_jobs(N_JOBS, SEED)
    actions = gen_actions(N_JOBS, P_LIST, SEED + 1)

    print(f"[b2-small] N_JOBS={N_JOBS} B={B} n_on={N_ON} n_cloud={N_CLOUD} p_list={P_LIST}")
    print(
        "[b2-small] obs flags: "
        f"SCHEDULER_OBS_URGENCY={os.environ.get('SCHEDULER_OBS_URGENCY')} "
        f"SCHEDULER_OBS_EFFICIENCY={os.environ.get('SCHEDULER_OBS_EFFICIENCY')} "
        f"DISTRIBUTED_PCN_USE_EVENT_OBS={os.environ.get('DISTRIBUTED_PCN_USE_EVENT_OBS')} "
        f"SCHEDULER_LEARNER_BITMAP={os.environ.get('SCHEDULER_LEARNER_BITMAP')} n_window={N_WINDOW}"
    )

    t0 = time.time()
    got = run_lockstep_rollout(
        jobs, actions, N_ON, N_CLOUD, e_max=8192, k=16, n_window=N_WINDOW, tpb=1,
        collect_obs=True,
    )
    print(f"[b2-small] lockstep(step+obs): {time.time()-t0:.3f}s ovf={got['ovf'].tolist()}")
    assert np.all(got["ovf"] == 0), f"lockstep ovf set: {got['ovf'].tolist()}"

    t0 = time.time()
    replay = build_replay_dataset(
        jobs, actions, got["start_times"], N_ON, N_CLOUD, nproc=min(B, os.cpu_count() or 1),
        env_kwargs=dict(n_window=N_WINDOW),
    )
    print(f"[b2-small] replay_obs_builder: {time.time()-t0:.3f}s")

    fail = False
    n_obs_checked = 0
    for b in range(B):
        ep = replay[b]
        ref_obs = ep["obs"]
        got_obs = got["obs"][b]
        ok = np.array_equal(ref_obs, got_obs)
        n_obs_checked += ref_obs.size
        status = "OK" if ok else "MISMATCH"
        print(f"[b2-small] row={b} p={P_LIST[b]}: obs={status}")
        if not ok:
            fail = True
            diff_steps = np.flatnonzero(np.any(ref_obs != got_obs, axis=1))
            print(f"  obs mismatch at steps: {diff_steps[:10].tolist()} (total {len(diff_steps)})")
            j0 = int(diff_steps[0])
            d = np.flatnonzero(ref_obs[j0] != got_obs[j0])
            print(f"  step={j0} mismatching dims: {d[:20].tolist()}")
            for dd in d[:5]:
                print(f"    dim={dd}: ref={ref_obs[j0,dd]!r} got={got_obs[j0,dd]!r}")
    if fail:
        print("[b2-small] FAIL")
        return False
    print(f"[b2-small] PASS: obs {n_obs_checked} cells match (B={B} x N_JOBS={N_JOBS} x 224)")
    return True


def check_b2_big(big_ctx) -> bool:
    jobs, actions_full, n_on, n_cl, e_max, kk, raw_ref = big_ctx
    n_jobs = jobs.shape[0]
    actions = actions_full[:1]  # B=1本
    print(f"[b2-big] n_jobs={n_jobs} n_on={n_on} n_cl={n_cl} B=1 (p={BIG_P_LIST[0]})")

    t0 = time.time()
    got = run_lockstep_rollout(
        jobs, actions, n_on, n_cl, e_max=e_max, k=kk, n_window=N_WINDOW, tpb=1,
        collect_obs=True,
    )
    dt = time.time() - t0
    print(f"[b2-big] lockstep(step+obs): {dt:.1f}s ovf={got['ovf'].tolist()}", flush=True)
    if got["ovf"].any():
        print(f"[b2-big] FAIL: ovf set {got['ovf'].tolist()}")
        return False

    # collect_obs=True(probe再利用経路)の配置結果も raw と一致することを確認。
    mism_sc = (
        (raw_ref["start_times"][:1] != got["start_times"])
        | (raw_ref["waits"][:1] != got["waits"])
        | (raw_ref["costs"][:1] != got["costs"])
    )
    if mism_sc.any():
        print(f"[b2-big] FAIL: (start,wait,cost) mismatch vs raw with collect_obs=True "
              f"({int(mism_sc.sum())} cells)")
        return False
    print(f"[b2-big] (start,wait,cost) with collect_obs=True: {n_jobs*3}/{n_jobs*3} match vs raw")

    t0 = time.time()
    replay = build_replay_dataset(
        jobs, actions, got["start_times"], n_on, n_cl, nproc=1,
        env_kwargs=dict(n_window=N_WINDOW),
    )
    print(f"[b2-big] replay_obs_builder: {time.time()-t0:.1f}s", flush=True)

    ref_obs = replay[0]["obs"]
    got_obs = got["obs"][0]
    ok = np.array_equal(ref_obs, got_obs)
    if not ok:
        diff_steps = np.flatnonzero(np.any(ref_obs != got_obs, axis=1))
        print(f"[b2-big] FAIL: obs mismatch at {len(diff_steps)} steps "
              f"(first: {diff_steps[:10].tolist()})")
        j0 = int(diff_steps[0])
        d = np.flatnonzero(ref_obs[j0] != got_obs[j0])
        print(f"  step={j0} mismatching dims: {d[:20].tolist()}")
        for dd in d[:5]:
            print(f"    dim={dd}: ref={ref_obs[j0,dd]!r} got={got_obs[j0,dd]!r}")
        return False
    print(f"[b2-big] PASS: obs {ref_obs.size} cells match (N_JOBS={n_jobs} x 224)")
    return True


# ---------------------------------------------------------------------------
# (c) 性能
# ---------------------------------------------------------------------------
def perf_test(big_ctx) -> None:
    jobs, _actions_small, n_on, n_cl, e_max, kk, _raw_ref = big_ctx
    n_jobs = jobs.shape[0]
    B_PERF = 64

    def gen_actions_spread(n_jobs, B, seed, p_lo=0.2, p_hi=1.0):
        rng = np.random.default_rng(seed)
        ps = np.linspace(p_lo, p_hi, B)
        return (rng.random((B, n_jobs)) < ps[:, None]).astype(np.int8)

    actions = gen_actions_spread(n_jobs, B_PERF, SEED + 7)

    print(f"\n[perf] n_jobs={n_jobs} B={B_PERF} E_MAX={e_max} K={kk} tpb=1")

    # warmup(JITコンパイル+CUDA初期化を小問題で済ませる)
    run_lockstep_rollout(jobs[:256], actions[:1, :256], n_on, n_cl, e_max=1024, k=kk,
                          n_window=N_WINDOW, tpb=1, collect_obs=False)
    run_lockstep_rollout(jobs[:256], actions[:1, :256], n_on, n_cl, e_max=1024, k=kk,
                          n_window=N_WINDOW, tpb=1, collect_obs=True)

    # (c-1) 既存 run_raw_rollout(全周カーネル内、比較基準)
    t0 = time.time()
    ref = run_raw_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=kk, tpb=1)
    dt_ref = time.time() - t0
    print(f"[perf] run_raw_rollout(全周1カーネル): {dt_ref:.2f}s ovf={int(ref['ovf'].sum())}")

    # (c-2) lockstep: step_kernelのみ(B-1単体、発行税込み)
    t0 = time.time()
    got1 = run_lockstep_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=kk,
                                 n_window=N_WINDOW, tpb=1, collect_obs=False)
    dt_step_only = time.time() - t0
    per_step_us = dt_step_only / n_jobs * 1e6
    print(f"[perf] lockstep step_kernelのみ: {dt_step_only:.2f}s "
          f"({per_step_us:.1f}us/step, launch数={n_jobs}) ovf={int(got1['ovf'].sum())} "
          f"倍率(対run_raw_rollout)={dt_step_only/dt_ref:.2f}x")

    # (c-3) lockstep: step+obs 両方(Phase3で実際に必要な構成)
    t0 = time.time()
    got2 = run_lockstep_rollout(jobs, actions, n_on, n_cl, e_max=e_max, k=kk,
                                 n_window=N_WINDOW, tpb=1, collect_obs=True)
    dt_step_obs = time.time() - t0
    per_step_us2 = dt_step_obs / n_jobs * 1e6
    print(f"[perf] lockstep step+obs: {dt_step_obs:.2f}s "
          f"({per_step_us2:.1f}us/step, launch数={n_jobs*2}) ovf={int(got2['ovf'].sum())} "
          f"倍率(対run_raw_rollout)={dt_step_obs/dt_ref:.2f}x "
          f"倍率(対step単体)={dt_step_obs/dt_step_only:.2f}x")

    mism = (
        (ref["start_times"] != got1["start_times"])
        | (ref["waits"] != got1["waits"])
        | (ref["costs"] != got1["costs"])
    )
    print(f"[perf] B={B_PERF} 本での start/wait/cost 再検算一致: "
          f"{'OK' if not mism.any() else f'MISMATCH {int(mism.sum())}'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-big", action="store_true")
    ap.add_argument("--skip-perf", action="store_true")
    args = ap.parse_args()

    ok = True
    ok &= check_b1_small()
    ok &= check_b2_small()

    big_ctx = None
    if not args.skip_big:
        res = check_b1_big()
        if isinstance(res, tuple):
            b1_big_ok, big_ctx = res
        else:
            b1_big_ok, big_ctx = res, None
        ok &= b1_big_ok
        if big_ctx is not None:
            ok &= check_b2_big(big_ctx)
        else:
            ok = False

    if not args.skip_perf and big_ctx is not None:
        perf_test(big_ctx)

    print("\n[verify_lockstep] " + ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
