"""Phase 3 ベンチ: rollout データ工場 vs Ray CPU Actor 収集。

計測:
  - 工場 episodes/sec を NJ∈{128,512} × B∈{256,1024}（指令付きサンプリング、
    warm(コンパイル込み)1回 + ベスト計測。Transition 変換込みも併記）
  - CPU Actor サンプリング逐次 1 コア実測（pcn_agent._run_episode eval_mode=False
    ×N_CPU_EPS 本）→ 16 actor 線形換算（楽観上限; 実際は fork 並列でメモリ帯域 ~3× の
    実測経験あり）
  - 「1 iteration 分の rollout（16actor×8ep=128ep）」を工場が何秒で出すか

参考（train.log 実測; 報告用）: run_synth512_nded512_r1 Phase3 = 595.3s / 100 iter /
16ep = 実効 2.69 eps/s（ただし learn・eval と async overlap した実効値）。

usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/bench_factory.py
  env: NJS=128,512 BS=256,1024 REPEAT=2 N_CPU_EPS=5 SEED=1
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
os.environ["PCN_FILM"] = "0"
os.environ["PCN_FOURIER_CMD"] = "0"
os.environ["PCN_OBS_LOG"] = "1"
os.environ.setdefault("PCN_HIDDEN_DIM", "512")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

MAIN_REPO = "/home/noguchi/scheduler-sim-for-cb"
NJS = [int(x) for x in os.environ.get("NJS", "128,512").split(",")]
BS = [int(x) for x in os.environ.get("BS", "256,1024").split(",")]
REPEAT = int(os.environ.get("REPEAT", "2"))
N_CPU_EPS = int(os.environ.get("N_CPU_EPS", "5"))
SEED = int(os.environ.get("SEED", "1"))
SAME_CMD = os.environ.get("SAME_CMD", "0") == "1"  # 1=同一指令バッチ(同質; 資源別ビューが効く)
SKIP_CPU = os.environ.get("SKIP_CPU", "0") == "1"
CFG = os.environ.get("CFG", f"{MAIN_REPO}/experiments/distributed_pcn/job_synthetic_pcn.yml")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def default_ckpt(nj: int) -> str:
    pats = [
        f"{MAIN_REPO}/experiments/distributed_pcn/"
        f"run_synth{nj}_nded{nj}_r1/*/iteration_100/model_iter_100.pth",
        # 1024 は nded 系が無いので素の synth run（run_synth1024_a 等）へフォールバック
        f"{MAIN_REPO}/experiments/distributed_pcn/"
        f"run_synth{nj}_a/*/iteration_100/model_iter_100.pth",
        f"{MAIN_REPO}/experiments/distributed_pcn/"
        f"run_synth{nj}_*/*/iteration_100/model_iter_100.pth",
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise AssertionError(pats[0])


def load_ref(nj: int) -> dict:
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"refeval_nj{nj}_seed{SEED}_ncmd40_*.npz")))
    if hits:
        return {k: v for k, v in np.load(hits[-1]).items()}
    return build_ref_fallback(nj)


def build_ref_fallback(nj: int) -> dict:
    """refeval キャッシュが無い NJ（1024 等）用: env から jobs と指令帯を生成してキャッシュ。

    指令は p=0/1 の2本の Bernoulli rollout で cost/wait 端点を実測し、
    その間を linspace した (cost, wait) を objectives_to_command で f32 化
    （ベンチ用途: NN 入力の代表帯であればよい）。
    """
    cache = os.path.join(DATA_DIR, f"benchref_nj{nj}_seed{SEED}.npz")
    if os.path.exists(cache):
        return {k: v for k, v in np.load(cache).items()}

    from scripts.pcn_replay_snapshot import create_eval_env, load_config
    from src.utils.pf_command_eval import objectives_to_command

    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=SEED, n_jobs=nj)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    ends = []
    for p in (0.0, 1.0):
        env.reset()
        rng = np.random.default_rng(1000 + int(p))
        tw = tc = 0.0
        n = 0
        done = False
        while not done:
            r = env.step(1 if rng.random() < p else 0)
            tw += -float(r[1][0])
            tc += -float(r[1][1])
            n += 1 if r[2] else 0
            done = r[-1]
        ends.append([tc, tw / max(1, n)])
    (c0, w0), (c1, w1) = ends
    cg = np.linspace(min(c0, c1), max(c0, c1), 40)
    wg = np.interp(cg, [min(c0, c1), max(c0, c1)],
                   [w0 if c0 < c1 else w1, w1 if c0 < c1 else w0])
    cmds = np.stack([
        objectives_to_command(float(cc), float(ww), nj).astype(np.float32)
        for cc, ww in zip(cg, wg)
    ])
    out = {
        "jobs": jobs, "commands": cmds,
        "n_on": np.int32(env.n_on_premise_node), "n_cl": np.int32(env.n_cloud_node),
        "n_window": np.int32(env.n_window),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"[benchref] wrote {cache} (端点 cost=[{c0:.0f},{c1:.0f}])")
    return out


def cpu_sampling_rate(nj: int, ref: dict, ckpt: str) -> float:
    """CPU Actor サンプリング（distributed Actor と同一演算列）の逐次 1 コア s/ep。"""
    import torch as th

    from scripts.pcn_replay_snapshot import create_eval_env, load_config
    from src.agents.pcn_agent import PCN

    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=SEED, n_jobs=nj)
    env.reset()
    ag = PCN(
        env, device="cpu", state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1.0, 1.0, 1.0 / max(1, nj)], dtype=np.float32),
        learning_rate=1e-3, batch_size=512,
        hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
        project_name="t", experiment_name="PCN", log=False, use_enhanced_model=False,
    )
    st = th.load(ckpt, map_location="cpu", weights_only=False)
    ag.model.load_state_dict(st.get("model_state_dict", st), strict=False)
    ag.model.eval()
    dr = ref["commands"][len(ref["commands"]) // 2].astype(np.float32)
    mx = np.full(2, np.inf, dtype=np.float32)
    ag._run_episode(env, dr.copy(), np.float32(nj), mx, eval_mode=False)  # warm
    t0 = time.perf_counter()
    for i in range(N_CPU_EPS):
        th.manual_seed(i)
        ag._run_episode(env, dr.copy(), np.float32(nj), mx, eval_mode=False)
    return (time.perf_counter() - t0) / N_CPU_EPS


def make_commands(cmds40: np.ndarray, nj: int, b: int):
    if SAME_CMD:
        return [(cmds40[len(cmds40) // 2], float(nj))] * b
    return [(cmds40[i % len(cmds40)], float(nj)) for i in range(b)]


def timed_factory(run_factory, jobs, b, commands, policy, chunk, envkw):
    """chunk ずつ逐次実行して総 wall を返す（episode_id はグローバル通し）。"""
    t0 = time.perf_counter()
    last = None
    for i0 in range(0, b, chunk):
        bb = min(chunk, b - i0)
        last = run_factory(
            jobs, bb, commands=commands[i0:i0 + bb], policy=policy,
            seed0=99, episode_id0=i0, **envkw,
        )
    return time.perf_counter() - t0, last


def main():
    import torch as th
    from jaxlib.xla_extension import XlaRuntimeError

    from factory_jax import SamplingPolicy, episodes_to_transitions, run_factory

    dev = "cuda" if th.cuda.is_available() else "cpu"
    print(f"# factory bench device={dev} repeat={REPEAT} "
          f"({'同一指令(同質)' if SAME_CMD else '多様指令(40cmd展開)'}バッチ)")
    rows = []
    for nj in NJS:
        ref = load_ref(nj)
        jobs = np.asarray(ref["jobs"], dtype=np.float64)
        envkw = dict(n_on=int(ref["n_on"]), n_cl=int(ref["n_cl"]),
                     n_window=int(ref["n_window"]))
        ckpt = default_ckpt(nj)
        cpu_per_ep = float("nan") if SKIP_CPU else cpu_sampling_rate(nj, ref, ckpt)
        print(f"NJ={nj}: CPU逐次サンプリング {cpu_per_ep:.3f}s/ep "
              f"(16actor線形換算 {16 / cpu_per_ep:.1f} eps/s) ckpt={os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt))))}")
        policy = SamplingPolicy(ckpt, nj, device=dev)
        cmds40 = ref["commands"]
        for b in BS:
            commands = make_commands(cmds40, nj, b)
            chunk = b
            while True:  # OOM auto_chunk
                try:
                    best = None
                    for rep in range(REPEAT + 1):
                        dt, res = timed_factory(run_factory, jobs, b, commands,
                                                policy, chunk, envkw)
                        if rep == 0:
                            warm = dt
                            t1 = time.perf_counter()
                            eps = episodes_to_transitions(res)
                            conv_dt = (time.perf_counter() - t1) * (b / max(1, len(eps)))
                        elif best is None or dt < best:
                            best = dt
                    break
                except XlaRuntimeError as e:
                    if "RESOURCE_EXHAUSTED" not in str(e) or chunk <= 32:
                        raise
                    chunk //= 2
                    print(f"  OOM → auto_chunk B={b}→chunk={chunk} で再試行", flush=True)
            eps_s = b / best
            eps_s_conv = b / (best + conv_dt)
            ray16 = 16 / cpu_per_ep
            rows.append((nj, b, chunk, best, warm, conv_dt, eps_s, eps_s_conv,
                         cpu_per_ep, ray16))
            print(
                f"NJ={nj:4d} B={b:5d}(chunk={chunk}): rollout {best:6.2f}s "
                f"(warm+compile {warm:.1f}s) +conv {conv_dt:.1f}s → {eps_s:7.1f} eps/s "
                f"(変換込み {eps_s_conv:.1f}) vs Ray16actor楽観 {ray16:.1f} eps/s "
                f"= x{eps_s / ray16:.1f}",
                flush=True,
            )

        # 1 iteration 分（16actor×8ep=128ep）。B=128 は初形状なので warm 1回でコンパイルを外す
        commands = make_commands(cmds40, nj, 128)
        timed_factory(run_factory, jobs, 128, commands, policy, 128, envkw)
        dt128, res = timed_factory(run_factory, jobs, 128, commands, policy, 128, envkw)
        t0 = time.perf_counter()
        episodes_to_transitions(res)
        conv128 = time.perf_counter() - t0
        ray_iter = 8 * cpu_per_ep  # 16actor×8ep を 16 並列線形で = 8ep 分の逐次時間
        print(
            f"NJ={nj}: 1iter分(16actor×8ep=128ep) 工場 {dt128:.2f}s(+conv {conv128:.2f}s) "
            f"vs Ray16actor楽観 {ray_iter:.1f}s = x{ray_iter / (dt128 + conv128):.1f}"
        )

    print("\n## summary")
    print("| NJ | B(chunk) | rollout | +変換 | eps/s | eps/s(変換込) | CPU s/ep | Ray16楽観 eps/s | 倍率 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for nj, b, chunk, best, warm, conv, eps_s, eps_sc, cpu, ray16 in rows:
        print(
            f"| {nj} | {b}({chunk}) | {best:.1f}s | {conv:.1f}s | {eps_s:.1f} | {eps_sc:.1f} "
            f"| {cpu:.3f} | {ray16:.1f} | x{eps_sc / ray16:.1f} |"
        )


if __name__ == "__main__":
    main()
