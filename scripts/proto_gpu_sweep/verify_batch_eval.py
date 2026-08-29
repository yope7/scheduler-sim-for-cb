"""Phase 2 一致検証: GPU バッチ eval vs 既存 CPU 経路（ag._run_episode 1本ずつ）。

参照 = eval_b2_compare.py の greedy 流儀そのもの:
  - env: create_eval_env(CFG, job_seed=SEED, n_jobs=NJ)（SchedulingEnvEventNative）
  - 指令: rp掃引40本（Bernoulli(p), rng=default_rng(1000+i)）→ cost範囲 linspace NCMD 本
          → wg=np.interp → objectives_to_command(cost, wait, NJ) f32
  - 実行: ag._run_episode(env, dr, np.float32(NJ), mx=inf, eval_mode=True)
          モデルは CPU（学習中eval の Actor は常に CPU; distributed_pcn.py:1040-1041）
  - 記録: transitions から行動列と観測列（obs は policy 入力と同一; _obs_for_policy は
          event-native env では素通し）、達成 [cost, avg_wait] = r[5]
参照は npz にキャッシュ（REFRESH=1 で再生成）。

GPU バッチ側 = batch_eval_jax.run_batch_eval（B=NCMD 同時, record_obs=True）。

合否:
  - 観測 (T,221) が全ステップ・全指令でビット一致（uint32 view）
  - 行動列 (T,) が全指令で一致
  - 達成 [cost, avg_wait] が全指令で厳密一致
  - タイ再計算の統計（回数 / flip 数 / バッチ↔参照 logit 最大偏差 / 最小非再計算ギャップ）

usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/proto_gpu_sweep/verify_batch_eval.py
  env: NJ=512 CKPT=<pth> SEED=1 NCMD=40 TIE_EPS=1e-3 REFRESH=0 REF_DEVICE=cpu
       既定は NJ に応じ run_synth{128,512}_nded*_r1 の iteration_100 を自動解決
"""
from __future__ import annotations

import glob
import os
import sys

# --- 学習時フラグと完全一致させる（src import より前）---
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
import torch as th  # noqa: E402

MAIN_REPO = "/home/noguchi/scheduler-sim-for-cb"
NJ = int(os.environ.get("NJ", "512"))
SEED = int(os.environ.get("SEED", "1"))
NCMD = int(os.environ.get("NCMD", "40"))
TIE_EPS = float(os.environ.get("TIE_EPS", "1e-3"))
REF_DEVICE = os.environ.get("REF_DEVICE", "cpu")
CFG = os.environ.get("CFG", f"{MAIN_REPO}/experiments/distributed_pcn/job_synthetic_pcn.yml")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def default_ckpt(nj: int) -> str:
    pat = f"{MAIN_REPO}/experiments/distributed_pcn/run_synth{nj}_nded{nj}_r1/*/iteration_100/model_iter_100.pth"
    hits = sorted(glob.glob(pat))
    assert hits, f"no ckpt for {pat}"
    return hits[-1]


CKPT = os.environ.get("CKPT") or default_ckpt(NJ)


def build_commands(env) -> np.ndarray:
    """eval_b2_compare.py:150-158 の rp掃引→cg/wg→objectives_to_command を忠実に再現。"""
    from src.utils.pf_command_eval import objectives_to_command

    rp = []
    for i, p in enumerate(np.linspace(0, 1, 40)):
        rng = np.random.default_rng(1000 + i)
        env.reset()
        tw = tc = 0.0
        n = 0
        done = False
        st = 0
        while not done and st < NJ + 5:
            a = 1 if rng.random() < p else 0
            r = env.step(a)
            tw += -float(r[1][0])
            tc += -float(r[1][1])
            n += 1 if r[2] else 0
            done = r[-1]
            st += 1
        rp.append([tc, tw / max(1, n)])
    rp = np.array(rp)
    cg = np.linspace(float(rp[:, 0].min()), float(rp[:, 0].max()), NCMD)
    wg = np.interp(cg, np.sort(rp[:, 0]), rp[np.argsort(rp[:, 0]), 1])
    return np.stack(
        [objectives_to_command(float(cc), float(ww), NJ).astype(np.float32)
         for cc, ww in zip(cg, wg)]
    )


def run_reference(cache_path: str) -> dict:
    """既存 CPU 経路で NCMD 指令を 1 本ずつ greedy 実行し、obs/行動/達成を記録。"""
    if os.path.exists(cache_path) and os.environ.get("REFRESH", "0") != "1":
        print(f"[ref] cache hit: {cache_path}")
        return {k: v for k, v in np.load(cache_path).items()}

    from scripts.pcn_replay_snapshot import create_eval_env, load_config
    from src.agents.pcn_agent import PCN

    cfg = load_config(CFG)
    env = create_eval_env(cfg, job_seed=SEED, n_jobs=NJ)
    env.reset()
    jobs = np.asarray(env.jobs, dtype=np.float64).copy()
    st = th.load(CKPT, map_location="cpu", weights_only=False)
    ag = PCN(
        env,
        device=REF_DEVICE,
        state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1.0, 1.0, 1.0 / max(1, NJ)], dtype=np.float32),
        learning_rate=1e-3,
        batch_size=512,
        hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
        project_name="t",
        experiment_name="PCN",
        log=False,
        use_enhanced_model=False,
    )
    ag.model.load_state_dict(st.get("model_state_dict", st), strict=False)
    ag.model.eval()

    commands = build_commands(env)                     # (NCMD,2) f32
    mx = np.full(2, np.inf, dtype=np.float32)
    actions = np.zeros((NCMD, NJ), dtype=np.int8)
    obs = np.zeros((NCMD, NJ, env.observation_space.shape[0]), dtype=np.float32)
    achieved = np.zeros((NCMD, 2), dtype=np.float64)
    import time as _time

    t0 = _time.perf_counter()
    for i, dr in enumerate(commands):
        r = ag._run_episode(env, dr.copy(), np.float32(NJ), mx, eval_mode=True)
        trs = r[0]
        assert len(trs) == NJ, f"episode len {len(trs)} != {NJ}"
        actions[i] = [int(t.action) for t in trs]
        obs[i] = np.stack([t.observation for t in trs])
        achieved[i] = [float(r[5][0]), float(r[5][1])]
    ref_wall = _time.perf_counter() - t0
    out = {
        "jobs": jobs,
        "commands": commands,
        "actions": actions,
        "obs": obs,
        "achieved": achieved,
        "n_on": np.int32(env.n_on_premise_node),
        "n_cl": np.int32(env.n_cloud_node),
        "n_window": np.int32(env.n_window),
        "ref_wall_s": np.float64(ref_wall),
        "ckpt": np.str_(CKPT),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f"[ref] wrote {cache_path} (greedy {NCMD}cmd = {ref_wall:.1f}s 逐次1コア)")
    return out


def main():
    tag = os.path.basename(os.path.dirname(os.path.dirname(CKPT)))  # 20260708_xxxx
    cache = os.path.join(DATA_DIR, f"refeval_nj{NJ}_seed{SEED}_ncmd{NCMD}_{tag}.npz")
    ref = run_reference(cache)

    from batch_eval_jax import HybridGreedyPolicy, run_batch_eval

    policy = HybridGreedyPolicy(
        CKPT, NJ, hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
        device="cuda" if th.cuda.is_available() else "cpu",
        ref_device=REF_DEVICE, tie_eps=TIE_EPS,
    )
    res = run_batch_eval(
        ref["jobs"], list(ref["commands"]), policy,
        n_on=int(ref["n_on"]), n_cl=int(ref["n_cl"]), n_window=int(ref["n_window"]),
        record_obs=True,
    )

    n_bad = 0
    obs_ref = ref["obs"].view(np.uint32)
    obs_gpu = res["obs"].view(np.uint32)
    for i in range(NCMD):
        act_ok = np.array_equal(res["actions"][i], ref["actions"][i])
        obs_ok = np.array_equal(obs_gpu[i], obs_ref[i])
        ach_ok = np.array_equal(res["achieved"][i], ref["achieved"][i])
        if act_ok and obs_ok and ach_ok:
            continue
        n_bad += 1
        # 原因分析: 最初に割れた step / 観測特徴を特定
        da = np.flatnonzero(res["actions"][i] != ref["actions"][i])
        do_step, do_feat = -1, -1
        dobs = np.argwhere(obs_gpu[i] != obs_ref[i])
        if dobs.size:
            do_step, do_feat = int(dobs[0][0]), int(dobs[0][1])
        print(
            f"MISMATCH cmd {i}: dr={ref['commands'][i]} act_ok={act_ok}"
            f"(first_diff_t={da[0] if da.size else -1}) obs_ok={obs_ok}"
            f"(first t={do_step} feat={do_feat}) ach_ok={ach_ok} "
            f"ref={ref['achieved'][i]} gpu={res['achieved'][i]}"
        )
    s = res["stats"]
    print(
        f"\n[tie-recheck] rows={s['n_rows']} recheck={s['n_recheck']} "
        f"({s['n_recheck'] / max(1, s['n_rows']):.2%}) argmax_flip={s['n_flip']} "
        f"max|batch-ref|logit={s['max_dev']:.3g} "
        f"min_gap(non-recheck)={s['min_gap_norecheck']:.3g} tie_eps={TIE_EPS:g}"
    )
    print(f"[time] env={s['t_env']:.1f}s obs={s['t_obs']:.1f}s nn={s['t_nn']:.1f}s")
    tagm = "OK " if n_bad == 0 else "FAIL"
    print(
        f"{tagm} NJ={NJ} ckpt={os.path.basename(CKPT)}: "
        f"{NCMD - n_bad}/{NCMD} 指令が 観測bit/行動列/達成点 完全一致"
    )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
