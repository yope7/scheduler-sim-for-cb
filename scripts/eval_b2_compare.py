#!/usr/bin/env python3
"""b2比較用: 1方策(env flag で決定)を未知seedで走らせ、greedy/best-of-k/rp掃引を npz に保存。
   _FOURIER_CMD はモジュール import 時に固定なので、film(=0)とfourier(=1)は別プロセスで実行する。
   後段の plot_b2_compare.py が、両者の greedy を「共通の真PF」に重ねて gap を比較する。

   高速化: NPROC>1 で独立エピソード(rp/greedy/samp)を spawn 並列実行(CPU worker, env が CPU律速)。
   greedy は argmax=決定的でビット一致。samp は th.multinomial(global RNG)なので per-episode seed
   (SAMP_SEED_BASE + i*K + k)で再現可能化→並列でも質(参照PF分布)不変。NPROC<=1 は従来の逐次(samp は
   共有RNG)で完全後方互換。env コードは一切変更しない=配置スケジュール=結果不変。
   usage: CKPT=.. CFG=.. NJ=128 SEEDS=1,2 NCMD=40 KSAMP=10 NPROC=32 OUT=truepf_x.npz python scripts/eval_b2_compare.py"""
import os
# 並列(NPROC>1)時は各 worker の numpy/BLAS/torch を単一スレッド化する。
# でないと worker数 × BLASスレッド数(既定=コア数) のスレッドが全コアを奪い合い(over-subscription)、
# 並列化しても速くならない。numpy/torch の import より前に環境変数で設定する必要がある。
if int(os.environ.get("NPROC", "1")) > 1:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")
os.environ.setdefault("DISTRIBUTED_PCN_USE_EVENT_OBS", "1")
os.environ.setdefault("SCHEDULER_LEARNER_BITMAP", "0")
if os.environ.get("OBS_URGENCY", "1") == "1":
    os.environ["SCHEDULER_OBS_URGENCY"] = "1"
# occupancy(レバレッジ)観測は既定OFF。学習時に有効だった場合のみ eval も合わせる必要があるので、
# OBS_OCCUPANCY=1 か SCHEDULER_OBS_OCCUPANCY=1 が渡された時だけ有効化（dim 不一致回避）。
if os.environ.get("OBS_OCCUPANCY", "0") == "1":
    os.environ["SCHEDULER_OBS_OCCUPANCY"] = "1"
import numpy as np, torch as th
from scripts.pcn_replay_snapshot import create_eval_env, load_config
from src.agents.pcn_agent import PCN
from src.utils.pf_command_eval import objectives_to_command

CKPT = os.environ["CKPT"]; CFG = os.environ["CFG"]; NJ = int(os.environ.get("NJ", "128"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2").split(",")]
NCMD = int(os.environ.get("NCMD", "40")); K = int(os.environ.get("KSAMP", "10"))
NPROC = int(os.environ.get("NPROC", "1"))
SAMP_SEED_BASE = int(os.environ.get("SAMP_SEED_BASE", "7000"))
# CG_LIST: 指令 cost 列の明示指定(カンマ区切りfloat)。逆写像較正などで linspace の代わりに
# 任意の指令列を使う用。指定時は NCMD を無視して cg=CG_LIST とする(wg は従来どおり rp から interp)。
# 未指定(既定)は完全に従来どおり。
CG_LIST = os.environ.get("CG_LIST", "")
# WG_LIST: 指令 wait(平均待ち)列の明示指定(カンマ区切りfloat, CG_LIST と同型・同じ長さ必須)。
# 指定時は rp 補間の wg を置き換える。未指定(既定)は完全に従来どおり。
WG_LIST = os.environ.get("WG_LIST", "")
# NPROC>1(fork並列)では親プロセスで CUDA を初期化しない(fork後にCUDAは使えない)。worker は CPU。
if NPROC > 1:
    DEVICE = "cpu"
else:
    DEVICE = os.environ.get("DEVICE", "cuda" if th.cuda.is_available() else "cpu")
OUT = os.environ.get("OUT", "truepf_x.npz")
# H1b(スケール転移実験)用: donor checkpoint の指令正規化(center/scale)と scaling_factor を
# ロード後に上書きする(既定OFF=空。指定しない限り従来と完全一致)。
OVERRIDE_DR_FROM = os.environ.get("OVERRIDE_DR_FROM", "")
# DR_AUTO_RENORM(workload実レンジで指令再正規化): sequential は _apply 内、並列は
# タスクに scale を同梱し worker の _ep で毎回 set する(2026-08-17 並列対応。既定OFF=従来ビット一致)。


def _apply_dr_override(tg, ckpt_path):
    import torch as _th
    don = _th.load(ckpt_path, map_location="cpu", weights_only=False)
    dsd = don.get("model_state_dict", don)
    tg.set_desired_return_normalization(dsd["desired_return_center"].numpy(),
                                        dsd["desired_return_scale"].numpy())
    tg.scaling_factor.data.copy_(dsd["scaling_factor"])
    print(f"[dr-override] from {ckpt_path}: scale={dsd['desired_return_scale'].tolist()} "
          f"sf={dsd['scaling_factor'].tolist()}", flush=True)


# --- worker 側のプロセス内グローバル（spawn 後に _winit で構築）---
_W = {}


def _winit(cfg, ckpt_path, nj, seed, use_enh, device):
    """worker プロセス1個につき env(seed) と model を一度だけ構築。"""
    import numpy as _np, torch as _th
    _th.set_num_threads(1)  # 各 worker を単一スレッド化→ NPROC 個でコア飽和(over-subscription回避)
    from scripts.pcn_replay_snapshot import create_eval_env as _cee
    from src.agents.pcn_agent import PCN as _PCN
    env = _cee(cfg, job_seed=seed, n_jobs=nj)
    st = _th.load(ckpt_path, map_location="cpu", weights_only=False)
    ag = _PCN(env, device=device, state_dim=env.observation_space.shape[0],
              scaling_factor=_np.array([1., 1., 1. / max(1, nj)], dtype=_np.float32), learning_rate=1e-3,
              batch_size=512, hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")), project_name="t", experiment_name="PCN", log=False,
              use_enhanced_model=use_enh)
    tg = ag.network if ag.use_enhanced_model else ag.model
    tg.load_state_dict(st.get("model_state_dict", st), strict=False); tg.eval()
    if OVERRIDE_DR_FROM:
        _apply_dr_override(tg, OVERRIDE_DR_FROM)
    _W["env"] = env; _W["ag"] = ag; _W["mx"] = _np.full(2, _np.inf, dtype=_np.float32); _W["nj"] = nj


def _rp_one(p, seed):
    env = _W["env"]; nj = _W["nj"]; rng = np.random.default_rng(seed); env.reset()
    tw = tc = 0.0; n = 0; done = False; st = 0
    while not done and st < nj + 5:
        a = 1 if rng.random() < p else 0
        r = env.step(a); tw += -float(r[1][0]); tc += -float(r[1][1]); n += 1 if r[2] else 0; done = r[-1]; st += 1
    return [tc, tw / max(1, n)]


def _ep(dr, eval_mode, seed=None, dr_scale=None):
    if seed is not None:
        th.manual_seed(seed)
    env = _W["env"]; ag = _W["ag"]
    if dr_scale is not None:
        # [DR_AUTO_RENORM 並列対応] worker 内モデルへ workload 実レンジの指令正規化を設定(冪等)
        _tg = ag.network if ag.use_enhanced_model else ag.model
        _tg.set_desired_return_normalization(np.zeros(2, dtype=np.float32),
                                             np.asarray(dr_scale, dtype=np.float32))
    r = ag._run_episode(env, np.asarray(dr, dtype=np.float32), np.float32(_W["nj"]), _W["mx"], eval_mode=eval_mode)
    return [float(r[5][0]), float(r[5][1])]


def _task(t):
    """独立タスク1個を実行。t=(kind, ...)。結果は (kind, idx[, k], [cost,wait])。"""
    kind = t[0]
    if kind == "rp":
        return ("rp", t[1], _rp_one(t[2], t[3]))
    if kind == "greedy":
        _sc = t[3] if len(t) > 3 else None
        return ("greedy", t[1], _ep(t[2], True, dr_scale=_sc))
    # samp: seed あり=再現可能(並列), None=共有RNG(逐次後方互換)
    _sc = t[5] if len(t) > 5 else None
    return ("samp", t[1], t[2], _ep(t[3], False, t[4], dr_scale=_sc))


def _build_tasks(cg, wg, parallel):
    tasks = []
    for i, p in enumerate(np.linspace(0, 1, 40)):
        tasks.append(("rp", i, float(p), 1000 + i))
    for i, (cc, ww) in enumerate(zip(cg, wg)):
        dr = objectives_to_command(float(cc), float(ww), NJ).astype(np.float32).tolist()
        tasks.append(("greedy", i, dr))
        for k in range(K):
            seed = (SAMP_SEED_BASE + i * K + k) if parallel else None
            tasks.append(("samp", i, k, dr, seed))
    return tasks


def _run_seed_parallel(cfg, seed, use_enh):
    """rp の掃引→ cg を先に作る必要があるので、rp は先に逐次1パスで作ってから command を並列。
    ただし rp も独立なので全部1バッチで並列実行する: 先に worker で rp を回し cg を決めるのは
    順序依存なので、rp は全 worker 共通の軽い掃引(40本)として別フェーズで並列実行する。"""
    import multiprocessing as mp
    th.set_num_threads(1)  # 親を単一スレッド化(fork前)→ worker 飽和を妨げない
    ctx = mp.get_context("fork")  # fork=親のimport済torchを継承→ spawn の ~2.3s/worker 再importを回避(親はCUDA未初期化)
    initargs = (cfg, CKPT, NJ, seed, use_enh, "cpu")
    nw = min(NPROC, 40 + NCMD * (1 + K))  # worker 数はタスク数を超えない
    rp_tasks = [("rp", i, float(p), 1000 + i) for i, p in enumerate(np.linspace(0, 1, 40))]
    # pool は1個だけ作り rp フェーズと cmd フェーズで再利用（init オーバーヘッドを半減）
    with ctx.Pool(nw, initializer=_winit, initargs=initargs) as pool:
        rp_res = pool.map(_task, rp_tasks, chunksize=2)
        rp = np.array([r[2] for r in sorted(rp_res, key=lambda x: x[1])])
        # CG_MAX: 指令 cost 上限の明示指定(既定=rp最大で従来どおり)。学習分布外の指令が
        # Fourier 命令エンコードの外挿エイリアスで折り返すのを避け、真PF範囲等で公正に評価する用。
        _cg_hi = float(os.environ.get("CG_MAX", "0")) or float(rp[:, 0].max())
        cg = np.linspace(float(rp[:, 0].min()), min(_cg_hi, float(rp[:, 0].max())), NCMD)
        if CG_LIST:
            cg = np.array([float(x) for x in CG_LIST.split(",")], dtype=np.float64)
        wg = np.interp(cg, np.sort(rp[:, 0]), rp[np.argsort(rp[:, 0]), 1])
        if WG_LIST:
            wg = np.array([float(x) for x in WG_LIST.split(",")], dtype=np.float64)
            assert len(wg) == len(cg), f"WG_LIST length {len(wg)} != cg length {len(cg)}"
        # [DR_AUTO_RENORM 並列対応] scale は rp 掃引後に決まるが worker は fork 済みのため、
        # タスクに scale を同梱し _ep 側で毎回 set する(冪等)。逐次(NPROC=1)経路と同じ式。
        _dr_scale = None
        if os.environ.get("DR_AUTO_RENORM", "0") == "1":
            _cmax = max(abs(float(cg.min())), abs(float(cg.max())), 1.0)
            _wmax = max(abs(float((wg * NJ).min())), abs(float((wg * NJ).max())), 1.0)
            _dr_scale = [float(_wmax), float(_cmax)]
            print(f"[DR_AUTO_RENORM] (parallel) scale→[wait*nj={_wmax:.3g}, cost={_cmax:.3g}]", flush=True)
        cmd_tasks = []
        for i, (cc, ww) in enumerate(zip(cg, wg)):
            dr = objectives_to_command(float(cc), float(ww), NJ).astype(np.float32).tolist()
            cmd_tasks.append(("greedy", i, dr, _dr_scale))
            for k in range(K):
                cmd_tasks.append(("samp", i, k, dr, SAMP_SEED_BASE + i * K + k, _dr_scale))
        res = pool.map(_task, cmd_tasks, chunksize=2)
    greedy = [None] * len(cg)
    samp = []
    for r in res:
        if r[0] == "greedy":
            greedy[r[1]] = r[2]
        else:
            samp.append(r[3])
    return np.array(greedy), np.array(samp), rp


def _run_seed_sequential(seed, use_enh):
    """NPROC<=1: 従来と完全一致(samp は共有 global RNG, seed なし)。"""
    env = create_eval_env(config, job_seed=seed, n_jobs=NJ)
    ag = PCN(env, device=DEVICE, state_dim=env.observation_space.shape[0],
             scaling_factor=np.array([1., 1., 1. / max(1, NJ)], dtype=np.float32), learning_rate=1e-3,
             batch_size=512, hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")), project_name="t", experiment_name="PCN", log=False,
             use_enhanced_model=use_enh)
    tg = ag.network if ag.use_enhanced_model else ag.model
    tg.load_state_dict(state.get("model_state_dict", state), strict=False); tg.eval()
    if OVERRIDE_DR_FROM:
        _apply_dr_override(tg, OVERRIDE_DR_FROM)
    mx = np.full(2, np.inf, dtype=np.float32)
    rp = []
    for i, p in enumerate(np.linspace(0, 1, 40)):
        rng = np.random.default_rng(1000 + i); env.reset(); tw = tc = 0.0; n = 0; done = False; st = 0
        while not done and st < NJ + 5:
            a = 1 if rng.random() < p else 0
            r = env.step(a); tw += -float(r[1][0]); tc += -float(r[1][1]); n += 1 if r[2] else 0; done = r[-1]; st += 1
        rp.append([tc, tw / max(1, n)])
    rp = np.array(rp)
    _cg_hi = float(os.environ.get("CG_MAX", "0")) or float(rp[:, 0].max())
    cg = np.linspace(float(rp[:, 0].min()), min(_cg_hi, float(rp[:, 0].max())), NCMD)
    if CG_LIST:
        cg = np.array([float(x) for x in CG_LIST.split(",")], dtype=np.float64)
    wg = np.interp(cg, np.sort(rp[:, 0]), rp[np.argsort(rp[:, 0]), 1])
    if WG_LIST:
        wg = np.array([float(x) for x in WG_LIST.split(",")], dtype=np.float64)
        assert len(wg) == len(cg), f"WG_LIST length {len(wg)} != cg length {len(cg)}"
    # 推論適応: この workload の rp掃引レンジから指令正規化scaleを再設定(center=0固定)。
    # 学習scaleが別scaleで焼き込まれてると指令が圧縮/OODになり崩壊→workload自身のレンジで正規化し直す。
    if os.environ.get("DR_AUTO_RENORM", "0") == "1":
        _cmax = max(abs(float(cg.min())), abs(float(cg.max())), 1.0)
        _wmax = max(abs(float((wg * NJ).min())), abs(float((wg * NJ).max())), 1.0)
        tg.set_desired_return_normalization(np.array([0.0, 0.0], dtype=np.float32),
                                            np.array([_wmax, _cmax], dtype=np.float32))
        print(f"[DR_AUTO_RENORM] scale→[wait*nj={_wmax:.3g}, cost={_cmax:.3g}] (workload実レンジ)", flush=True)
    greedy, samp = [], []
    for cc, ww in zip(cg, wg):
        dr = objectives_to_command(float(cc), float(ww), NJ).astype(np.float32)
        r = ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=True)
        greedy.append([float(r[5][0]), float(r[5][1])])
        for _ in range(K):
            r = ag._run_episode(env, dr, np.float32(NJ), mx, eval_mode=False)
            samp.append([float(r[5][0]), float(r[5][1])])
    return np.array(greedy), np.array(samp), rp


if __name__ == "__main__":
    config = load_config(CFG)
    state = th.load(CKPT, map_location="cpu", weights_only=False)
    use_enh = (state.get("model_type", "") == "EnhancedPCNModel")
    out = {}
    for sd in SEEDS:
        if NPROC > 1:
            greedy, samp, rp = _run_seed_parallel(config, sd, use_enh)
        else:
            greedy, samp, rp = _run_seed_sequential(sd, use_enh)
        out[f"greedy_{sd}"] = greedy; out[f"samp_{sd}"] = samp; out[f"rp_{sd}"] = rp
        print(f"seed={sd}: greedy={len(greedy)} samp={len(samp)} (NPROC={NPROC})")
    np.savez(OUT, **out); print(f"saved {OUT}")
