"""Phase 6: 完全融合 rollout — env step + JAX方策 + obs + Bernoulli を1つの lax.scan に融合。

現 run_factory（factory_jax.py）は `for t in range(T):` の Python ループで、毎ステップ
  obs_h = np.asarray(obs_dev)         # GPU→host
  p = policy.p_cloud(...)  # torch .cpu().numpy()
とホスト往復する（T=512 で 512 往復、5万で 5万往復）。JAX高速化の典型ミス#1。

ここでは PCN 方策（DiscreteActionsDefaultModel, synth レシピ: FILM=0/Fourier=0=素の
Hadamard, OBS_LOG=1, ADAPTIVE_RETURN_NORMALIZATION=1）を JAX に移植し、torch ckpt 重みを
JAX 配列でロード。env step（alloc∘apply）+ obs（make_obs_fn）+ 方策 forward +
Bernoulli(per-step JAX RNG) を1つの jit 済み lax.scan に融合してホスト往復をゼロにする。

工場は bit 一致不要（分布一致=KS で確認）。E_max はスキャン内で固定（動的成長・compaction
なし）。無効イベントは INF64/-1/occ=False で自然排除されるので結果不変。大N（alive~N の
過負荷ワークロード）では E_max=N が要り O(N²) 以上になる（saturation 不成立=別途報告）。

usage: CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. で import
"""
from __future__ import annotations

import functools
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np

# [PCN_GPU_FACTORY] torch forward(pcn_agent.py:622)は (dr-center)/scale の後にさらに
# /_DESIRED_RETURN_SCALE で割る。pcn_p_cloud はこれを落としており dr が 10000倍大きくなって
# greedy(argmax)評価が常に onprem へ倒れていた（sampling では分布が崩れず学習は通り露見せず）。
_DESIRED_RETURN_SCALE = float(os.environ.get("PCN_DESIRED_RETURN_SCALE", "10000"))

from batch_env_jax import (  # noqa: E402
    INF64,
    _KERNELS,
    make_alloc_fn,
    make_apply_fn,
    make_initial_state,
)
from batch_eval_jax import precompute_job_queue  # noqa: E402
from factory_jax import (  # noqa: E402
    EVENT_FEATURES,
    JOB_QUEUE_SIZE,
    N_EVENTS_OBS,
    OBS_DIM,
    make_obs_fn,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402


# ---------------------------------------------------------------------------
# JAX ネイティブ PCN 方策（torch DiscreteActionsDefaultModel の synth forward 移植）
# ---------------------------------------------------------------------------
def params_from_state_dict(sd: dict) -> dict:
    """torch state_dict（在メモリ tensor / numpy どちらでも）→ 工場 forward の params dict。

    PCN_FC_DEPTH 可変: fc.{i}.weight (i=0,2,4,...) を全て列挙して層タプルにする
    （torch 側 DiscreteActionsDefaultModel: [Linear+ReLU]*(depth-1)+Linear+LogSoftmax）。
    FILM 時は film_gamma/film_beta、Fourier 時は fourier_freqs buffer（gaussian でも
    torch と同一行列）を抽出する。
    """
    import re

    def g(k):
        v = sd[k]
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        return jnp.asarray(arr, dtype=jnp.float32)

    fc_ids = sorted(
        int(m.group(1))
        for k in sd.keys()
        for m in [re.match(r"^fc\.(\d+)\.weight$", k)]
        if m
    )
    p = {
        "fc_w": tuple(g(f"fc.{i}.weight") for i in fc_ids),
        "fc_b": tuple(g(f"fc.{i}.bias") for i in fc_ids),
        "center": g("desired_return_center"), "scale": g("desired_return_scale"),  # (2,)
        "sf": g("scaling_factor"),                                 # (3,) = [1,1,1/N]
    }
    if _ARCH_ATTN:
        # AttnActionsModel: s_emb=_AttnStateEncoder(pcn_agent.py:743-818)
        n_layers = len({k.split(".")[2] for k in sd.keys()
                        if k.startswith("s_emb.layers.")})
        ap = {
            "ev_w": g("s_emb.event_proj.weight"), "ev_b": g("s_emb.event_proj.bias"),
            "qu_w": g("s_emb.queue_proj.weight"), "qu_b": g("s_emb.queue_proj.bias"),
            "type": g("s_emb.type_emb"),
            "pool_w": g("s_emb.pool_proj.weight"), "pool_b": g("s_emb.pool_proj.bias"),
            "oln_w": g("s_emb.out_ln.weight"), "oln_b": g("s_emb.out_ln.bias"),
            "layers": tuple(
                {
                    "ln1_w": g(f"s_emb.layers.{i}.ln1.weight"),
                    "ln1_b": g(f"s_emb.layers.{i}.ln1.bias"),
                    "ipw": g(f"s_emb.layers.{i}.attn.in_proj_weight"),
                    "ipb": g(f"s_emb.layers.{i}.attn.in_proj_bias"),
                    "opw": g(f"s_emb.layers.{i}.attn.out_proj.weight"),
                    "opb": g(f"s_emb.layers.{i}.attn.out_proj.bias"),
                    "ln2_w": g(f"s_emb.layers.{i}.ln2.weight"),
                    "ln2_b": g(f"s_emb.layers.{i}.ln2.bias"),
                    "f1_w": g(f"s_emb.layers.{i}.ffn.0.weight"),
                    "f1_b": g(f"s_emb.layers.{i}.ffn.0.bias"),
                    "f2_w": g(f"s_emb.layers.{i}.ffn.2.weight"),
                    "f2_b": g(f"s_emb.layers.{i}.ffn.2.bias"),
                }
                for i in range(n_layers)
            ),
        }
        if "s_emb.global_fuse.weight" in sd:
            ap["gf_w"] = g("s_emb.global_fuse.weight")
            ap["gf_b"] = g("s_emb.global_fuse.bias")
        p["attn"] = ap
    else:
        p["s_w"] = g("s_emb.0.weight")                             # (512,D)
        p["s_b"] = g("s_emb.0.bias")
    if _FILM:
        p["film_g_w"] = g("film_gamma.weight")
        p["film_g_b"] = g("film_gamma.bias")
        p["film_b_w"] = g("film_beta.weight")
        p["film_b_b"] = g("film_beta.bias")
    else:
        p["c_w"] = g("c_emb.0.weight")                             # (512,cmd_in)
        p["c_b"] = g("c_emb.0.bias")
    if _CMD_BALANCE and "command_balance" in sd:
        p["balance"] = g("command_balance")
    if _FOURIER and "fourier_freqs" in sd:
        p["fourier"] = g("fourier_freqs")
    # [per-channel bands] PCN_FOURIER_BANDS_COST: cost チャネル(index1)専用の周波数 buffer。
    # torch 側(BasePCNModel.__init__)は BANDS_COST≠BANDS の時のみ登録する。
    if _FOURIER and "fourier_freqs_cost" in sd:
        p["fourier_cost"] = g("fourier_freqs_cost")
    return p


def load_pcn_params(ckpt_path: str) -> dict:
    """torch checkpoint から JAX 方策パラメータ（重み+正規化バッファ）を抽出。"""
    import torch as th

    st = th.load(ckpt_path, map_location="cpu", weights_only=False)
    return params_from_state_dict(st.get("model_state_dict", st))


import os as _os
_FOURIER = _os.environ.get("PCN_FOURIER_CMD", "0") == "1"
_FBANDS = int(_os.environ.get("PCN_FOURIER_BANDS", "4"))
# [per-channel bands] cost チャネル(index1)のバンド数個別指定(torch pcn_agent.py と同名 env)。
# 未設定なら _FBANDS と同一=従来経路。
_FBANDS_COST_RAW = _os.environ.get("PCN_FOURIER_BANDS_COST", "")
_FBANDS_COST = int(_FBANDS_COST_RAW) if _FBANDS_COST_RAW != "" else _FBANDS
_F_COST_CH = 1  # c=[wait*nj, cost, horizon] の cost index（pcn_agent._FOURIER_COST_CH と同値）
_FMODE = _os.environ.get("PCN_FOURIER_MODE", "geometric").strip().lower()
_ADAPTIVE_RN = _os.environ.get("PCN_ADAPTIVE_RETURN_NORMALIZATION", "1") == "1"
# --- forward 系レシピ(torch pcn_agent.py と同名 env; gate 解除 2026-08) ---
_OBS_LOG = _os.environ.get("PCN_OBS_LOG", "0") == "1"          # sign*log1p 圧縮
_FILM = _os.environ.get("PCN_FILM", "0") == "1"                # FiLM 変調 forward
_CMD_BALANCE = _os.environ.get("PCN_COMMAND_BALANCE", "0") == "1"
_COND_WAIT_ROBUST = _os.environ.get("PCN_COND_WAIT_ROBUST", "off").strip().lower()
_COND_WAIT_K = float(_os.environ.get("PCN_COND_WAIT_K", "10.0"))
_COND_WAIT_Z0 = float(_os.environ.get("PCN_COND_WAIT_Z0", "1e-3"))  # logexpand用(pcn_agentと同一)
_COND_ADD_SCALE = float(_os.environ.get("PCN_COND_ADD_SCALE", "0"))
# --- rollout 中の指令更新(torch distributed_pcn.py:1580-1584 / pcn_agent.py:4370-4407) ---
_COST_HOLD = _os.environ.get("PCN_COST_HOLD", "0") == "1"
_DR_UB_RAW = _os.environ.get("PCN_DESIRED_RETURN_UB", "")
_DR_UB = float(_DR_UB_RAW) if _DR_UB_RAW != "" else None
# --- アーキ: PCN_ARCH=attn(_AttnStateEncoder+AttnActionsModel; pcn_agent.py:743-851) ---
_ARCH_ATTN = _os.environ.get("PCN_ARCH", "").strip().lower() == "attn"
_ATTN_HEADS = int(_os.environ.get("PCN_ATTN_HEADS", "4"))
# --- s_emb dropout(PCN_S_EMB_DROPOUT; pcn_agent.py:652-654) ---
# CPU 側 actor はモデルを train モードのまま rollout/eval するため、フラグ>0 なら
# _act の全 forward で dropout が有効(探索ノイズ)。工場も同分布で再現する
# (per-step/episode 独立 Bernoulli(1-p) マスク ×1/(1-p) = F.dropout と同一)。
_S_EMB_DROPOUT_P = min(max(float(_os.environ.get("PCN_S_EMB_DROPOUT", "0") or "0"), 0.0), 0.95)
# --- env 側レシピ ---
_SLOWDOWN = _os.environ.get("SCHEDULER_WAIT_METRIC", "wait") == "slowdown"
_OBS_OCCUPANCY = _os.environ.get("SCHEDULER_OBS_OCCUPANCY", "0") == "1"
_OBS_OCC_PRIOR = _os.environ.get("SCHEDULER_OBS_OCC_PRIOR", "0") == "1"
_ORACLE_NPZ = _os.environ.get("SCHEDULER_ORACLE_NPZ", "")
# [SCHEDULER_OBS_EFFICIENCY] 現ジョブの雲送り効率3次元(event_c_env.py:34-41 と同一定数)。
# 動的量(オンプレ/クラウド予測開始時刻の差)を含むので job_extra_feats(静的)では作れず、
# 各カーネルの body 内で s_on/s_cl から計算する(factory_defer_ev / factory_defer)。
_OBS_EFFICIENCY = _os.environ.get("SCHEDULER_OBS_EFFICIENCY", "0") == "1"
EFF_COST_K = 20.0
EFF_GAIN_K = 16.0
EFF_RATIO_K = 40.0
EFF_EXTRA_DIM = 3
# [SCHEDULER_OBS_BUDGET_RATIO] 現ジョブの「残り予算との比」3次元(event_c_env.py の
# BUD_A_K/BUD_FRAC_K/BUD_COVER_K と同一定数)。動的量(dr[:,1]=残り予算、現在位置以降の
# 雲コスト合計)を含むので job_extra_feats(静的)では作れず、各カーネルの body 内で
# dr と state.jobs_w/jobs_h から計算する(factory_defer_ev / factory_defer)。
_OBS_BUDGET_RATIO = _os.environ.get("SCHEDULER_OBS_BUDGET_RATIO", "0") == "1"
BUD_A_K = 20.0
BUD_FRAC_K = 0.6931471805599453
BUD_COVER_K = 20.0
BUDGET_EXTRA_DIM = 3


def obs_extra_dim() -> int:
    """urgency(基底221に含む)より後ろの追加観測次元数(event_c_env.get_observation と同順)。"""
    return ((1 if _OBS_OCCUPANCY else 0) + (1 if _ORACLE_NPZ else 0)
            + (EFF_EXTRA_DIM if _OBS_EFFICIENCY else 0)
            + (BUDGET_EXTRA_DIM if _OBS_BUDGET_RATIO else 0))


_oracle_cache = None


def _oracle_vals_np(n_jobs: int):
    """SCHEDULER_ORACLE_NPZ の per-job 真解雲率 (n_jobs,) f64。範囲外は 0（torch と同値）。"""
    global _oracle_cache
    if _oracle_cache is None:
        _oracle_cache = np.load(_ORACLE_NPZ)["vals"].astype(np.float32)
    out = np.zeros(n_jobs, dtype=np.float64)
    n = min(n_jobs, len(_oracle_cache))
    out[:n] = _oracle_cache[:n].astype(np.float64)
    return out


def job_extra_feats(jobs, n_window: int, n_on: int):
    """(T,k) f64: ジョブ index 静的な追加観測列 [occupancy?, oracle?]。

    event_c_env._front_job_leverage / _front_job_oracle と同値:
      occupancy(rank): searchsorted(sort(pt*nodes), occ, side='right')/N
      occupancy(OCC_PRIOR): clip(log1p((pt/n_window)*(nodes/n_on)*1e3)/4.3, 0, 1)
      oracle: npz vals[j]（範囲外 0）
    どちらもジョブ行だけの関数なので事前計算できる(count 工場用)。"""
    jobs = np.asarray(jobs, dtype=np.float64)
    cols = []
    if _OBS_OCCUPANCY:
        if _OBS_OCC_PRIOR:
            occ_frac = (jobs[:, 1] / float(max(1, n_window))) * (
                jobs[:, 2] / max(1.0, float(n_on)))
            cols.append(np.clip(np.log1p(occ_frac * 1.0e3) / 4.3, 0.0, 1.0))
        else:
            occ = jobs[:, 1] * jobs[:, 2]
            so = np.sort(occ)
            rank = np.searchsorted(so, occ, side="right").astype(np.float64) / len(so)
            cols.append(np.clip(rank, 0.0, 1.0))
    if _ORACLE_NPZ:
        cols.append(_oracle_vals_np(len(jobs)))
    if not cols:
        return np.zeros((len(jobs), 0), dtype=np.float64)
    return np.stack(cols, axis=1)


def _fourier_freqs_np(n_bands: int = None):
    import numpy as _np
    n = _FBANDS if n_bands is None else int(n_bands)
    if _FMODE == "linear":
        base = float(_os.environ.get("PCN_FOURIER_BASE", "6.283185307179586"))
        return _np.array([(k + 1) for k in range(n)], dtype=_np.float32) * base
    if _FMODE == "gaussian":
        # RFF (pcn_agent.py:539-542)と同一: torch.Generator(seed固定)由来の |N(0,1)|*scale。
        # torch で生成するので同 seed なら同一行列（通常は ckpt の fourier_freqs buffer を優先）。
        import torch as _th
        _g = _th.Generator().manual_seed(int(_os.environ.get("PCN_FOURIER_SEED", "0")))
        _sc = float(_os.environ.get("PCN_FOURIER_SCALE", "1.0"))
        return (_th.abs(_th.randn(n, generator=_g)) * _sc).numpy().astype(_np.float32)
    # geometric (既定, NeRF式 2^k)
    return _np.array([2.0 ** k for k in range(n)], dtype=_np.float32)


def pcn_logits(params: dict, obs, dr, hz, dkey=None):
    """logits(B,A) を JAX で。torch BasePCNModel.forward(pcn_agent.py:611-696) と同じ演算列。

    obs(B,D) f32, dr(B,2) f32, hz(B,) f32 → logits(B,A)。
    対応レシピ(モジュール定数; import 前に env 設定):
      PCN_FC_DEPTH 可変(fc_w/fc_b の層数に自動追従) / PCN_OBS_LOG / PCN_FILM /
      PCN_ARCH=attn / PCN_FOURIER_MODE 全種(ckpt の fourier_freqs buffer 優先=torch と
      同一行列) / PCN_COMMAND_BALANCE / PCN_COND_WAIT_ROBUST / PCN_COND_ADD_SCALE /
      PCN_S_EMB_DROPOUT(dkey 必須; CPU actor は train モードのため rollout でも有効)。
    LogSoftmax は単調なので argmax/categorical には Linear 出力(logits)で同値。
    bit一致は保証しない(GEMM順序差; 分布は同一)。
    """
    p = params
    dr = jnp.clip(dr, -1e12, 1e12)
    # torch(pcn_agent.py:621-624)は排他分岐: ADAPTIVE時は(dr-center)/scaleのみ、/10000は適用しない
    if _ADAPTIVE_RN:
        dr = (dr - p["center"]) / p["scale"]
    elif _DESIRED_RETURN_SCALE > 0:
        dr = dr / _DESIRED_RETURN_SCALE
    if _CMD_BALANCE and "balance" in p:
        dr = dr * p["balance"]          # _balance_desired_return(pcn_agent.py:596-600)
    if _COND_WAIT_ROBUST == "logexpand":
        # [2026-08-27] pcn_agent._robust_cond_wait の logexpand モードの鏡写し(モード不問の
        # softlog適用=サイレント規約ずれを封印)。y=sign(z)·log1p(|z|/z0)/log1p(1/z0)。
        w0 = dr[:, 0]
        dr = dr.at[:, 0].set(
            jnp.sign(w0) * jnp.log1p(jnp.abs(w0) / _COND_WAIT_Z0)
            / float(np.log1p(1.0 / _COND_WAIT_Z0)))
    elif _COND_WAIT_ROBUST not in ("off", "", "0", "softlog"):
        raise RuntimeError(
            f"factory_fused: PCN_COND_WAIT_ROBUST={_COND_WAIT_ROBUST!r} は未対応"
            "(サイレント旧式適用の禁止)")
    elif _COND_WAIT_ROBUST == "softlog":
        # _robust_cond_wait(pcn_agent.py): wait次元(index0)のみ soft-log 圧縮
        w0 = dr[:, 0]
        dr = dr.at[:, 0].set(
            jnp.sign(w0) * _COND_WAIT_K * jnp.log1p(jnp.abs(w0) / _COND_WAIT_K))
    hzc = jnp.clip(hz, 0.0, 1e6)[:, None]
    s = jnp.clip(obs, -1e6, 1e6)
    if _OBS_LOG:
        s = jnp.sign(s) * jnp.log1p(jnp.abs(s))
    # 条件 c = [dr, hz] * scaling_factor
    c = jnp.concatenate([dr, hzc], axis=-1) * p["sf"]
    c = jnp.nan_to_num(c)
    if _FOURIER:
        # torch _encode_cmd (pcn_agent.py:602-609)と同一: [c, sin(fk·c), cos(fk·c)] flatten(D-major)
        _fr = p["fourier"] if "fourier" in p else jnp.asarray(_fourier_freqs_np())
        if "fourier_cost" in p or _FBANDS_COST != _FBANDS:
            # [per-channel bands] cost チャネル(index1)のみ別周波数集合。torch _encode_cmd の
            # per-channel 分岐と同一レイアウト(D-major: [c, d0[sin,cos], d1[sin,cos], d2[sin,cos]])。
            _frc = (p["fourier_cost"] if "fourier_cost" in p
                    else jnp.asarray(_fourier_freqs_np(_FBANDS_COST)))
            parts = [c]
            for _d in range(c.shape[1]):
                _ff = _frc if _d == _F_COST_CH else _fr
                proj = c[:, _d : _d + 1] * _ff[None, :]
                parts.append(jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1))
            c = jnp.concatenate(parts, axis=-1)
        else:
            proj = c[:, :, None] * _fr[None, None, :]
            feats = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
            c = jnp.concatenate([c, feats.reshape(c.shape[0], -1)], axis=-1)
    if _ARCH_ATTN:
        s_emb = _attn_state_emb(p["attn"], s)
    else:
        s_emb = jax.nn.sigmoid(s @ p["s_w"].T + p["s_b"])
    if _S_EMB_DROPOUT_P > 0.0:
        # torch forward :652-654: s = F.dropout(s_emb, p)（train モード=actor 全 forward）。
        # mask~Bernoulli(1-p), 生存を 1/(1-p) 倍 = F.dropout と同一分布。
        if dkey is None:
            raise RuntimeError("PCN_S_EMB_DROPOUT>0 には dkey(乱数キー)が必要")
        keep = jax.random.bernoulli(dkey, 1.0 - _S_EMB_DROPOUT_P, s_emb.shape)
        s_emb = jnp.where(keep, s_emb / (1.0 - _S_EMB_DROPOUT_P), 0.0)
    s_emb = jnp.nan_to_num(s_emb)
    if _FILM:
        # torch forward :664-670: γ(c)=1+film_gamma(c), β(c)=film_beta(c), fc(s*γ+β)。c_emb は不使用。
        gamma = 1.0 + (c @ p["film_g_w"].T + p["film_g_b"])
        beta = c @ p["film_b_w"].T + p["film_b_b"]
        gamma = jnp.nan_to_num(gamma, nan=1.0, posinf=1.0, neginf=1.0)
        beta = jnp.nan_to_num(beta)
        h = s_emb * gamma + beta
    else:
        c_emb = jax.nn.sigmoid(c @ p["c_w"].T + p["c_b"])
        c_emb = jnp.nan_to_num(c_emb)
        if _COND_ADD_SCALE > 0.0:
            h = s_emb * c_emb + _COND_ADD_SCALE * c_emb
        else:
            h = s_emb * c_emb
    # fc: [Linear+ReLU]*(depth-1) + Linear(A) （PCN_FC_DEPTH 可変; 層数は params に追従）
    fc_w = p["fc_w"] if "fc_w" in p else (p["f0_w"], p["f2_w"])
    fc_b = p["fc_b"] if "fc_b" in p else (p["f0_b"], p["f2_b"])
    for wgt, b in zip(fc_w[:-1], fc_b[:-1]):
        h = jax.nn.relu(h @ wgt.T + b)
    logits = h @ fc_w[-1].T + fc_b[-1]                       # (B,A)
    return jnp.nan_to_num(logits)


def pcn_p_cloud(params: dict, obs, dr, hz, dkey=None):
    """P(action=1|obs,dr,hz) を JAX で（2行動系）。pcn_logits と同一演算列。

    LogSoftmax → exp → [:,1] = softmax[1] = P(cloud)。
    """
    logits = pcn_logits(params, obs, dr, hz, dkey=dkey)
    logp = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return jnp.exp(logp[:, 1])


# ---------------------------------------------------------------------------
# PCN_ARCH=attn: _AttnStateEncoder(pcn_agent.py:743-818)の JAX 移植
# ---------------------------------------------------------------------------
def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * w + b


def _mha(x, ipw, ipb, opw, opb, n_heads):
    """torch nn.MultiheadAttention(batch_first, self-attn) と同演算。x(B,T,D)。"""
    B, T, D = x.shape
    qkv = x @ ipw.T + ipb                          # (B,T,3D)
    q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
    dh = D // n_heads

    def split(a):
        return a.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)   # (B,H,T,dh)

    q, k, v = split(q), split(k), split(v)
    att = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(dh).astype(x.dtype),
                         axis=-1)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return o @ opw.T + opb


def _attn_state_emb(ap: dict, state):
    """_AttnStateEncoder.forward(pcn_agent.py:797-818) と同じ演算列。state(B,D>=220)。"""
    B = state.shape[0]
    ev = state[:, :180].reshape(B, 30, 6)
    qu = state[:, 180:220].reshape(B, 5, 8)
    tok_ev = ev @ ap["ev_w"].T + ap["ev_b"] + ap["type"][0]
    type_qu = jnp.concatenate(
        [ap["type"][1:2], jnp.broadcast_to(ap["type"][2:3], (4, ap["type"].shape[1]))],
        axis=0)
    tok_qu = qu @ ap["qu_w"].T + ap["qu_b"] + type_qu
    x = jnp.concatenate([tok_ev, tok_qu], axis=1)           # (B,35,d)
    for ly in ap["layers"]:
        h = _layer_norm(x, ly["ln1_w"], ly["ln1_b"])
        x = x + _mha(h, ly["ipw"], ly["ipb"], ly["opw"], ly["opb"], _ATTN_HEADS)
        h2 = _layer_norm(x, ly["ln2_w"], ly["ln2_b"])
        x = x + (jax.nn.relu(h2 @ ly["f1_w"].T + ly["f1_b"]) @ ly["f2_w"].T + ly["f2_b"])
    cur = x[:, 30, :]                                        # 現ジョブトークン
    pooled = jnp.concatenate([cur, x.mean(axis=1)], axis=-1)
    h = pooled @ ap["pool_w"].T + ap["pool_b"]
    if "gf_w" in ap:
        g = state[:, 220:]
        h = jnp.concatenate([h, g], axis=-1) @ ap["gf_w"].T + ap["gf_b"]
    return _layer_norm(h, ap["oln_w"], ap["oln_b"])


# ---------------------------------------------------------------------------
# 融合 rollout（1つの jit lax.scan）
# ---------------------------------------------------------------------------
def run_fused(jobs, commands, ckpt_path=None, params=None, *, seed=0,
              n_on=256, n_cl=1024, n_window=100, kernel="v1h"):
    """融合 rollout: jobs(T,8) × B=len(commands) 指令サンプリング。

    commands: [(dr(2,) f32, hz float), ...]。params 指定時は ckpt 再ロードを省く。
    返り値 dict: actions(B,T) i8, rewards(B,T,2) f64, achieved(B,3) f64。
    """
    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    B = len(commands)
    if params is None:
        params = load_pcn_params(ckpt_path)
    state = make_initial_state(np.broadcast_to(jobs, (B, T, 8)), n_on=n_on, n_cl=n_cl)
    h_max = int(jobs[:, 2].max())
    jq = jnp.asarray(precompute_job_queue(jobs))            # (T,40) f64
    arr = jnp.asarray(jobs[:, 0].astype(np.int64))          # (T,)
    dr0 = jnp.asarray(np.stack([np.asarray(c[0], np.float32) for c in commands]))
    hz0 = jnp.asarray(np.array([float(c[1]) for c in commands], np.float32))
    key = jax.random.PRNGKey(seed)

    # dr0/hz0 を初期 carry に注入するため _fused_rollout を薄くラップ（scan 内で置換）
    final, actions, rewards = _fused_rollout_run(
        state, params, key, jq, arr, dr0, hz0,
        n_on=n_on, n_cl=n_cl, h_max=h_max, n_window=n_window,
        norm_time=float(max(1, n_window)), norm_nodes=float(max(n_on, n_cl)),
        kernel=kernel,
    )
    jax.block_until_ready(actions)
    cost = np.asarray(final.cost_total, dtype=np.float64)
    avg_wait = np.asarray(final.waits.sum(axis=1), dtype=np.float64) / T
    makespan = np.asarray(jnp.max(final.ev_end, axis=(1, 2)), dtype=np.float64)
    return {
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards).transpose(1, 0, 2),   # (B,T,2)
        "achieved": np.stack([cost, makespan, avg_wait], axis=1),
    }


@functools.partial(
    jax.jit,
    static_argnames=("n_on", "n_cl", "h_max", "n_window", "norm_time", "norm_nodes", "kernel"),
)
def _fused_rollout_run(state, params, key, jq, arr, dr0, hz0, *, n_on, n_cl, h_max,
                       n_window, norm_time, norm_nodes, kernel):
    """dr0/hz0 を初期 carry にした融合 scan（本体）。"""
    B = state.jobs_w.shape[0]
    E = state.ev_start.shape[2]
    alloc = make_alloc_fn(h_max, kernel, E)
    apply = make_apply_fn(n_on, n_cl, h_max)
    obs_fn = make_obs_fn(n_window, norm_time, norm_nodes)

    def body(carry, xs):
        st, dr, hz = carry
        t, jq_t, arr_t = xs
        s_on, nd_on, s_cl, nd_cl = alloc(st)
        pw = jnp.maximum(0, s_on - arr_t)
        obs = obs_fn(st.ev_obs, t.astype(jnp.int32), st.time, jq_t, pw)
        p = pcn_p_cloud(params, obs, dr, hz)
        u = jax.random.uniform(jax.random.fold_in(key, t), (B,))
        acts = (u < p).astype(jnp.int32)
        st2, out = apply(st, acts, s_on, nd_on, s_cl, nd_cl)
        dr2 = (dr - out.rewards.astype(jnp.float32)).astype(jnp.float32)
        hz2 = jnp.maximum(hz - 1.0, 1.0)
        return (st2, dr2, hz2), (acts.astype(jnp.int8), out.rewards)

    ts = jnp.arange(jq.shape[0], dtype=jnp.int64)
    (final, _, _), (actions, rewards) = lax.scan(
        body, (state, dr0, hz0), (ts, jq, arr)
    )
    return final, actions, rewards


# ---------------------------------------------------------------------------
# 圧縮融合 rollout: scan 内で alloc/obs 両バッファを compact し E_max を固定
# ---------------------------------------------------------------------------
# 段3(やり直し)で「正しい容量(ノード粒度)では生存イベント数が N でなく到着同時実行数で
# 飽和」することが判明した。ここでは alloc 用イベント配列(資源別)と obs 用イベント配列
# (統一)を **scan 内で毎ステップ compact** し、E_alloc / E_obs を飽和値+マージンに固定する。
# これで融合 rollout が O(N²) → O(N·E_max) に線形化し、E=N が要らず 5万が回る。
# compact は「以後の全ステップで overlap/候補にならない死んだイベント」を落とすだけなので
# 結果不変（make_compact_fn と同じ論拠; verify_factory[c] が非融合で担保済み）。
# alloc 閾値 = suffmin(min 将来到着), obs 閾値 = time - n_window（time は非減少なので安全）。


def _obs_from_buffer(ob, ol, tm, jq_t, pw, *, n_window, norm_time, norm_nodes,
                     extras=None):
    """圧縮 obs バッファ ob(B,Eo,6) + per-B 有効数 ol(B,) から obs(B,221[+k]) を作る。

    make_obs_fn と同一の演算だが n_valid を per-B（ol）にした版（融合の各エピソードで
    obs バッファの詰まり具合が違うため）。
    extras: (B,k) f32 追加観測列(occupancy/oracle 等; event_c_env.get_observation の
    urgency より後ろと同順)。None なら従来 221 次元のまま（既定パス bit 不変）。
    """
    B, E, _ = ob.shape
    ws = jnp.maximum(0, tm - n_window)                          # (B,)
    starts = ob[:, :, 0]
    ends = ob[:, :, 1]
    keep = (jnp.arange(E)[None, :] < ol[:, None]) & (
        ends >= ws[:, None].astype(ob.dtype)
    )
    kk = min(N_EVENTS_OBS, E)
    idx64 = jnp.arange(E, dtype=jnp.int64)
    key = jnp.where(
        keep, starts.astype(jnp.int64) * jnp.int64(E + 1) + idx64[None, :],
        jnp.int64(-1),
    )
    topv, _ = jax.lax.top_k(key, kk)
    n_take = (topv >= 0).sum(axis=1).astype(jnp.int32)
    k = jnp.arange(N_EVENTS_OBS, dtype=jnp.int32)
    row_ok = k[None, :] < n_take[:, None]
    desc_pos = jnp.clip(n_take[:, None] - 1 - k[None, :], 0, kk - 1)
    sel = jnp.take_along_axis(topv, desc_pos[:, :kk] if kk < N_EVENTS_OBS else desc_pos, axis=1)
    if kk < N_EVENTS_OBS:
        sel = jnp.pad(sel, ((0, 0), (0, N_EVENTS_OBS - kk)), constant_values=-1)
    src_idx = (sel % jnp.int64(E + 1)).astype(jnp.int32)
    feats = jnp.take_along_axis(ob, src_idx[:, :, None], axis=1, mode="clip")
    wsf = ws[:, None].astype(jnp.float64)
    ev6 = jnp.stack(
        [
            jnp.clip((feats[:, :, 0] - wsf) / norm_time, 0.0, 1.0),
            jnp.clip((feats[:, :, 1] - wsf) / norm_time, 0.0, 1.0),
            jnp.clip(feats[:, :, 2] / norm_time, 0.0, 1.0),
            feats[:, :, 3],
            jnp.clip(feats[:, :, 4] / norm_nodes, 0.0, 1.0),
            jnp.clip(feats[:, :, 5] / norm_nodes, 0.0, 1.0),
        ],
        axis=2,
    )
    ev6 = jnp.where(row_ok[:, :, None], ev6, 0.0)
    ev_flat = ev6.reshape(B, N_EVENTS_OBS * EVENT_FEATURES).astype(jnp.float32)
    jq = jnp.broadcast_to(jq_t.astype(jnp.float32)[None, :], (B, JOB_QUEUE_SIZE))
    urg = jnp.clip(jnp.log1p(pw.astype(jnp.float64)) / 16.0, 0.0, 1.0).astype(jnp.float32)
    parts = [ev_flat, jq, urg[:, None]]
    if extras is not None:
        parts.append(extras.astype(jnp.float32))
    return jnp.concatenate(parts, axis=1)


def _compact_res(es_r, ee_r, occ, thr):
    """資源1つのイベント配列を alive(end>=thr) で前詰め（make_compact_fn と同一）。"""
    E = ee_r.shape[1]
    alive = ee_r >= thr
    order = jnp.argsort(~alive, axis=1, stable=True)
    cnt = alive.sum(axis=1).astype(jnp.int32)
    pos_ok = jnp.arange(E)[None, :] < cnt[:, None]
    st = jnp.where(pos_ok, jnp.take_along_axis(es_r, order, axis=1), INF64)
    en = jnp.where(pos_ok, jnp.take_along_axis(ee_r, order, axis=1), -1)
    occ2 = jnp.take_along_axis(occ, order[:, :, None], axis=1) & pos_ok[:, :, None]
    return st, en, occ2, cnt


def _compact_obs(ob, ol, thr):
    """obs バッファを alive(end>=thr, 有効slot のみ)で前詰め。thr は (B,)。"""
    E = ob.shape[1]
    alive = (jnp.arange(E)[None, :] < ol[:, None]) & (ob[:, :, 1] >= thr[:, None])
    order = jnp.argsort(~alive, axis=1, stable=True)
    cnt = alive.sum(axis=1).astype(jnp.int32)
    pos_ok = jnp.arange(E)[None, :] < cnt[:, None]
    ob2 = jnp.where(pos_ok[:, :, None], jnp.take_along_axis(ob, order[:, :, None], axis=1), 0.0)
    return ob2, cnt


@functools.partial(
    jax.jit,
    static_argnames=("n_on", "n_cl", "h_max", "n_window", "norm_time", "norm_nodes",
                     "kernel", "e_alloc", "e_obs", "kcompact", "elastic_cloud"),
)
def _fused_compact_run(jobs_cols, params, key, jq, dr0, hz0, *, n_on, n_cl, h_max,
                       n_window, norm_time, norm_nodes, kernel, e_alloc, e_obs, kcompact,
                       elastic_cloud):
    """圧縮融合 scan 本体。jobs_cols=(arr(T,),w(T,),h(T,),suffnext(T,)) i64。

    elastic_cloud=True: cloud を無限弾性(burst)とみなし start=arrival（待ち0・占有追跡不要）。
    cloud 側カーネル・occ_cl・cloud compaction を丸ごと省く（~2x）。物理的に cloud bursting は
    従量課金の弾性資源なので、有限 n_cl でモデル化する方がむしろ人工的。obs には cloud イベントも
    載せる（start=arr, start_node=0）。
    """
    arr_c, w_c, h_c, suff_c = jobs_cols
    B = dr0.shape[0]
    T = arr_c.shape[0]
    kfn = _KERNELS[kernel]
    bidx = jnp.arange(B)

    es = jnp.full((B, 2, e_alloc), INF64, dtype=jnp.int64)
    ee = jnp.full((B, 2, e_alloc), -1, dtype=jnp.int64)
    oon = jnp.zeros((B, e_alloc, n_on), dtype=bool)
    ocl = jnp.zeros((B, e_alloc, n_cl if not elastic_cloud else 1), dtype=bool)
    el = jnp.zeros((B, 2), dtype=jnp.int32)
    ob = jnp.zeros((B, e_obs, 6), dtype=jnp.float64)
    ol = jnp.zeros((B,), dtype=jnp.int32)
    tm = jnp.zeros((B,), dtype=jnp.int64)
    cost = jnp.zeros((B,), dtype=jnp.int64)
    wsum = jnp.zeros((B,), dtype=jnp.int64)
    mksp = jnp.zeros((B,), dtype=jnp.int64)
    ovf = jnp.zeros((), dtype=bool)            # E_max 超過検出

    def body(carry, xs):
        es, ee, oon, ocl, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf = carry
        t, jq_t, arr, w, h, suffnext = xs
        arr_b = jnp.full((B,), arr, jnp.int64)
        w_b = jnp.full((B,), w, jnp.int64)
        h_b = jnp.full((B,), h, jnp.int32)
        s_on, _, nd_on, _, _ = kfn(es[:, 0], ee[:, 0], oon, w_b, h_b, arr_b,
                                   jnp.zeros((B,), bool), h_max=h_max)
        if elastic_cloud:
            # 弾性cloud: 常に arrival に配置（待ち0）。カーネル不要。start_node は 0。
            s_cl = arr_b
            nd_cl = jnp.zeros((B, h_max), dtype=jnp.int32)
        else:
            s_cl, _, nd_cl, _, _ = kfn(es[:, 1], ee[:, 1], ocl, w_b, h_b, arr_b,
                                       jnp.ones((B,), bool), h_max=h_max)
        pw = jnp.maximum(0, s_on - arr)
        obs = _obs_from_buffer(ob, ol, tm, jq_t, pw, n_window=n_window,
                               norm_time=norm_time, norm_nodes=norm_nodes)
        p = pcn_p_cloud(params, obs, dr, hz)
        u = jax.random.uniform(jax.random.fold_in(key, t), (B,))
        use_cloud = u < p
        start = jnp.where(use_cloud, s_cl, s_on)
        nodes = jnp.where(use_cloud[:, None], nd_cl, nd_on)        # (B,h_max)
        end = start + w
        wait = start - arr
        cost_step = jnp.where(use_cloud, w * h, 0).astype(jnp.int64)
        rew = jnp.stack([-wait, -cost_step], axis=1).astype(jnp.float64)

        # --- append alloc ---
        karr = jnp.arange(h_max, dtype=jnp.int32)
        valid_k = karr[None, :] < h_b[:, None]
        if elastic_cloud:
            # onprem ジョブだけ alloc 配列/occ_on に追記（cloud は占有追跡しない）。
            is_on = ~use_cloud
            slot0 = el[bidx, 0]
            slot_w = jnp.where(is_on, slot0, e_alloc)              # cloud → 範囲外 drop
            es = es.at[bidx, 0, slot_w].set(start, mode="drop")
            ee = ee.at[bidx, 0, slot_w].set(end, mode="drop")
            nd_on_i = jnp.where(valid_k & is_on[:, None], nodes, n_on)
            oon = oon.at[bidx[:, None], slot0[:, None], nd_on_i].set(True, mode="drop")
            el = el.at[bidx, 0].add(is_on.astype(jnp.int32))
        else:
            r = use_cloud.astype(jnp.int32)
            slot = el[bidx, r]
            es = es.at[bidx, r, slot].set(start, mode="drop")
            ee = ee.at[bidx, r, slot].set(end, mode="drop")
            nd_on_i = jnp.where(valid_k & (~use_cloud)[:, None], nodes, n_on)
            nd_cl_i = jnp.where(valid_k & use_cloud[:, None], nodes, n_cl)
            oon = oon.at[bidx[:, None], slot[:, None], nd_on_i].set(True, mode="drop")
            ocl = ocl.at[bidx[:, None], slot[:, None], nd_cl_i].set(True, mode="drop")
            el = el.at[bidx, r].add(1)
        # --- append obs ---
        feat = jnp.stack([start.astype(jnp.float64), end.astype(jnp.float64),
                          w_b.astype(jnp.float64), use_cloud.astype(jnp.float64),
                          nodes[:, 0].astype(jnp.float64), h_b.astype(jnp.float64)], axis=1)
        ob = ob.at[bidx, ol].set(feat, mode="drop")
        ol = ol + 1
        # overflow 検出（append 直後に E_max に達したら記録）
        ovf = ovf | (el.max() >= e_alloc) | (ol.max() >= e_obs)

        tm = start
        cost = cost + cost_step
        wsum = wsum + wait
        mksp = jnp.maximum(mksp, end)
        dr = (dr - rew.astype(jnp.float32)).astype(jnp.float32)
        hz = jnp.maximum(hz - 1.0, 1.0)

        # --- compact は kcompact ステップ毎（occ gather が重いので lax.cond で間引く）---
        # E_alloc/E_obs は「飽和値 + kcompact」を覆うので、間の append は溢れない。
        # 死んだイベントが残っていてもカーネルは INF64/-1/occ=False で自然排除=結果不変。
        def do_compact(op):
            es, ee, oon, ocl, ob, ol, tm = op
            st0, en0, oon2, _ = _compact_res(es[:, 0], ee[:, 0], oon, suffnext)
            c0 = (ee[:, 0] >= suffnext).sum(1).astype(jnp.int32)
            if elastic_cloud:
                # cloud は alloc 配列を持たない → resource0 のみ compact
                es2 = es.at[:, 0].set(st0)
                ee2 = ee.at[:, 0].set(en0)
                el2 = el.at[:, 0].set(c0)
                ocl2 = ocl
            else:
                st1, en1, ocl2, _ = _compact_res(es[:, 1], ee[:, 1], ocl, suffnext)
                c1 = (ee[:, 1] >= suffnext).sum(1).astype(jnp.int32)
                es2 = jnp.stack([st0, st1], axis=1)
                ee2 = jnp.stack([en0, en1], axis=1)
                el2 = jnp.stack([c0, c1], axis=1)
            ob2, ol2 = _compact_obs(ob, ol, tm - n_window)
            return es2, ee2, oon2, ocl2, el2, ob2, ol2

        def no_compact(op):
            es, ee, oon, ocl, ob, ol, tm = op
            return es, ee, oon, ocl, el, ob, ol

        es, ee, oon, ocl, el, ob, ol = lax.cond(
            (t + 1) % kcompact == 0, do_compact, no_compact,
            (es, ee, oon, ocl, ob, ol, tm),
        )

        carry = (es, ee, oon, ocl, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf)
        return carry, (use_cloud.astype(jnp.int8), rew)

    xs = (jnp.arange(T, dtype=jnp.int64), jq, arr_c, w_c, h_c, suff_c)
    carry0 = (es, ee, oon, ocl, el, ob, ol, tm, cost, wsum, mksp, dr0, hz0, ovf)
    carry, (actions, rewards) = lax.scan(body, carry0, xs)
    (es, ee, oon, ocl, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf) = carry
    return cost, mksp, wsum, ovf, actions, rewards


def run_fused_compact(jobs, commands, *, params=None, ckpt_path=None, seed=0,
                      n_on=256, n_cl=1024, n_window=100, kernel="v1h",
                      e_alloc=1024, e_obs=2048, kcompact=64, elastic_cloud=False):
    """圧縮融合 rollout（E_alloc/E_obs 固定; 大N可）。返り値は run_fused と同形式 + overflow。"""
    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    B = len(commands)
    if params is None:
        params = load_pcn_params(ckpt_path)
    arr = jobs[:, 0].astype(np.int64)
    w = jobs[:, 1].astype(np.int64)
    h = jobs[:, 2].astype(np.int64)
    suffmin = np.minimum.accumulate(arr[::-1])[::-1]
    # 各ステップ t の compact 閾値 = min(arrival[t+1:])。最終は +inf 相当(全部落としてよい)。
    suffnext = np.empty(T, dtype=np.int64)
    suffnext[:-1] = suffmin[1:]
    suffnext[-1] = np.iinfo(np.int64).max // 4
    h_max = int(h.max())
    jq = jnp.asarray(precompute_job_queue(jobs))
    jobs_cols = (jnp.asarray(arr), jnp.asarray(w), jnp.asarray(h), jnp.asarray(suffnext))
    dr0 = jnp.asarray(np.stack([np.asarray(c[0], np.float32) for c in commands]))
    hz0 = jnp.asarray(np.array([float(c[1]) for c in commands], np.float32))
    key = jax.random.PRNGKey(seed)
    cost, mksp, wsum, ovf, actions, rewards = _fused_compact_run(
        jobs_cols, params, key, jq, dr0, hz0,
        n_on=n_on, n_cl=n_cl, h_max=h_max, n_window=n_window,
        norm_time=float(max(1, n_window)), norm_nodes=float(max(n_on, n_cl)),
        kernel=kernel, e_alloc=int(e_alloc), e_obs=int(e_obs), kcompact=int(kcompact),
        elastic_cloud=bool(elastic_cloud),
    )
    jax.block_until_ready(actions)
    cost = np.asarray(cost, np.float64)
    mksp = np.asarray(mksp, np.float64)
    avg_wait = np.asarray(wsum, np.float64) / T
    return {
        "actions": np.asarray(actions).T,                       # (B,T)
        "rewards": np.asarray(rewards).transpose(1, 0, 2),      # (B,T,2)
        "achieved": np.stack([cost, mksp, avg_wait], axis=1),   # (B,3)
        "overflow": bool(np.asarray(ovf)),
    }


# ---------------------------------------------------------------------------
# レバー1: オンプレ「区間/カウント表現」で N軸(1520)を消す
# ---------------------------------------------------------------------------
# オンプレは非連続配置(cont_only=False)なので、配置可能性は「free_count >= height」だけで
# 決まる。重なるイベントは必ず互いに素なノードを占めるので
#   occupied(t) = Σ_e height_e·[e が t と重なる]   (密occマスク不要)
# = einsum('bte,be->bt') で N軸なしに得られる。free = N_on - occupied。start は最早成立候補。
# → alloc カーネルの O(B·T·E·N) が O(B·T·E) に落ちる(N=1520 を丸ごと除去)。
# cost/wait は配置ノードに不変なので **ビット一致**(obs の start_node のみ 0 プレースホルダ=
# サンプリング摂動のみ、KSで担保)。cloud は弾性(start=arrival)。
from gpu_sweep_jax import _candidates as _cand_fn  # noqa: E402


def _compact_count(es, ee, eh, thr):
    """区間表現(start/end/height)を alive(end>=thr)で前詰め。"""
    E = ee.shape[1]
    alive = ee >= thr
    order = jnp.argsort(~alive, axis=1, stable=True)
    cnt = alive.sum(axis=1).astype(jnp.int32)
    pos_ok = jnp.arange(E)[None, :] < cnt[:, None]
    st = jnp.where(pos_ok, jnp.take_along_axis(es, order, axis=1), INF64)
    en = jnp.where(pos_ok, jnp.take_along_axis(ee, order, axis=1), -1)
    hh = jnp.where(pos_ok, jnp.take_along_axis(eh, order, axis=1), 0)
    return st, en, hh, cnt


@functools.partial(
    jax.jit,
    static_argnames=("n_on", "h_max", "n_window", "norm_time", "norm_nodes",
                     "e_alloc", "e_obs", "kcompact"),
)
def _fused_count_run(jobs_cols, params, key, jq, dr0, hz0, *, n_on, h_max,
                     n_window, norm_time, norm_nodes, e_alloc, e_obs, kcompact):
    """区間/カウント表現の融合 scan(オンプレ N軸なし + 弾性cloud)。"""
    arr_c, w_c, h_c, suff_c = jobs_cols
    B = dr0.shape[0]
    T = arr_c.shape[0]
    bidx = jnp.arange(B)

    es = jnp.full((B, e_alloc), INF64, dtype=jnp.int64)     # onprem イベント start
    ee = jnp.full((B, e_alloc), -1, dtype=jnp.int64)        # end
    eh = jnp.zeros((B, e_alloc), dtype=jnp.int32)           # height(占有ノード数)
    oon = jnp.zeros((B, e_alloc, n_on), dtype=bool)         # 占有ノード(start_node 読取専用)
    el = jnp.zeros((B,), dtype=jnp.int32)                   # onprem 生存数
    n_idx = jnp.arange(n_on, dtype=jnp.int32)
    ob = jnp.zeros((B, e_obs, 6), dtype=jnp.float64)
    ol = jnp.zeros((B,), dtype=jnp.int32)
    tm = jnp.zeros((B,), dtype=jnp.int64)
    cost = jnp.zeros((B,), dtype=jnp.int64)
    wsum = jnp.zeros((B,), dtype=jnp.int64)
    mksp = jnp.zeros((B,), dtype=jnp.int64)
    ovf = jnp.zeros((), dtype=bool)

    def body(carry, xs):
        es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf = carry
        t, jq_t, arr, w, h, suffnext = xs
        # --- オンプレ配置可能性(カウント表現; N軸なし O(B·T·E)) ---
        cand, valid = _cand_fn(es, ee, jnp.full((B,), arr, jnp.int64))   # (B,Tc)
        t_end = cand + w
        ov = (es[:, None, :] < t_end[:, :, None]) & (ee[:, None, :] > cand[:, :, None])
        occupied = jnp.einsum("bte,be->bt", ov.astype(jnp.int32), eh)     # (B,Tc)
        free_cnt = n_on - occupied
        feasible = (free_cnt >= h) & valid
        t_idx = jnp.argmax(feasible, axis=1)
        found = feasible.any(axis=1)
        fb = jnp.maximum(arr, jnp.max(ee, axis=1))
        s_on = jnp.where(found, jnp.take_along_axis(cand, t_idx[:, None], 1)[:, 0], fb)
        # --- start_node は選ばれた候補 t_idx でだけ per-node 占有を作り実値を読む(O(B·E·N)) ---
        ov_at = (es < (s_on + w)[:, None]) & (ee > s_on[:, None])         # (B,E) 重なるイベント
        per_node = jnp.einsum("be,ben->bn", ov_at.astype(jnp.int32),
                              oon.astype(jnp.int32))                       # (B,n_on)
        free_node = per_node == 0
        # 非連続=最下位 free から height 個。start_node = 最下位 free。
        sort_key = jnp.where(free_node, n_idx[None, :], n_on + n_idx[None, :])
        nodes_sorted = jnp.sort(sort_key, axis=1)                         # (B,n_on)
        sel_nodes = nodes_sorted[:, :h_max]                               # (B,h_max)
        occ_at = sel_nodes[:, 0]                                          # 最下位 free
        pw = jnp.maximum(0, s_on - arr)
        # --- obs → policy → sample ---
        obs = _obs_from_buffer(ob, ol, tm, jq_t, pw, n_window=n_window,
                               norm_time=norm_time, norm_nodes=norm_nodes)
        p = pcn_p_cloud(params, obs, dr, hz)
        u = jax.random.uniform(jax.random.fold_in(key, t), (B,))
        use_cloud = u < p
        start = jnp.where(use_cloud, arr, s_on)          # cloud 弾性=arr
        end = start + w
        wait = start - arr
        cost_step = jnp.where(use_cloud, w * h, 0).astype(jnp.int64)
        rew = jnp.stack([-wait, -cost_step], axis=1).astype(jnp.float64)
        # --- append onprem(cloud はスキップ) ---
        is_on = ~use_cloud
        slot = jnp.where(is_on, el, e_alloc)             # cloud→drop
        es = es.at[bidx, slot].set(start, mode="drop")
        ee = ee.at[bidx, slot].set(end, mode="drop")
        eh = eh.at[bidx, slot].set(jnp.int32(h), mode="drop")
        # 選んだ height 個のノードを占有記録（onprem のみ; cloud/範囲外は drop）
        karr = jnp.arange(h_max, dtype=jnp.int32)
        valid_k = karr[None, :] < jnp.int32(h)
        nd_i = jnp.where(valid_k & is_on[:, None], sel_nodes, n_on)
        oon = oon.at[bidx[:, None], slot[:, None], nd_i].set(True, mode="drop")
        el = el + is_on.astype(jnp.int32)
        # --- append obs(全ジョブ; onprem=実 start_node, cloud=0) ---
        snode = jnp.where(use_cloud, 0, occ_at).astype(jnp.float64)
        feat = jnp.stack([start.astype(jnp.float64), end.astype(jnp.float64),
                          jnp.full((B,), w, jnp.float64), use_cloud.astype(jnp.float64),
                          snode, jnp.full((B,), h, jnp.float64)], axis=1)
        ob = ob.at[bidx, ol].set(feat, mode="drop")
        ol = ol + 1
        ovf = ovf | (el.max() >= e_alloc) | (ol.max() >= e_obs)
        tm = start
        cost = cost + cost_step
        wsum = wsum + wait
        mksp = jnp.maximum(mksp, end)
        dr = (dr - rew.astype(jnp.float32)).astype(jnp.float32)
        hz = jnp.maximum(hz - 1.0, 1.0)

        def do_compact(op):
            es, ee, eh, oon, ob, ol, tm = op
            st, en, hh, c = _compact_count(es, ee, eh, suffnext)
            E = ee.shape[1]
            order = jnp.argsort(~(ee >= suffnext), axis=1, stable=True)
            pos_ok = jnp.arange(E)[None, :] < c[:, None]
            oon2 = jnp.take_along_axis(oon, order[:, :, None], axis=1) & pos_ok[:, :, None]
            ob2, ol2 = _compact_obs(ob, ol, tm - n_window)
            return st, en, hh, oon2, c, ob2, ol2

        def no_compact(op):
            es, ee, eh, oon, ob, ol, tm = op
            return es, ee, eh, oon, el, ob, ol

        es, ee, eh, oon, el, ob, ol = lax.cond(
            (t + 1) % kcompact == 0, do_compact, no_compact,
            (es, ee, eh, oon, ob, ol, tm))

        carry = (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf)
        return carry, (use_cloud.astype(jnp.int8), rew)

    xs = (jnp.arange(T, dtype=jnp.int64), jq, arr_c, w_c, h_c, suff_c)
    carry0 = (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr0, hz0, ovf)
    carry, (actions, rewards) = lax.scan(body, carry0, xs)
    (es, ee, eh, oon, el, ob, ol, tm, cost, wsum, mksp, dr, hz, ovf) = carry
    return cost, mksp, wsum, ovf, actions, rewards


def run_fused_count(jobs, commands, *, params=None, ckpt_path=None, seed=0,
                    n_on=1520, n_window=100, e_alloc=512, e_obs=1152, kcompact=64):
    """レバー1: オンプレ区間/カウント表現 + 弾性cloud の融合 rollout(N軸なし)。"""
    jobs = np.asarray(jobs, dtype=np.float64)
    T = jobs.shape[0]
    if params is None:
        params = load_pcn_params(ckpt_path)
    arr = jobs[:, 0].astype(np.int64)
    w = jobs[:, 1].astype(np.int64)
    h = jobs[:, 2].astype(np.int64)
    suffmin = np.minimum.accumulate(arr[::-1])[::-1]
    suffnext = np.empty(T, dtype=np.int64)
    suffnext[:-1] = suffmin[1:]
    suffnext[-1] = np.iinfo(np.int64).max // 4
    h_max = int(h.max())
    jq = jnp.asarray(precompute_job_queue(jobs))
    jobs_cols = (jnp.asarray(arr), jnp.asarray(w), jnp.asarray(h), jnp.asarray(suffnext))
    dr0 = jnp.asarray(np.stack([np.asarray(c[0], np.float32) for c in commands]))
    hz0 = jnp.asarray(np.array([float(c[1]) for c in commands], np.float32))
    key = jax.random.PRNGKey(seed)
    cost, mksp, wsum, ovf, actions, rewards = _fused_count_run(
        jobs_cols, params, key, jq, dr0, hz0, n_on=n_on, h_max=h_max, n_window=n_window,
        norm_time=float(max(1, n_window)), norm_nodes=float(n_on),
        e_alloc=int(e_alloc), e_obs=int(e_obs), kcompact=int(kcompact))
    jax.block_until_ready(actions)
    return {
        "actions": np.asarray(actions).T,
        "rewards": np.asarray(rewards).transpose(1, 0, 2),
        "achieved": np.stack([np.asarray(cost, np.float64), np.asarray(mksp, np.float64),
                              np.asarray(wsum, np.float64) / T], axis=1),
        "overflow": bool(np.asarray(ovf)),
    }
