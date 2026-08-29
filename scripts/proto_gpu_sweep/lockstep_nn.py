"""lockstep_nn — B-3: ロックステップGPU rollout への torch NN 方策(greedy)接続。

設計書: scripts/proto_gpu_sweep/lockstep_design.md B-3。
リファレンス: src/agents/pcn_agent.py の _act(eval_mode=True) と _run_episode(eval_mode=True)。

構成(1 step = obs_kernel 1 + torch NN forward + step_kernel 1 + torch 指令更新):
  1. obs_kernel(j): 全 B エピソードの現在観測 obs[:, j, :224] を GPU 構築(B-2 のカーネル)。
  2. NN act: obs[B,224] + 指令(desired_return[B,2], horizon[B,1]) を model forward し
     行動 = argmax(scores, dim=1)(greedy, pcn_agent._act eval_mode=True と同一規則)。
     TorchScript trace(PCN_JIT_ACT と同じ th.jit.trace、ビット一致・Python層除去)を既定使用。
  3. step_kernel(j): 選んだ行動で j 番目のジョブを配置(B-1 のカーネル)。
  4. 指令更新(_run_episode の写し、B本ベクトル化):
       reward = [-wait_j, -cost_j] (f64)
       desired_return = float32(float64(desired_return) - reward)   # CPU の
         (desired_return - reward).astype(np.float32) と同一(f64減算→f32丸め)
       PCN_COST_HOLD=1 なら desired_return[:,1] を初期値へ固定(同 env var を読む)
       desired_horizon = clamp(desired_horizon - 1, min=1.0) (f32; 全ジョブ scheduled=毎step減算)
     eval_mode=True なので max_return クリップは行わない(_run_episode と同一)。

zero-copy 接続: obs/actions/waits/costs は torch cuda tensor として確保し、
numba.cuda.as_cuda_array()(__cuda_array_interface__ 経由・同一ポインタ確認済み)の
ビューを numba カーネルへ渡す。numba カーネル(legacy default stream)と torch の
既定 current stream(同じく legacy default stream)は同一ストリームで直列化されるため、
ループ内にホスト同期は一切不要(全 step 非同期発行→最後に synchronize 1回)。

非対応(このプロトタイプのスコープ外):
  - sample(温度あり)行動。greedy のみ。
  - anchor residual(PCN_ANCHOR_RESIDUAL)・SCHEDULER_OBS_BUDGET_RATIO。
  - defer(action=2)。argmax が 2 を返し得る action_dim>=3 の checkpoint は不可(assert)。

必須環境変数(このモジュールは読むだけ。呼び出し側が pcn_agent import 前に設定):
  観測: SCHEDULER_OBS_URGENCY=1 SCHEDULER_OBS_EFFICIENCY=1
        DISTRIBUTED_PCN_USE_EVENT_OBS=1 SCHEDULER_LEARNER_BITMAP=0
  モデル(run_j50000_gpu_v5 学習時条件, eval_jscale_c3.sh 準拠):
        PCN_FOURIER_CMD=1 PCN_FC_DEPTH=4 PCN_COND_ADD_SCALE=0.25
        PCN_COMMAND_BALANCE=1 PCN_OBS_LOG=1
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import torch as th
from numba import cuda

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# [PCN_LOCKSTEP_BLOCK] 1ブロック=1エピソード版のカーネルを使う(既定ON)。1本の中で
# 占有カウント更新と空きノード探索を手分けする。5万・実trace weekB で 1スレッド版と
# (start,wait,cost) も観測も全件一致を確認済み(verify_lockstep_block.py)。
# 実測(B=2, cand_m=2048): 10.057 → 1.082 ms/step。=0 で従来の 1スレッド版に戻る。
_LOCKSTEP_BLOCK = os.environ.get("PCN_LOCKSTEP_BLOCK", "1") == "1"
if _LOCKSTEP_BLOCK:
    from lockstep_kernel_block import (  # noqa: E402
        OBS_TOTAL_DIM, N_EVENTS_OBS, TPB as _LOCKSTEP_TPB, _get_obs_kernel,
        _get_step_kernel, alloc_state,
    )
else:
    from lockstep_kernel import (  # noqa: E402
        OBS_TOTAL_DIM, N_EVENTS_OBS, _get_obs_kernel, _get_step_kernel, alloc_state,
    )

    _LOCKSTEP_TPB = 1

# _run_episode の PCN_COST_HOLD gate と同一(pcn_agent._COST_HOLD の写し)。
_COST_HOLD = os.environ.get("PCN_COST_HOLD", "0") == "1"


def build_policy_model(sd: dict, n_jobs: int, device: str = "cuda",
                       hidden_dim: int = 512):
    """state_dict から DiscreteActionsDefaultModel を構築・ロードして返す。

    アーキテクチャを決める環境変数(PCN_FOURIER_CMD/PCN_FC_DEPTH等)は import 時に
    焼き込まれるため、呼び出し側がこの関数より前に設定しておくこと。
    scaling_factor / desired_return_center|scale / command_balance / fourier_freqs は
    すべて state_dict に含まれておりロードで上書きされる(eval_b2_compare と同じ)。
    """
    from src.agents.pcn_agent import DiscreteActionsDefaultModel  # env var 焼き込み後に import

    state_dim = int(sd["s_emb.0.weight"].shape[1])
    fc_idx = max(int(m.group(1)) for k in sd
                 if (m := re.match(r"fc\.(\d+)\.weight$", k)))
    action_dim = int(sd[f"fc.{fc_idx}.weight"].shape[0])
    model = DiscreteActionsDefaultModel(
        state_dim, action_dim, 2,
        np.array([1.0, 1.0, 1.0 / max(1, n_jobs)], dtype=np.float32),
        hidden_dim=hidden_dim,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint missing keys: {missing}")
    model.eval()
    return model.to(device)


def load_policy_model(ckpt_path: str, n_jobs: int, device: str = "cuda",
                      hidden_dim: int = 512):
    """checkpoint ファイルから build_policy_model する薄いラッパ。"""
    st = th.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    return build_policy_model(sd, n_jobs, device=device, hidden_dim=hidden_dim)


def run_lockstep_greedy(
    jobs, model, commands, n_on: int, n_cl: int, *,
    n_window: int, horizons=None,
    e_max: int = 8192, k: int = 16, tpb: int = 1,
    device: str = "cuda", use_jit: bool = True,
    return_obs: bool = False, timing: dict | None = None,
    progress: int = 0, mode: str = "greedy", sample_seed: int = 0,
) -> dict:
    """指令付き greedy rollout を GPU 完結(ロックステップ)で B 本実行する。

    Args:
        jobs: (T,8) raw形式ジョブ列(run_raw_rollout と同一)。
        model: load_policy_model の戻り値(または同シグネチャの nn.Module, device 上)。
        commands: (B,2) float32 desired_return(objectives_to_command の [-wait*nj, -cost] 規約)。
        n_on, n_cl: ノード数。n_window: 観測正規化窓(env と一致させること)。
        horizons: (B,) desired_horizon 初期値。None なら全て float32(T)
                  (eval_b2_compare._ep の np.float32(nj) と同一)。
        return_obs: True なら記録済み obs[B,T,224] (torch.float32 cuda tensor) を返す
                    (B-4 の学習教材用。リプレイ不要)。
        timing: dict を渡すと 'total_s', 'per_step_us' 等の計測値を書き込む。
        progress: 0 より大きいとき、その step 数ごとに経過を表示(発行ベース)。
        mode: "greedy"=argmax(既定) / "sample"=probs=exp(log_softmax) から multinomial
              抽選(pcn_agent._act eval_mode=False の分布と同一。乱数は torch cuda
              Generator(sample_seed) で決定論=CPU 側とはビット別・分布一致が要件)。
        sample_seed: mode="sample" の Generator シード。

    Returns:
        dict(actions: (B,T) int8, start_times/waits/costs: (B,T) int64,
             node_start: (B,T) int32, objectives: (B,2) float64 = (total_cost, mean_wait),
             ovf: (B,) int32, peak_ev_on/peak_ev_cl: (B,) int32,
             final_desired_return: (B,2) float32,
             obs: torch cuda tensor (B,T,224) float32 or None)
    """
    jobs = np.ascontiguousarray(jobs, dtype=np.float64)
    commands = np.ascontiguousarray(commands, dtype=np.float32)
    T = int(jobs.shape[0])
    B = int(commands.shape[0])
    _adim = int(getattr(model, "action_dim", 2))
    if _adim > 2:
        raise ValueError(f"action_dim={_adim}: defer(action=2)はロックステップ非対応")
    if mode not in ("greedy", "sample"):
        raise ValueError(f"mode={mode!r} (greedy/sample のみ)")
    gen = None
    if mode == "sample":
        gen = th.Generator(device=device)
        gen.manual_seed(int(sample_seed) & 0x7FFFFFFFFFFFFFFF)

    arrivals = jobs[:, 0].astype(np.float64)
    suffix_min = np.minimum.accumulate(arrivals[::-1])[::-1].copy()

    step_kernel = _get_step_kernel()
    obs_kernel = _get_obs_kernel()

    # --- GPU 常駐状態(B-1)と、torch 側とゼロコピー共有する I/O バッファ ---
    # obs は torch f32 tensor に直接書く(f64計算→f32格納の単一丸め=astype(f32)と同一)ため、
    # alloc_state の f64 obs バッファは確保しない(collect_obs=False)。obs用の補助バッファ
    # (buf/cand/last_ins)だけ個別に確保する。
    s = alloc_state(B, T, n_on, n_cl, e_max=e_max, k=k, collect_obs=False)
    d_jobs = cuda.to_device(jobs)
    d_suffix = cuda.to_device(suffix_min)
    d_buf_start = cuda.to_device(np.zeros((B, N_EVENTS_OBS), dtype=np.int64))
    d_buf_t = cuda.to_device(np.zeros((B, N_EVENTS_OBS), dtype=np.int32))
    # [PCN_LOCKSTEP_CAND_M] 64 をハードコードしていたが、lockstep_kernel_block.py:36-41 の実測表では
    # 64 が最悪値(3.308 ms/step)で 2048 の 3.1倍遅い。容量は正確性に無関係(全件一致確認済み)。
    # 小さいと生存30件が埋まらず、tid==0 の1スレッドだけで過去イベント全走査 O(j) に落ちる。
    # 自前確保していたため環境変数も効いていなかった。既定値に合わせる(2026-08-28)。
    cand_m = int(os.environ.get("PCN_LOCKSTEP_CAND_M", "2048"))
    d_cand_start = cuda.to_device(np.zeros((B, cand_m), dtype=np.int64))
    d_cand_t = cuda.to_device(np.zeros((B, cand_m), dtype=np.int32))
    d_cand_n = cuda.to_device(np.zeros((B,), dtype=np.int32))
    d_last_ins = cuda.to_device(np.full((B,), -1, dtype=np.int32))

    obs_th = th.zeros((B, T, OBS_TOTAL_DIM), dtype=th.float32, device=device)
    act_th = th.zeros((B, T), dtype=th.int8, device=device)
    waits_th = th.zeros((B, T), dtype=th.int64, device=device)
    costs_th = th.zeros((B, T), dtype=th.int64, device=device)
    d_obs = cuda.as_cuda_array(obs_th)      # 同一ポインタのビュー(コピーなし)
    d_act = cuda.as_cuda_array(act_th)
    d_waits = cuda.as_cuda_array(waits_th)
    d_costs = cuda.as_cuda_array(costs_th)

    # --- 指令テンソル(B本ベクトル化) ---
    dr = th.as_tensor(commands, dtype=th.float32, device=device)          # [B,2]
    if horizons is None:
        h = th.full((B, 1), float(np.float32(T)), dtype=th.float32, device=device)
    else:
        h = th.as_tensor(np.asarray(horizons, dtype=np.float32),
                         device=device).reshape(B, 1)
    hold_target = dr[:, 1].clone() if _COST_HOLD else None

    norm_time = float(max(1, n_window))
    norm_nodes = float(max(1, max(n_on, n_cl)))
    if _LOCKSTEP_BLOCK:
        # 1ブロック=1エピソード。tpb 引数(1スレッド版の名残)は無視する。
        threads_per_block = _LOCKSTEP_TPB
        blocks = B
    else:
        threads_per_block = max(1, int(tpb))
        blocks = (B + threads_per_block - 1) // threads_per_block

    def _launch_obs(j: int) -> None:
        obs_kernel[blocks, threads_per_block](
            d_jobs, d_act, d_suffix, j, B, T, n_on, n_cl,
            float(n_window), norm_time, norm_nodes,
            s.start_times, s.node_start, s.time_state,
            s.ev_on_start, s.ev_on_end, s.ev_on_nfrag, s.ev_on_frag_lo, s.ev_on_frag_hi,
            s.order_end_on, s.order_start_on, s.counted_on, s.count_on,
            s.n_ev_on, s.prune_at_on,
            s.ev_cl_start, s.ev_cl_end, s.ev_cl_nfrag, s.ev_cl_frag_lo, s.ev_cl_frag_hi,
            s.order_end_cl, s.order_start_cl, s.counted_cl, s.count_cl,
            s.n_ev_cl, s.prune_at_cl,
            s.remap, s.pick_lo, s.pick_hi,
            d_buf_start, d_buf_t,
            d_cand_start, d_cand_t, d_cand_n, d_last_ins,
            s.ureuse_j, s.ureuse_start, s.ureuse_nfrag, s.ureuse_lo, s.ureuse_hi,
            s.creuse_j, s.creuse_start, s.creuse_lo0, s.creuse_hi0,
            s.ovf_flag, d_obs,
        )

    def _launch_step(j: int) -> None:
        step_kernel[blocks, threads_per_block](
            d_jobs, d_act, d_suffix, j, B, n_on, n_cl, e_max,
            s.ev_on_start, s.ev_on_end, s.ev_on_nfrag, s.ev_on_frag_lo, s.ev_on_frag_hi,
            s.order_end_on, s.order_start_on, s.counted_on, s.count_on,
            s.ev_cl_start, s.ev_cl_end, s.ev_cl_nfrag, s.ev_cl_frag_lo, s.ev_cl_frag_hi,
            s.order_end_cl, s.order_start_cl, s.counted_cl, s.count_cl,
            s.n_ev_on, s.n_ev_cl, s.prune_at_on, s.prune_at_cl,
            s.remap, s.pick_lo, s.pick_hi,
            s.ureuse_j, s.ureuse_start, s.ureuse_nfrag, s.ureuse_lo, s.ureuse_hi,
            s.creuse_j, s.creuse_start, s.creuse_lo0, s.creuse_hi0,
            s.start_times, d_waits, d_costs, s.node_start, s.time_state,
            s.ovf_flag, s.peak_ev_on, s.peak_ev_cl,
        )

    # --- NN forward: PCN_JIT_ACT と同じ th.jit.trace(ビット一致・重み参照保持) ---
    # [2026-08-30 R5] 観測は kernel が常に全幅(224: 末尾3=efficiency)で構築するが、
    # 観測221次元の checkpoint(SCHEDULER_OBS_EFFICIENCY=0 で学習)は先頭221だけを読む。
    # efficiency は「末尾に追加」の規約で、先頭221次元は ON/OFF でビット一致(08-30検証)
    # なのでスライスは正確。全幅 checkpoint は従来どおり=ビット不変。
    sdim = int(model.s_emb[0].weight.shape[1]) if hasattr(model, "s_emb") else int(obs_th.shape[2])
    assert sdim in (int(obs_th.shape[2]), int(obs_th.shape[2]) - 3), \
        f"未対応の観測次元: model={sdim} obs={int(obs_th.shape[2])}"
    policy = model
    if use_jit:
        _launch_obs(0)  # trace 用の実入力(実 step0 観測)を先に作る
        with th.no_grad():
            policy = th.jit.trace(model, (obs_th[:, 0, :sdim], dr, h))
        # trace 実行は状態を変えない(forward は純関数)。obs[:,0,:] は step0 で再構築される。

    t0 = time.time()
    with th.no_grad():
        for j in range(T):
            _launch_obs(j)
            scores = policy(obs_th[:, j, :sdim], dr, h)
            if gen is None:
                act = th.argmax(scores, dim=1)
            else:
                # pcn_agent._act(eval_mode=False): probs=exp(scores) → multinomial。
                act = th.multinomial(th.exp(scores), 1, generator=gen).squeeze(1)
            act_th[:, j] = act.to(th.int8)
            _launch_step(j)
            # --- 指令更新(_run_episode の写し, f64減算→f32丸めで CPU と同一) ---
            w64 = waits_th[:, j].to(th.float64)
            c64 = costs_th[:, j].to(th.float64)
            reward = th.stack((-w64, -c64), dim=1)          # [B,2] f64
            dr = (dr.double() - reward).float()
            if hold_target is not None:
                dr[:, 1] = hold_target
            h = th.clamp(h - 1.0, min=1.0)
            if progress and (j + 1) % progress == 0:
                th.cuda.synchronize()
                print(f"  [lockstep-nn] {j+1}/{T} steps "
                      f"({(time.time()-t0)/(j+1)*1e6:.0f}us/step)", flush=True)
    th.cuda.synchronize()
    cuda.synchronize()
    total_s = time.time() - t0
    if timing is not None:
        timing["total_s"] = total_s
        timing["per_step_us"] = total_s / max(1, T) * 1e6

    actions = act_th.cpu().numpy()
    start_times = s.start_times.copy_to_host()
    waits = waits_th.cpu().numpy()
    costs = costs_th.cpu().numpy()
    node_start = s.node_start.copy_to_host()
    ovf = s.ovf_flag.copy_to_host()
    peak_ev_on = s.peak_ev_on.copy_to_host()
    peak_ev_cl = s.peak_ev_cl.copy_to_host()

    total_cost = costs.sum(axis=1).astype(np.float64)
    mean_wait = waits.mean(axis=1).astype(np.float64)
    objectives = np.stack([total_cost, mean_wait], axis=1)
    # makespan = max(end) = max(start+width)(env.calc_objective_values と同一)。
    widths = jobs[:, 1].astype(np.int64)
    makespan = (start_times + widths[None, :]).max(axis=1)

    return dict(
        actions=actions, start_times=start_times, waits=waits, costs=costs,
        node_start=node_start, objectives=objectives, makespan=makespan, ovf=ovf,
        peak_ev_on=peak_ev_on, peak_ev_cl=peak_ev_cl,
        final_desired_return=dr.cpu().numpy(),
        obs=obs_th if return_obs else None,
    )
