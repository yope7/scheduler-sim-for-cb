"""[PCN_GPU_FACTORY] GPU rollout データ工場の Ray Actor 本体。

scripts/proto_gpu_sweep/factory_jax.py（Phase 3/3.5 で検証済みのバッチ env +
バッチ方策サンプリング）を、分散 PCN 学習ループ（distributed_pcn.train）から
使える Ray Actor に包む。差し込みは distributed_pcn.py の [PCN_GPU_FACTORY]
マーカー箇所のみ・既定 OFF（PCN_GPU_FACTORY=1 で有効化）。

役割の分担:
  - Phase 1 ランダム収集（run_random）: Bernoulli(p) スイープを GPU バッチ生産。
    ヒューリスティック種まき（wtth / giant-defer / NSGA）は従来 Actor のまま。
  - Phase 3 指令付き rollout（run_commands）: pre_fetched_commands 互換の
    (desired_return, horizon[, base_return]) リストを受け、サンプリング rollout を
    量産して ReplayBuffer.add_batch へ直接積む（既存 Actor.run と同じ流儀）。
    command_outcomes / episode_summaries も既存 Actor と同形式で返す。

重み同期: 既存 Actor.run の冒頭（distributed_pcn.py:1092-1097）と同じ順序点 =
run_commands 冒頭で Learner actor キューに get_weights_ref を積んで受ける
（learner.learn 完了後に update_weights_ref された共有 ObjectRef）。メインループ側
で ray.get すると async オーバーラップがブロックするため、取得は工場プロセス内で行う。

失敗時: run_* は例外を内部で捕捉して失敗 dict（episodes_generated=0,
_factory_failed=メッセージ）を返す＝学習ループは止まらない。連続 2 回失敗で
healthy()=False になり、以後の波は従来 Actor 経路に戻る（distributed_pcn 側で判定）。

対応済み機能（CPU 実装と機能等価; 2026-08 監査で gate 解除・検証済み）:
  - defer（3 行動, SCHEDULER_ALLOW_DEFER=1）: factory_defer（フル per-node env=有限cloud・
    ジョブ列回転・可変長）。配列フロー必須（PCN_GPU_FACTORY_ARRAY=0 との併用は不可=gate）。
  - defer無し（2 行動）: PCN_GPU_COUNT_KERNEL=ev（新既定, 2026-08-06）で factory_defer_ev
    を defer_max=0 で回す=イベント駆動の厳密カーネル（待ちも torch env と一致）。
    旧 count 近似（onprem ノード共有非計上 + cloud 弾性化; 過密で待ちが真値から乖離）へは
    PCN_GPU_COUNT_KERNEL=count で退避可。溢れフラグ発火時は種別バッファを拡大して自動再実行。
  - forward 系: PCN_FC_DEPTH 任意 / PCN_OBS_LOG 0,1 / PCN_FILM / PCN_FOURIER_MODE 全種
    （gaussian は ckpt の fourier_freqs buffer を使い torch と同一行列）/
    PCN_COMMAND_BALANCE / PCN_COND_WAIT_ROBUST / PCN_COND_ADD_SCALE。
  - rollout 指令更新: PCN_COST_HOLD / PCN_DESIRED_RETURN_UB（actor :1579-1584 と同一）。
  - env/観測: SCHEDULER_WAIT_METRIC=slowdown / SCHEDULER_OBS_OCCUPANCY
    (+SCHEDULER_OBS_OCC_PRIOR) / SCHEDULER_ORACLE_NPZ（obs 末尾に urgency→occupancy→
    oracle の順で付加; event_c_env.get_observation と同順）。
  - PCN_MIX_REGIMES: defer 工場=エピソード別ジョブ配列 / count 工場=レジーム別バッチ分割
    （分布一致）。学習中 eval はリスト先頭=基準レジーム固定（actor :1681 と同じ）。
  - PCN_MIX_JOBS（N 混合学習）: N 別バッチ分割 + horizon=max(1,N-2) 上書き
    （actor :1473-1474）。学習中 eval は基準N（=max, actor evaluate_episode と同じ）。

既知の制限（ping() が検査し、非対応構成では起動自体を諦めて従来経路にする）:
  - PCN_ANCHOR_SET（アンカー残差; 行動意味論が変わる・使用実績なし）
  - SCHEDULER_LEARNER_BITMAP=1 / DISTRIBUTED_PCN_USE_EVENT_OBS!=1 / PCN_SCALEFREE_ENV=1
    （旧 bitmap 観測経路・別 env。工場は event-native 観測専用）
  - SCHEDULER_OBS_JQ_NORM=1（jq の時刻依存正規化）/ PCN_DIMLESS_NORM=1（報酬無次元化の
    スケール計測プロトコル）

追加対応（2026-08）: PCN_ARCH=attn（_AttnStateEncoder MHA2層の JAX 移植）/
PCN_S_EMB_DROPOUT（CPU actor は train モードのまま rollout/eval するため dropout が
全 forward に乗る=工場も per-step Bernoulli マスクで同分布再現）。
  - EnhancedModel は env 非制御の定数 False（distributed_pcn.py:207）のため gate 不要。
  - PCN_REUSE_URGENCY_ALLOC は env 内部キャッシュの再利用（0/1 どちらでもビット一致・
    rollout 出力に影響なし）のため gate 不要。
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Sequence

import numpy as np

# [PCN_GPU_RAW_BLOCK] rawカーネルを「1ブロック=1エピソード(既定128スレッド)」の
# ブロック協調版に切り替える(既定ON)。1本の中で占有カウント更新と空きノード探索を手分けし、
# 5万・実trace weekB で 6.898→0.337 ms/step(20.5倍)、(start,wait,cost) は全件一致。
# =0 で従来の 1スレッド=1エピソード版に戻る。
_RAW_BLOCK = os.environ.get("PCN_GPU_RAW_BLOCK", "1") == "1"
_RAW_BLOCK_TPB = int(os.environ.get("PCN_GPU_RAW_BLOCK_TPB", "128"))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PROTO_DIR = os.path.join(_REPO_ROOT, "scripts", "proto_gpu_sweep")


def _mix_regimes() -> List[float]:
    """PCN_MIX_REGIMES のパース（actor 側 distributed_pcn._MIX_REGIMES と同一規則）。"""
    return [
        float(x)
        for x in os.environ.get("PCN_MIX_REGIMES", "").replace(" ", "").split(",")
        if x
    ]


def _eval_regime():
    """PCN_EVAL_REGIME のパース（actor 側 distributed_pcn._EVAL_REGIME_VAL と同一規則）。

    学習中 eval(=best_model 選抜)で使うレジーム。未設定なら None＝従来どおり
    _mix_regimes()[0](リスト先頭)を使う＝完全にビット不変。
    """
    _v = os.environ.get("PCN_EVAL_REGIME", "").strip()
    return float(_v) if _v else None


def _mix_jobs() -> List[int]:
    """PCN_MIX_JOBS のパース（actor 側 distributed_pcn._MIX_JOBS と同一規則: sorted set）。"""
    return sorted({
        int(x)
        for x in os.environ.get("PCN_MIX_JOBS", "").replace(" ", "").split(",")
        if x
    })


def _unsupported_flags() -> List[str]:
    """工場が非対応の構成フラグ（有効ならその名前を返す）。

    方針（gate 監査 2026-08）: 学習レシピ系 env var は「工場が再現する」か「ここで明示
    ブロックして CPU フォールバック」の二択のみ。サイレント無視は分布ずれ（例: regmix
    無視による条件付け崩壊）の温床なので禁止。

    対応済み（gate 解除・工場が同挙動で再現; 検証 2026-08）:
      PCN_FC_DEPTH 任意 / PCN_OBS_LOG 0,1 / PCN_COST_HOLD / PCN_DESIRED_RETURN_UB /
      PCN_MIX_REGIMES(count 工場含む) / PCN_MIX_JOBS(N混合) / PCN_FILM / PCN_ARCH=attn /
      PCN_S_EMB_DROPOUT / PCN_FOURIER_MODE=gaussian / PCN_FOURIER_BANDS_COST(costチャネル
      別バンド数; factory_fused.pcn_logits per-channel 分岐+fourier_freqs_cost buffer 抽出) /
      SCHEDULER_WAIT_METRIC=slowdown /
      SCHEDULER_OBS_OCCUPANCY(+SCHEDULER_OBS_OCC_PRIOR) / SCHEDULER_ORACLE_NPZ /
      SCHEDULER_OBS_EFFICIENCY(ev/dense カーネルのみ; count は下で明示ブロック) /
      SCHEDULER_OBS_BUDGET_RATIO(ev カーネルのみ; defer/count は下で明示ブロック) /
      PCN_COMMAND_BALANCE / PCN_COND_WAIT_ROBUST / PCN_COND_ADD_SCALE。
    """
    bad = []
    # [SCHEDULER_OBS_EFFICIENCY] 効率観測は s_cl(クラウド予測開始時刻)が必要。
    # 旧 count 工場(PCN_GPU_COUNT_KERNEL=count)は cloud を弾性近似(常に arrival 開始)して
    # おり s_cl を厳密に持たない → サイレントに別分布を学習させないよう明示ブロック。
    # 既定 (PCN_GPU_COUNT_KERNEL=ev / defer=ev,dense) は factory 側で同一式を再現済み。
    if (os.environ.get("SCHEDULER_OBS_EFFICIENCY", "0") == "1"
            and os.environ.get("SCHEDULER_ALLOW_DEFER", "0") != "1"
            and (os.environ.get("PCN_GPU_COUNT_KERNEL", "ev").strip() or "ev") == "count"):
        bad.append(
            "SCHEDULER_OBS_EFFICIENCY=1 + PCN_GPU_COUNT_KERNEL=count "
            "(旧count工場は弾性cloud近似で s_cl を持たない; 既定の ev カーネルを使うこと)")
    # [SCHEDULER_OBS_BUDGET_RATIO] 現状は defer 無し ev カーネル(factory_defer_ev.py)のみ
    # 実装済み。dense defer / 旧 count カーネルは未対応のため明示ブロック(サイレント不一致防止)。
    if os.environ.get("SCHEDULER_OBS_BUDGET_RATIO", "0") == "1":
        if os.environ.get("SCHEDULER_ALLOW_DEFER", "0") == "1":
            bad.append(
                "SCHEDULER_OBS_BUDGET_RATIO=1 + SCHEDULER_ALLOW_DEFER=1 "
                "(defer工場は未対応; ev backend のみ実装済み)")
        elif (os.environ.get("PCN_GPU_COUNT_KERNEL", "ev").strip() or "ev") == "count":
            bad.append(
                "SCHEDULER_OBS_BUDGET_RATIO=1 + PCN_GPU_COUNT_KERNEL=count "
                "(旧count工場は未対応; 既定の ev カーネルを使うこと)")
    # defer(SCHEDULER_ALLOW_DEFER=1, 3行動)は factory_defer(フル per-node env +
    # ジョブ列回転)で対応済み。可変長エピソードのため配列フロー必須。
    if (os.environ.get("SCHEDULER_ALLOW_DEFER", "0") == "1"
            and os.environ.get("PCN_GPU_FACTORY_ARRAY", "1") != "1"):
        bad.append("SCHEDULER_ALLOW_DEFER=1 + PCN_GPU_FACTORY_ARRAY=0 (defer は配列フロー必須)")
    if os.environ.get("SCHEDULER_WAIT_METRIC", "wait") not in ("wait", "slowdown"):
        bad.append(f"SCHEDULER_WAIT_METRIC={os.environ.get('SCHEDULER_WAIT_METRIC')} (wait/slowdown のみ)")
    # SCHEDULER_LEARNER_BITMAP=1: Learner が生イベント観測を bitmap へ復元する別観測経路
    # (旧 bitmap 系; 現行レシピは event-native 観測で不使用)。工場 obs は event 形式のみ。
    if os.environ.get("SCHEDULER_LEARNER_BITMAP", "0") == "1":
        bad.append("SCHEDULER_LEARNER_BITMAP=1 (bitmap 復元; 旧観測経路のため gate 維持)")
    # DISTRIBUTED_PCN_USE_EVENT_OBS!=1: bitmap 観測 env(旧経路)。工場は event-native 専用。
    if os.environ.get("DISTRIBUTED_PCN_USE_EVENT_OBS", "0") != "1":
        bad.append("DISTRIBUTED_PCN_USE_EVENT_OBS!=1 (工場は event-native 観測専用; 旧 bitmap 経路は gate 維持)")
    # PCN_ANCHOR_SET: アンカー残差方策(行動を anchor 遺伝子との XOR 残差で表現 + 達成値
    # による事後リアンカー)。rollout の行動意味論そのものが変わる大掛かりな機構で、
    # 現行レシピ(weekA 系)では不使用のため gate 維持(対応する場合は _run_episode の
    # XOR/select_by_values 一式の移植が必要)。
    if os.environ.get("PCN_ANCHOR_SET", ""):
        bad.append("PCN_ANCHOR_SET (アンカー残差; 使用実績なしのため gate 維持)")
    # PCN_MIX_JOBS と PCN_MIX_REGIMES の併用は actor 側でも後勝ち(regimes が jobs_set を
    # 上書き)の未定義的挙動 → 工場は明示ブロック。
    if _mix_jobs() and _mix_regimes():
        bad.append("PCN_MIX_JOBS + PCN_MIX_REGIMES 併用 (actor 側でも上書き競合する未定義構成)")
    # defer バックエンドは ev(イベント駆動; 既定) / dense(旧フル per-node) のみ。
    if os.environ.get("PCN_GPU_DEFER_KERNEL", "ev").strip() not in ("", "ev", "dense"):
        bad.append(
            f"PCN_GPU_DEFER_KERNEL={os.environ.get('PCN_GPU_DEFER_KERNEL')} (ev/dense のみ)")
    # defer無し(count系)バックエンドは ev(イベント駆動厳密; 既定) / count(旧近似退避) のみ。
    if os.environ.get("PCN_GPU_COUNT_KERNEL", "ev").strip() not in ("", "ev", "count"):
        bad.append(
            f"PCN_GPU_COUNT_KERNEL={os.environ.get('PCN_GPU_COUNT_KERNEL')} (ev/count のみ)")
    # --- forward 系レシピの一致 gate ---
    try:
        int(os.environ.get("PCN_FC_DEPTH", "2") or "2")
    except ValueError:
        bad.append(f"PCN_FC_DEPTH={os.environ.get('PCN_FC_DEPTH')} (不正値)")
    # PCN_ARCH: mlp(既定)と attn(_AttnStateEncoder JAX 移植済み)のみ。他は未知として弾く。
    if os.environ.get("PCN_ARCH", "").strip() not in ("", "mlp", "attn"):
        bad.append(f"PCN_ARCH={os.environ.get('PCN_ARCH')} (mlp/attn のみ対応)")
    # PCN_SCALEFREE_ENV=1: 観測102次元の別 env(ScaleFreeSchedulingEnv, 実験系)。
    # 工場は event-native 観測(221+α)専用のため gate 維持。
    if os.environ.get("PCN_SCALEFREE_ENV", "0") == "1":
        bad.append("PCN_SCALEFREE_ENV=1 (別envのため gate 維持)")
    # 工場 obs は urgency 常時付与(基底221次元)。urgency OFF 構成(220)は未対応。
    if os.environ.get("SCHEDULER_OBS_URGENCY", "0") != "1":
        bad.append("SCHEDULER_OBS_URGENCY!=1 (工場 obs は urgency 前提の221次元)")
    # SCHEDULER_OBS_JQ_NORM=1: job_queue 5x8 のスケールフリー正規化(col7 が現在時刻依存)。
    # count 工場は jq を事前計算しており毎ステップ時刻補正が要る=未対応(gate)。
    if os.environ.get("SCHEDULER_OBS_JQ_NORM", "0") == "1":
        bad.append("SCHEDULER_OBS_JQ_NORM=1 (jq正規化は時刻依存列のため未対応; gate)")
    # PCN_DIMLESS_NORM=1: 報酬をworkload達成レンジ(2点掃引で計測)で無次元化。
    # スケール計測プロトコル込みの移植が必要=未対応(gate)。
    if os.environ.get("PCN_DIMLESS_NORM", "0") == "1":
        bad.append("PCN_DIMLESS_NORM=1 (報酬無次元化スケール計測が未移植; gate)")
    return bad


class GPUFactoryWorker:
    """Ray Actor: GPU rollout 工場（ray.remote(num_gpus=1) で包んで起動する）。"""

    def __init__(self, config: dict, learner, buffer):
        # JAX/torch import 前に GPU を確定。num_gpus=0 で学習器と同一GPUへ co-locate する場合、
        # Ray は CUDA_VISIBLE_DEVICES を空にするため PCN_GPU_FACTORY_DEVICE で使用物理GPUを明示する。
        _dev = os.environ.get("PCN_GPU_FACTORY_DEVICE", "").strip()
        if _dev:
            os.environ["CUDA_VISIBLE_DEVICES"] = _dev
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        for p in (_REPO_ROOT, _PROTO_DIR):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.config = config
        self.learner = learner
        self.buffer = buffer
        self.n_jobs = int(config["param_env"]["n_jobs"])
        self.n_on = int(config["param_env"]["n_on_premise_node"])
        self.n_cl = int(config["param_env"]["n_cloud_node"])
        self.n_window = int(config["param_env"]["n_window"])
        self.chunk_b = int(os.environ.get("PCN_GPU_FACTORY_CHUNK", "256"))
        # イベント配列/観測バッファの静的上限（40960 では live onprem event 数がノード容量で頭打ち
        # になるため 2048/3072 で足りる想定。overflow=True 返却時はこの env を増やす）。
        self.e_alloc = int(os.environ.get("PCN_GPU_FACTORY_E_ALLOC", "2048"))
        self.e_obs = int(os.environ.get("PCN_GPU_FACTORY_E_OBS", "3072"))
        self.kcompact = int(os.environ.get("PCN_GPU_FACTORY_KCOMPACT", "64"))
        # Phase1 要約(episode_summaries)は先頭N件のみ JSONL 出力される debug 用。長い
        # エピソード(例 40960)で per-step actions 列を持つと Ray 直列化 + JSONL 書出しが
        # ホスト側で数分かかる(GPU rollout とは無関係のオーバーヘッド)。このしきい値超えの
        # エピソードは actions 列を省く(集計 action_counts は維持)。
        self._summary_maxlen = int(os.environ.get("PCN_GPU_FACTORY_SUMMARY_MAXLEN", "8192"))
        # 配列フロー（既定ON）: Transition オブジェクトに展開せず配列のまま buffer→learner へ
        # 流す。40960 級で learner 初期取込 >29 分（Ray 直列化 655万オブジェクト）を潰す。
        # PCN_GPU_FACTORY_ARRAY=0 で従来 Transition 経路（検証済み・小Nと比較用）に戻せる。
        self.array_mode = os.environ.get("PCN_GPU_FACTORY_ARRAY", "1") == "1"
        self.seed0 = int(os.environ.get("PCN_GPU_FACTORY_SEED", "20260708"))
        # [defer 3行動] SCHEDULER_ALLOW_DEFER=1 のとき defer 工場で rollout する。
        # count 版(弾性cloud)は defer 非対応のまま。
        # バックエンド(PCN_GPU_DEFER_KERNEL):
        #   "ev"(既定) = factory_defer_ev: イベント駆動・区間RLE厳密カーネル。
        #                ノード×時刻/イベント×ノードの密テンソルを作らず、
        #                大容量正容量(4096×22700/90800)でも OOM しない。
        #   "dense"    = 旧 factory_defer(フル per-node env; 小規模検証・退避用)。
        self.defer = os.environ.get("SCHEDULER_ALLOW_DEFER", "0") == "1"
        self.defer_kernel = (
            os.environ.get("PCN_GPU_DEFER_KERNEL", "ev").strip() or "ev")
        self.defer_max = int(os.environ.get("SCHEDULER_DEFER_MAX", "3"))
        self.defer_offset = int(os.environ.get("SCHEDULER_DEFER_OFFSET", "1"))
        # [2026-08-06 バグ修正] defer無し(count系)の生産バックエンド:
        #   "ev"(新既定) = factory_defer_ev を defer_max=0(3行動目封じ)で回す。
        #                  イベント駆動の厳密カーネル=torch env と待ちも完全一致。
        #   "count"      = 旧 run_fused_count_*(退避用)。onprem 配置が「時間非重複
        #                  イベントのノード共有を数えない」近似 + cloud 弾性化のため、
        #                  過密設定で待ちが真値から大きく乖離する(コストは厳密一致)。
        self.count_kernel = (
            os.environ.get("PCN_GPU_COUNT_KERNEL", "ev").strip() or "ev")
        # [defer 高速運用] Phase3 sample/greedy の scan 長を T*(この係数) に短縮
        # (例 1.5)。未完走(all_done=False)チャンクはフル長 T*(1+defer_max) で自動
        # 再実行(同 seed/episode_id=同一軌道の続きなので結果は完全一致)。空=OFF(既定)。
        _tf = os.environ.get("PCN_GPU_DEFER_TSCAN_FRAC", "").strip()
        self.defer_tscan_frac = float(_tf) if _tf else 0.0
        self._collect_outcomes = os.environ.get("DISTRIBUTED_PCN_CMD_OUTCOMES", "0") == "1"
        # [PCN_GPU_RAW_KERNEL] Phase1 ランダム収集を「rawカーネル(numba CUDA, 厳密env)+
        # 決定論リプレイ(step_with_start_hint で観測構築)」経路に切り替える(既定 OFF=
        # 従来経路と 1bit も変わらない)。defer 非対応・配列フロー(array_mode)必須。
        # numba CUDA は nvvm が要る(CUDA_HOME 指定; verify_raw_rollout.py ヘッダ参照)。
        # 初期化に失敗した場合は明確なエラーを印字して従来経路へフォールバックする。
        self.raw_kernel = os.environ.get("PCN_GPU_RAW_KERNEL", "0") == "1"
        self._raw_ready = None        # None=未判定 / True/False=probe 結果キャッシュ
        self.raw_e_max = int(os.environ.get("PCN_GPU_RAW_E_MAX", "16384"))
        self.raw_k = int(os.environ.get("PCN_GPU_RAW_K", "128"))
        # [PCN_GPU_RAW_P3] Phase3 指令 rollout を「ロックステップ(step/obsカーネル+torch NN)」
        # 経路(scripts/proto_gpu_sweep/lockstep_nn.py)に切り替える(既定 OFF=従来ビット不変)。
        # obs はロックステップ中に GPU 構築・記録済み=CPU リプレイ不要。numba CUDA(nvvm)必須。
        # defer 非対応・配列フロー必須(_use_raw_phase3 で gate)。
        self.raw_p3 = os.environ.get("PCN_GPU_RAW_P3", "0") == "1"
        self._torch_model = None      # lockstep NN 方策(初回構築→以後 load_state_dict のみ)
        # 観測リプレイの並列ワーカー数(fork)。Ray actor 内で fork Pool が失敗したら逐次へ。
        self.raw_replay_nproc = int(os.environ.get(
            "PCN_RAW_REPLAY_NPROC", str(min(16, os.cpu_count() or 1))))
        self._ep_counter = 0          # episode_id グローバル通し（再現性・重複 uid 回避）
        self._fail_streak = 0
        self._jobs = None
        # [PCN_MIX_REGIMES] レジーム混合（ping で actor と同じ生成法によりキャッシュ）。
        # 空リスト=OFF（既定パスは 1bit も変えない）。
        self._regimes: List[float] = []
        self._regime_jobs: Dict[float, np.ndarray] = {}
        self._regime_rng = None
        # [PCN_MIX_JOBS] N 混合学習（ping で actor と同じ生成法によりキャッシュ）。
        self._mixn: List[int] = []
        self._mixn_jobs: Dict[int, np.ndarray] = {}
        self._mixn_rng = None

    # ---- 初期化・ヘルス ----------------------------------------------------
    def ping(self) -> Dict[str, Any]:
        """env/jobs 構築 + 対応構成チェック + JAX 初期化。train 起動時に 1 回呼ぶ。"""
        bad = _unsupported_flags()
        if bad:
            return {"ok": False, "reason": "unsupported: " + ", ".join(bad)}
        try:
            from scripts.pcn_replay_snapshot import create_eval_env

            # 学習 Actor/Learner と同一のジョブ列（JobGenerator(0, 1, ...) = job_seed=0）
            env = create_eval_env(self.config, job_seed=0, n_jobs=self.n_jobs)
            env.reset()
            self._jobs = np.asarray(env.jobs, dtype=np.float64).copy()
            obs_dim = int(env.observation_space.shape[0])

            # [PCN_MIX_REGIMES] actor(distributed_pcn :1095-1119)と同じ生成法:
            # SCHEDULER_ARRIVAL_SCALE を一時設定して JobGenerator（create_eval_env 経由=
            # 同一 job_seed=0・同一 env reset 加工）でレジーム別ジョブ配列をキャッシュ。
            self._regimes = _mix_regimes()
            if self._regimes:
                _prev_as = os.environ.get("SCHEDULER_ARRIVAL_SCALE")
                try:
                    for _rs in self._regimes:
                        os.environ["SCHEDULER_ARRIVAL_SCALE"] = str(_rs)
                        _env_r = create_eval_env(self.config, job_seed=0, n_jobs=self.n_jobs)
                        _env_r.reset()
                        self._regime_jobs[_rs] = np.asarray(
                            _env_r.jobs, dtype=np.float64).copy()
                finally:
                    if _prev_as is None:
                        os.environ.pop("SCHEDULER_ARRIVAL_SCALE", None)
                    else:
                        os.environ["SCHEDULER_ARRIVAL_SCALE"] = _prev_as
                # actor 同型の乱数系列(0x5EED + actor_id; 工場は識別子 9000)
                self._regime_rng = np.random.default_rng(0x5EED + 9000)
                # [PCN_EVAL_REGIME] 学習中 eval のレジーム。未設定 or 未知値なら従来どおり先頭。
                _er = _eval_regime()
                self._eval_regime = (
                    _er if (_er is not None and _er in self._regime_jobs) else self._regimes[0]
                )
                print(
                    f"[PCN_GPU_FACTORY regmix] レジーム混合キャッシュ生成: "
                    f"scales∈{self._regimes} (基準n_jobs={self.n_jobs}, "
                    f"学習中evalレジーム x{self._eval_regime:g}"
                    f"{'' if _er is None else ' [PCN_EVAL_REGIME指定]'})",
                    flush=True,
                )

            # [PCN_MIX_JOBS] actor(_make_env :1076-1094)と同じ生成法で各 N のジョブ列を
            # キャッシュ（create_eval_env は同一 JobGenerator(0,1,...,N,0.2,0) 経路）。
            self._mixn = _mix_jobs()
            if self._mixn:
                if self.n_jobs != max(self._mixn):
                    return {"ok": False,
                            "reason": (f"PCN_MIX_JOBS: config n_jobs={self.n_jobs} != "
                                       f"max(mix)={max(self._mixn)} (mainはmaxへ揃える規約)")}
                for _n in self._mixn:
                    _env_n = create_eval_env(self.config, job_seed=0, n_jobs=_n)
                    _env_n.reset()
                    self._mixn_jobs[_n] = np.asarray(_env_n.jobs, dtype=np.float64).copy()
                # actor 同型の乱数系列(0xA5D1 + actor_id; 工場は識別子 9000)
                self._mixn_rng = np.random.default_rng(0xA5D1 + 9000)
                print(
                    f"[PCN_GPU_FACTORY mixjobs] N混合キャッシュ生成: N∈{self._mixn} "
                    f"(基準N={self.n_jobs}, 学習中evalは基準N固定)",
                    flush=True,
                )

            from factory_fused import obs_extra_dim
            from factory_jax import OBS_DIM

            _expected = OBS_DIM + obs_extra_dim()
            if obs_dim != _expected:
                return {"ok": False,
                        "reason": (f"obs_dim {obs_dim} != {_expected} "
                                   f"(基底221 + occupancy/oracle/efficiency 追加観測)")}
            import jax

            if not self.defer:
                # [2026-08-06] defer無しバックエンドの明示ログ(既定=ev 厳密カーネル)。
                if self.count_kernel == "ev":
                    print(
                        "[PCN_GPU_FACTORY] defer無しバックエンド=ev "
                        "(イベント駆動厳密カーネル・待ちも torch env と一致。"
                        "旧近似へは PCN_GPU_COUNT_KERNEL=count で退避)", flush=True)
                else:
                    print(
                        "[PCN_GPU_FACTORY] defer無しバックエンド=count "
                        "(旧近似退避: onprem ノード共有非計上 + cloud 弾性化のため"
                        "過密設定で待ちが真値から乖離する。既定は ev)", flush=True)
            return {"ok": True, "device": str(jax.devices()[0]), "n_jobs": self.n_jobs,
                    "n_on": self.n_on, "n_cl": self.n_cl,
                    "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "e_alloc": self.e_alloc, "e_obs": self.e_obs, "chunk_b": self.chunk_b,
                    "defer": self.defer,
                    "defer_kernel": self.defer_kernel if self.defer else None,
                    "defer_max": self.defer_max if self.defer else None,
                    "defer_offset": self.defer_offset if self.defer else None,
                    "count_kernel": None if self.defer else self.count_kernel,
                    "defer_tscan_frac": self.defer_tscan_frac or None,
                    "mix_regimes": self._regimes or None,
                    "eval_regime": getattr(self, "_eval_regime", None),
                    "mix_jobs": self._mixn or None,
                    "raw_kernel": self.raw_kernel or None,
                    "raw_block": (_RAW_BLOCK and self.raw_kernel) or None,
                    "eval_lockstep": self._use_lockstep_eval() or None,
                    "raw_p3": self.raw_p3 or None}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    def healthy(self) -> bool:
        return self._fail_streak < 2

    def warmup(self) -> float:
        """jit コンパイルを学習開始前に済ませる（chunk_b 形状で 1 rollout 空回し）。"""
        t0 = time.perf_counter()
        b = min(self.chunk_b, 32)
        # episode_id は SeedSequence 由来なので非負必須。10^9 台を warmup 予約領域にして
        # 実生産(0 起点のカウンタ)と重複しない=uid/乱数列が本番と衝突しない。
        # バックエンド分岐は _produce_random に集約(defer / ev / count 全対応)。
        self._produce_random(
            [0.5] * b, 10 ** 9)
        # [PCN_GPU_RAW_KERNEL] rawカーネルの numba コンパイル(~5s)も先に済ませる。
        # 先頭8ジョブ・B=1 の極小問題で十分(カーネルは形状非依存の 1 本コンパイル)。
        # 既定 OFF はこの分岐を一切通らない=従来 warmup と完全に同一。
        if self.raw_kernel and not self.defer and self.array_mode and self._jobs is not None \
                and self._raw_kernel_ready():
            _tiny = np.ascontiguousarray(self._jobs[:8], dtype=np.float64)
            _acts0 = np.zeros((1, _tiny.shape[0]), dtype=np.int8)
            if _RAW_BLOCK:
                from raw_rollout_kernel_block import run_raw_rollout_block

                run_raw_rollout_block(_tiny, _acts0, self.n_on, self.n_cl, e_max=64, k=16,
                                      tpb=_RAW_BLOCK_TPB)
                print("[PCN_GPU_RAW_KERNEL] warmup: rawカーネル(ブロック協調版, "
                      f"{_RAW_BLOCK_TPB}スレッド/本)のコンパイル完了", flush=True)
            else:
                from raw_rollout_kernel import run_raw_rollout

                run_raw_rollout(_tiny, _acts0, self.n_on, self.n_cl, e_max=64, k=16)
                print("[PCN_GPU_RAW_KERNEL] warmup: rawカーネル(1スレッド/本)のコンパイル完了",
                      flush=True)
        # [PCN_GPU_RAW_P3] ロックステップの step/obs カーネル(numba, ~15s)も先に温める。
        # NN は不要(カーネルコンパイルは行動列既知モードで済む)。既定 OFF は分岐を通らない。
        if self.raw_p3 and not self.defer and self.array_mode and self._jobs is not None \
                and self._raw_kernel_ready():
            from lockstep_kernel import run_lockstep_rollout

            _tiny = np.ascontiguousarray(self._jobs[:8], dtype=np.float64)
            run_lockstep_rollout(
                _tiny, np.zeros((1, _tiny.shape[0]), dtype=np.int8),
                self.n_on, self.n_cl, e_max=64, k=16, n_window=self.n_window,
                collect_obs=True)
            print("[PCN_GPU_RAW_P3] warmup: lockstepカーネルのコンパイル完了", flush=True)
        return time.perf_counter() - t0

    # ---- rollout 生産のバックエンド切替（defer=defer 工場 / 既定=count 工場）----
    def _defer_run_fn(self):
        """defer rollout 関数を返す（PCN_GPU_DEFER_KERNEL: ev=イベント駆動(既定) /
        dense=旧フル per-node。両者は同シグネチャ・同形式の返り値）。"""
        if self.defer_kernel == "dense":
            from factory_defer import run_fused_defer

            return run_fused_defer
        from factory_defer_ev import run_fused_defer_ev

        return run_fused_defer_ev

    def _run_defer_kernel(self, jobs, *, fast_tscan: bool = False, **kw):
        """defer 系カーネル(run_fused_defer_ev/run_fused_defer)の実行ラッパ。

        fast_tscan=True かつ PCN_GPU_DEFER_TSCAN_FRAC>0 (ev のみ) のとき scan 長を
        T*frac に短縮して回し、未完走(all_done=False)ならフル長で当該チャンクを
        自動再実行する(同 seed/episode_id0 なので軌道は完全に同一=結果一致)。"""
        fn = self._defer_run_fn()
        frac = self.defer_tscan_frac
        if fast_tscan and frac > 0 and self.defer_kernel != "dense":
            T = int(np.asarray(jobs).shape[-2])
            res = fn(jobs, t_scan=int(np.ceil(T * frac)), **kw)
            if bool(res["stats"].get("all_done", True)):
                return res
            full = T * (1 + int(kw.get("defer_max", self.defer_max)))
            print(
                f"[PCN_GPU_FACTORY defer] t_scan={frac:g}T で未完走エピソードあり → "
                f"フル t_scan={full} で当該チャンクを再実行(結果は同一軌道)", flush=True)
            return fn(jobs, t_scan=full, **kw)
        return fn(jobs, **kw)

    def _run_count_ev(self, jobs, **kw):
        """defer無し ev カーネル実行 + 溢れフラグ別のバッファ自動拡大再実行(1回)。

        ovf フラグは「結果が torch env と一致しない可能性」の検知器。発火した種別の
        バッファだけを拡大して同 seed/episode_id0 で再実行する(bernoulli は行動列
        state 非依存で不変・sample は正しい obs での正しい軌道になる=いずれも正解側)。
        4096×正容量の実測では既定 n_amb=512 で稀に amb_on が発火(待ち0.03%ずれ)、
        n_amb 2049 / k_pick 32 で全フラグ解消・torch 完全一致。"""
        from factory_defer_ev import run_fused_defer_ev

        res = run_fused_defer_ev(jobs, **kw)
        ovf = (res.get("stats") or {}).get("ovf") or {}
        if not any(ovf.values()):
            return res
        st = res["stats"]
        T = int(np.asarray(jobs).shape[-2])
        esc: Dict[str, int] = {}
        if ovf.get("amb_on") or ovf.get("amb_cl"):
            tgt = min(int(st["e_on"]) + 1, 2561)
            if tgt > int(st["n_amb"]):
                esc["n_amb"] = tgt
            if tgt > int(st["n_amb_cl"]):
                esc["n_amb_cl"] = tgt
        if ovf.get("frag") and int(st["k_pick"]) < 64:
            esc["k_pick"] = 64
        if ovf.get("pool"):
            esc["pool"] = 2 * int(st["pool"])
        if ovf.get("ev_on") or ovf.get("ev_cl"):
            esc["e_on"] = esc["e_cl"] = min(T + 1, 2 * int(st["e_on"]))
            esc["pool"] = max(esc.get("pool", 0), 2 * esc["e_on"])
        if not esc:
            return res
        print(f"[PCN_GPU_FACTORY ev] 溢れ検知 {ovf} → バッファ拡大で再実行 {esc}",
              flush=True)
        return run_fused_defer_ev(jobs, **{**kw, **esc})

    def _produce_random(self, chunk, episode_id0: int, jobs=None):
        """Phase1 Bernoulli(p) チャンク生産。defer 構成でも p は 0/1 のみ（CPU 側
        random_action_prob と同一分布）だが、env は defer 版=有限cloud で回す。
        jobs: None=基準ジョブ列（従来） / (T,8)=別 workload 共通（regmix/mixjobs の
        グループ分割） / (B,T,8)=エピソード別（regmix; defer のみ）。"""
        if self.defer:
            return self._run_defer_kernel(
                self._jobs if jobs is None else jobs,
                probs=chunk, seed=self.seed0, episode_id0=episode_id0,
                n_on=self.n_on, n_cl=self.n_cl, n_window=self.n_window,
                defer_max=self.defer_max, defer_offset=self.defer_offset)
        if self.count_kernel == "ev":
            # [2026-08-06 バグ修正] defer無しも厳密イベント駆動カーネル(有限cloud +
            # onprem和集合の厳密カウント)。defer_max=0 で3行動目を封じる=2行動運用。
            # bernoulli の乱数列は旧 count と同一(fold_in(key,t))=行動列は不変で、
            # env の配置(待ち)だけが厳密になる。
            return self._run_count_ev(
                self._jobs if jobs is None else jobs,
                probs=chunk, seed=self.seed0, episode_id0=episode_id0,
                n_on=self.n_on, n_cl=self.n_cl, n_window=self.n_window,
                defer_max=0, defer_offset=1)
        if jobs is not None and np.asarray(jobs).ndim != 2:
            raise RuntimeError(
                "count 工場はエピソード別(B,T,8)ジョブ配列非対応 (regmix/mixjobs は"
                "グループ分割で (T,8) を渡す)")
        from factory_phase1_random import run_fused_count_random

        return run_fused_count_random(
            self._jobs if jobs is None else jobs, chunk,
            seed=self.seed0, episode_id0=episode_id0,
            n_on=self.n_on, n_cl=self.n_cl, n_window=self.n_window,
            e_alloc=self.e_alloc, e_obs=self.e_obs, kcompact=self.kcompact)

    def _produce_cmd(self, cmds, params, episode_id0: int, greedy: bool = False,
                     jobs=None):
        """Phase3 指令付き rollout / 学習中 greedy eval のチャンク生産。
        jobs: None=基準ジョブ列（従来） / (T,8)=別 workload 共通 / (B,T,8)=エピソード別。"""
        if self.defer:
            return self._run_defer_kernel(
                self._jobs if jobs is None else jobs, fast_tscan=True,
                commands=cmds, params=params, greedy=greedy, seed=self.seed0,
                episode_id0=episode_id0, n_on=self.n_on, n_cl=self.n_cl,
                n_window=self.n_window, defer_max=self.defer_max,
                defer_offset=self.defer_offset)
        if self.count_kernel == "ev":
            # [2026-08-06 バグ修正] 厳密イベント駆動カーネル(上記 _produce_random 参照)。
            return self._run_count_ev(
                self._jobs if jobs is None else jobs,
                commands=cmds, params=params, greedy=greedy, seed=self.seed0,
                episode_id0=episode_id0, n_on=self.n_on, n_cl=self.n_cl,
                n_window=self.n_window, defer_max=0, defer_offset=1)
        if jobs is not None and np.asarray(jobs).ndim != 2:
            raise RuntimeError(
                "count 工場はエピソード別(B,T,8)ジョブ配列非対応 (regmix/mixjobs は"
                "グループ分割で (T,8) を渡す)")
        from factory_phase1_random import run_fused_count_cmd

        return run_fused_count_cmd(
            self._jobs if jobs is None else jobs, cmds,
            params=params, greedy=greedy, seed=self.seed0,
            episode_id0=episode_id0, n_on=self.n_on, n_cl=self.n_cl,
            n_window=self.n_window, e_alloc=self.e_alloc, e_obs=self.e_obs,
            kcompact=self.kcompact)

    def _mix_groups(self, n: int, phase: str):
        """混合構成（regmix / mixjobs）のエピソード束を yield する。

        yield: (jobs_arg, tags, poss, hz_override)
          jobs_arg: None=基準ジョブ列 / (T,8)=グループ共通 / (B,T,8)=エピソード別
          tags: 各エピソードの uid サフィックス（":r0.5" / ":n512" / ""）len(poss) 個
          poss: チャンク内位置（probs/commands の割当に使う）
          hz_override: [PCN_MIX_JOBS] そのグループの horizon 上書き max(1, N-2)
                       （actor :1473-1474 と同じ規約）。None=上書きなし。

        抽選は actor(:1432-1439)のエピソード毎一様抽選と同分布
        （エピソード単位ビット一致は不要・分布一致が要件）。regmix×defer は従来どおり
        エピソード別 (B,T,8) の 1 グループ、regmix×count と mixjobs は同 workload の
        エピソードを束ねるグループ分割（scan の T が共通である必要があるため）。"""
        if self._regimes and self.defer:
            idx = self._regime_rng.integers(0, len(self._regimes), size=n)
            regs = [self._regimes[int(i)] for i in idx]
            jobs_b = np.stack([self._regime_jobs[r] for r in regs])
            cnt = {r: regs.count(r) for r in self._regimes}
            print(
                f"[PCN_GPU_FACTORY regmix発火] {phase} chunk={n} レジーム抽選: "
                + ", ".join(f"x{r:g}:{cnt[r]}eps" for r in self._regimes),
                flush=True,
            )
            yield jobs_b, [f":r{r:g}" for r in regs], list(range(n)), None
            return
        if self._regimes:
            idx = self._regime_rng.integers(0, len(self._regimes), size=n)
            cnt = {r: int((idx == gi).sum()) for gi, r in enumerate(self._regimes)}
            print(
                f"[PCN_GPU_FACTORY regmix発火] {phase} chunk={n} レジーム抽選"
                f"(count工場グループ分割): "
                + ", ".join(f"x{r:g}:{cnt[r]}eps" for r in self._regimes),
                flush=True,
            )
            for gi, r in enumerate(self._regimes):
                poss = [k for k in range(n) if int(idx[k]) == gi]
                if poss:
                    yield self._regime_jobs[r], [f":r{r:g}"] * len(poss), poss, None
            return
        if self._mixn:
            idx = self._mixn_rng.integers(0, len(self._mixn), size=n)
            cnt = {N: int((idx == gi).sum()) for gi, N in enumerate(self._mixn)}
            print(
                f"[PCN_GPU_FACTORY mixjobs発火] {phase} chunk={n} N抽選: "
                + ", ".join(f"N{N}:{cnt[N]}eps" for N in self._mixn),
                flush=True,
            )
            for gi, N in enumerate(self._mixn):
                poss = [k for k in range(n) if int(idx[k]) == gi]
                if poss:
                    yield (self._mixn_jobs[N], [f":n{N}"] * len(poss), poss,
                           float(max(1.0, N - 2)))
            return
        yield None, [""] * n, list(range(n)), None

    # ---- 重み変換 -----------------------------------------------------------
    def _state_to_params(self, state) -> Dict[str, Any]:
        """学習器 state_dict（torch）→ 工場 forward の params dict。

        factory_fused.params_from_state_dict に委譲（PCN_FC_DEPTH 可変 fc 層 / FILM /
        Fourier buffer / command_balance を state_dict から自動抽出）。"""
        from factory_fused import params_from_state_dict

        return params_from_state_dict(state)

    # ---- Phase 1 raw カーネル経路 (PCN_GPU_RAW_KERNEL=1) ---------------------
    def _raw_kernel_ready(self) -> bool:
        """rawカーネル(numba CUDA)が使えるか 1 回だけ probe してキャッシュする。

        numba CUDA のカーネルコンパイルには nvvm(libNVVM)が必要で、CUDA_HOME が
        nvcc/nvvm 展開先(例: nvidia-cuda-nvcc-cu12 wheel)を向いていないと失敗する
        (scripts/proto_gpu_sweep/verify_raw_rollout.py のヘッダ参照)。失敗時は明確な
        エラーを印字して False(=呼び出し側が従来経路へフォールバック)。"""
        if self._raw_ready is not None:
            return self._raw_ready
        try:
            from numba import cuda as _nb_cuda
            if not _nb_cuda.is_available():
                raise RuntimeError("numba.cuda.is_available()=False (CUDAドライバ/デバイス不可)")
            from numba.cuda.cudadrv import nvvm as _nvvm
            _nvvm.NVVM()  # libnvvm 不在ならここで NvvmSupportError
            import raw_rollout_kernel  # noqa: F401  (_PROTO_DIR は __init__ で sys.path 済)
            self._raw_ready = True
        except Exception as e:  # noqa: BLE001
            print(
                f"[PCN_GPU_RAW_KERNEL] 初期化失敗 → Phase1 は従来経路にフォールバック: "
                f"{type(e).__name__}: {e}\n"
                f"  (numba CUDA には CUDA_HOME=nvcc/nvvm 展開先の指定が必要。"
                f"CUDA_HOME={os.environ.get('CUDA_HOME', '<未設定>')})",
                flush=True,
            )
            self._raw_ready = False
        return self._raw_ready

    def _use_raw_phase1(self) -> bool:
        """run_random が rawカーネル経路を使うべきか(非対応構成は理由を印字して従来経路)。"""
        if not self.raw_kernel:
            return False
        if self.defer:
            print("[PCN_GPU_RAW_KERNEL] defer構成(SCHEDULER_ALLOW_DEFER=1)は非対応 → 従来経路",
                  flush=True)
            return False
        if not self.array_mode:
            print("[PCN_GPU_RAW_KERNEL] PCN_GPU_FACTORY_ARRAY=0 は非対応(配列フロー必須) → 従来経路",
                  flush=True)
            return False
        return self._raw_kernel_ready()

    def _produce_random_raw(self, chunk, episode_id0: int, jobs=None):
        """Phase1 Bernoulli(p) チャンク生産の rawカーネル経路。

        (a) 行動列を numpy で事前生成: エピソード e の乱数は default_rng((seed0, e)) =
            チャンク割りに依存しない決定論(state 非依存なので事前生成と rollout 中生成は等価)。
        (b) raw_rollout_kernel(numba CUDA, 1スレッド=1エピソード, 厳密env)で
            start_times/waits/costs を計算(E_MAX/K は溢れフラグで自動拡大)。
        (c) step_with_start_hint 決定論リプレイ(replay_obs_builder)で観測・報酬・
            objective_values を構築(配置探索を省くだけで通常 step と同一の観測列)。
        返り値は他の _produce_* と同形式の res dict(obs/actions/rewards/achieved/
        episode_ids/probs/seed0/stats)。obs は (B,T,D)(build_factory_array_episodes は
        [:T] しか読まないため終端観測は持たない)。"""
        from replay_obs_builder import build_replay_dataset

        t0 = time.perf_counter()
        jobs_arr = np.ascontiguousarray(
            self._jobs if jobs is None else jobs, dtype=np.float64)
        if jobs_arr.ndim != 2:
            raise RuntimeError(
                "raw カーネルはエピソード別(B,T,8)ジョブ配列非対応 "
                "(regmix/mixjobs はグループ分割で (T,8) を渡す)")
        T = int(jobs_arr.shape[0])
        B = len(chunk)
        eids = np.arange(episode_id0, episode_id0 + B, dtype=np.int64)
        actions = np.zeros((B, T), dtype=np.int8)
        for k in range(B):
            rng = np.random.default_rng((self.seed0, int(eids[k])))
            actions[k] = (rng.random(T) < float(chunk[k])).astype(np.int8)

        # (b) GPU カーネル。溢れフラグで E_MAX/K を自動拡大して再実行
        # (verify_raw_rollout --big と同じ運用。行動列は state 非依存なので結果は同一軌道)。
        e_max, kk = self.raw_e_max, self.raw_k
        gpu = None
        for _ in range(4):
            if _RAW_BLOCK:
                # [PCN_GPU_RAW_BLOCK] 1ブロック=1エピソード(既定128スレッド)。1本の中の
                # 占有カウント更新と空きノード探索を手分けする版。5万・実trace weekB で
                # 1スレッド版と (start,wait,cost) 全件一致を確認済み
                # (scripts/proto_gpu_sweep/verify_raw_rollout_block.py)。実測 6.898→0.337 ms/step。
                from raw_rollout_kernel_block import run_raw_rollout_block

                gpu = run_raw_rollout_block(jobs_arr, actions, self.n_on, self.n_cl,
                                            e_max=e_max, k=kk, tpb=_RAW_BLOCK_TPB)
            else:
                # tpb=1: 1ブロック1スレッド(ワープ発散の回避)。tpb=32既定のまま呼ぶと
                # 32人組が別エピソードを進めて分岐のたび直列化し実測5倍遅化(v5 Phase1 32分/chunkの真因)。
                # ベンチ(125本/分)は tpb=1 で計測されたもの。PCN_RAW_TPB で上書き可。
                from raw_rollout_kernel import run_raw_rollout

                gpu = run_raw_rollout(jobs_arr, actions, self.n_on, self.n_cl, e_max=e_max, k=kk,
                                      tpb=int(os.environ.get("PCN_RAW_TPB", "1")))
            if not gpu["ovf"].any():
                break
            if (gpu["ovf"] == 1).any():
                e_max = max(2 * e_max, 65536)
            if (gpu["ovf"] == 2).any():
                kk *= 4
            print(f"[PCN_GPU_RAW_KERNEL] ovf={gpu['ovf'].tolist()} → "
                  f"E_MAX={e_max} K={kk} で再実行", flush=True)
        if gpu["ovf"].any():
            raise RuntimeError(f"raw kernel ovf 解消せず (E_MAX={e_max}, K={kk})")
        t_gpu = time.perf_counter() - t0

        # (c) 決定論リプレイ(観測構築)。fork Pool が Ray actor 内で失敗したら逐次で再実行。
        t1 = time.perf_counter()
        env_kwargs = dict(n_window=self.n_window)
        try:
            episodes = build_replay_dataset(
                jobs_arr, actions, gpu["start_times"], self.n_on, self.n_cl,
                nproc=self.raw_replay_nproc, env_kwargs=env_kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[PCN_GPU_RAW_KERNEL] 並列リプレイ失敗({type(e).__name__}: {e}) → "
                  f"逐次リプレイで再実行", flush=True)
            episodes = build_replay_dataset(
                jobs_arr, actions, gpu["start_times"], self.n_on, self.n_cl,
                nproc=1, env_kwargs=env_kwargs)
        t_replay = time.perf_counter() - t1

        obs = np.stack([ep["obs"] for ep in episodes])          # (B, T, D) f32
        rewards = np.stack([ep["rewards"] for ep in episodes])  # (B, T, 2) f32
        achieved = np.asarray(
            [ep["objective_values"] for ep in episodes], dtype=np.float64)  # (B,3)
        wall = time.perf_counter() - t0
        print(f"[PCN_GPU_RAW_KERNEL] chunk={B} T={T}: kernel {t_gpu:.1f}s + "
              f"replay {t_replay:.1f}s (nproc={self.raw_replay_nproc}) = {wall:.1f}s",
              flush=True)
        return {
            "obs": obs, "actions": actions, "rewards": rewards, "achieved": achieved,
            "episode_ids": eids, "probs": np.asarray(chunk, dtype=np.float64),
            "seed0": self.seed0,
            "stats": {"wall": wall, "wall_gpu": t_gpu, "wall_replay": t_replay},
        }

    # ---- Phase 3 raw ロックステップ経路 (PCN_GPU_RAW_P3=1) -------------------
    def _use_raw_phase3(self) -> bool:
        """run_commands がロックステップ経路を使うべきか(非対応構成は理由を印字して従来経路)。"""
        if not self.raw_p3:
            return False
        if self.defer:
            print("[PCN_GPU_RAW_P3] defer構成(SCHEDULER_ALLOW_DEFER=1)は非対応 → 従来経路",
                  flush=True)
            return False
        if not self.array_mode:
            print("[PCN_GPU_RAW_P3] PCN_GPU_FACTORY_ARRAY=0 は非対応(配列フロー必須) → 従来経路",
                  flush=True)
            return False
        return self._raw_kernel_ready()

    def _use_lockstep_eval(self) -> bool:
        """学習中 eval をロックステップ経路で回すか(既定 OFF)。

        既定 OFF の理由: 現行の eval は方策の推論も JAX 実装だが、ロックステップは torch の
        NN を使う。同じ重みでも fp32 の演算順序が違うため greedy の argmax が僅差で割れうる
        =eval の値が微妙に変わり、過去の実験との比較可能性に影響する
        (verify_lockstep_nn.py の合格条件が「行動一致率>99.9%・目的値の相対差<1e-5」で
         完全一致を諦めているのが根拠)。切り替える前に新旧の差を必ず測ること。
        速度は 5万・実trace の実測で 16指令 373秒 → 約100秒、格子372指令なら 127分 → 約4分。
        """
        if os.environ.get("PCN_GPU_EVAL_LOCKSTEP", "0") != "1":
            return False
        if self.defer:
            print("[PCN_GPU_EVAL_LOCKSTEP] defer構成は非対応 → 従来の eval 経路", flush=True)
            return False
        if not self.array_mode:
            print("[PCN_GPU_EVAL_LOCKSTEP] 配列フロー必須 → 従来の eval 経路", flush=True)
            return False
        return self._raw_kernel_ready()

    def _produce_cmd_raw(self, cmds, state, episode_id0: int, greedy: bool = False,
                         jobs=None):
        """Phase3 指令付き rollout のロックステップ経路(lockstep_nn.run_lockstep_greedy)。

        1 step = obs構築カーネル + torch NN(バッチB) + 配置カーネル + 指令更新(torch) を
        T 回まわす。obs[B,T,224] はロックステップ中に GPU 構築・記録済み=リプレイ不要。
        greedy=False(既定・Phase3生産)は pcn_agent._act(eval_mode=False) と同分布の
        multinomial サンプル(乱数は torch cuda Generator; CPU とビット別・分布一致が要件)。
        greedy 検証は verify_lockstep_nn.py で CPU _run_episode と完全一致を確認済み。
        返り値は他の _produce_* と同形式の res dict。"""
        from lockstep_nn import build_policy_model, run_lockstep_greedy

        t0 = time.perf_counter()
        jobs_arr = np.ascontiguousarray(
            self._jobs if jobs is None else jobs, dtype=np.float64)
        if jobs_arr.ndim != 2:
            raise RuntimeError(
                "raw P3 はエピソード別(B,T,8)ジョブ配列非対応 "
                "(regmix/mixjobs はグループ分割で (T,8) を渡す)")
        T = int(jobs_arr.shape[0])
        B = len(cmds)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        if self._torch_model is None:
            self._torch_model = build_policy_model(sd, self.n_jobs, device="cuda")
        else:
            self._torch_model.load_state_dict(sd, strict=False)

        commands = np.stack([np.asarray(c[0], dtype=np.float32) for c in cmds])
        horizons = np.asarray([float(c[1]) for c in cmds], dtype=np.float32)
        eids = np.arange(episode_id0, episode_id0 + B, dtype=np.int64)

        # 溢れフラグで E_MAX/K を自動拡大して再実行(_produce_random_raw と同じ運用)。
        e_max, kk = self.raw_e_max, self.raw_k
        out = None
        for _ in range(4):
            out = run_lockstep_greedy(
                jobs_arr, self._torch_model, commands, self.n_on, self.n_cl,
                n_window=self.n_window, horizons=horizons, e_max=e_max, k=kk,
                tpb=int(os.environ.get("PCN_RAW_TPB", "1")),
                mode=("greedy" if greedy else "sample"),
                sample_seed=self.seed0 * 1_000_003 + int(episode_id0),
                return_obs=True,
            )
            if not out["ovf"].any():
                break
            if (out["ovf"] == 1).any():
                e_max = max(2 * e_max, 65536)
            if (out["ovf"] == 2).any():
                kk *= 4
            print(f"[PCN_GPU_RAW_P3] ovf={out['ovf'].tolist()} → "
                  f"E_MAX={e_max} K={kk} で再実行", flush=True)
        if out["ovf"].any():
            raise RuntimeError(f"raw P3 lockstep ovf 解消せず (E_MAX={e_max}, K={kk})")

        obs = out["obs"].cpu().numpy()                                   # (B,T,D) f32
        rewards = np.stack(
            [-out["waits"], -out["costs"]], axis=-1).astype(np.float32)  # (B,T,2)
        achieved = np.stack([
            out["objectives"][:, 0],                     # total cost
            out["makespan"].astype(np.float64),          # makespan
            out["objectives"][:, 1],                     # avg wait
        ], axis=1)
        wall = time.perf_counter() - t0
        print(f"[PCN_GPU_RAW_P3] chunk={B} T={T} mode={'greedy' if greedy else 'sample'}: "
              f"lockstep {wall:.1f}s ({wall / max(1, T) * 1e6:.0f}us/step)", flush=True)
        return {
            "obs": obs, "actions": out["actions"], "rewards": rewards,
            "achieved": achieved,
            "commands0": commands.copy(),
            "episode_ids": eids, "seed0": self.seed0,
            "stats": {"wall": wall},
            "overflow": False,
        }

    # ---- Phase 1: ランダム収集 ----------------------------------------------
    def run_random(self, n_episodes: int, probs: Sequence[float]) -> Dict[str, Any]:
        """Bernoulli(p) スイープを一括生産（既存 actor.run(random_actions=True,
        random_action_probs=…) の総和と同じ分布ソース）。戻り dict は actor.run 互換。"""
        try:
            import ray

            from factory_jax import episodes_to_transitions

            probs_full = [float(probs[i % len(probs)]) for i in range(n_episodes)]
            # [PCN_GPU_RAW_KERNEL] rawカーネル+観測リプレイ経路(既定 OFF=従来と 1bit 不変)。
            _use_raw = self._use_raw_phase1()
            summaries = []
            prob_counts: Dict[float, int] = {}
            n_added = 0
            overflow = False
            wall = 0.0
            n_used = 0    # チャンク/グループ横断の episode_id 通し（従来 i0 と同値になる）
            for i0 in range(0, n_episodes, self.chunk_b):
                chunk = probs_full[i0:i0 + self.chunk_b]
                # [PCN_MIX_REGIMES/PCN_MIX_JOBS] エピソード毎一様抽選（OFF は単一グループ=従来）
                for _jobs_g, _tags, _poss, _hz_ov in self._mix_groups(len(chunk), "Phase1"):
                    sub = [chunk[k] for k in _poss]
                    eid0 = self._ep_counter + n_used
                    if _use_raw:
                        res = self._produce_random_raw(sub, eid0, jobs=_jobs_g)
                    else:
                        res = self._produce_random(sub, eid0, jobs=_jobs_g)
                    n_used += len(sub)
                    overflow = overflow or bool(res.get("overflow"))
                    wall += float(res["stats"]["wall"])
                    if self.array_mode:
                        from src.distributed.factory_episode import build_factory_array_episodes

                        if _use_raw:
                            _uids = [
                                f"phase1:raw:{res['seed0']}:{int(res['episode_ids'][k])}"
                                f":p{float(res['probs'][k])}" + _tags[k]
                                for k in range(len(sub))
                            ]
                        else:
                            _uids = [
                                f"gpufactory:{res['seed0']}:{int(res['episode_ids'][k])}:{float(res['probs'][k])}"
                                + _tags[k]
                                for k in range(len(sub))
                            ]
                        _rap = [float(res["probs"][k]) for k in range(len(sub))]
                        episodes = build_factory_array_episodes(res, uids=_uids, random_action_probs=_rap)
                        summaries.extend(self._summaries_from_res(
                            res, eid0 - self._ep_counter, sub, tags=_tags))
                    else:
                        episodes = episodes_to_transitions(res)
                        summaries.extend(self._summaries(episodes, eid0 - self._ep_counter, sub))
                    n_added += int(ray.get(self.buffer.add_batch.remote(episodes)))
                    for p in sub:
                        prob_counts[p] = prob_counts.get(p, 0) + 1
            self._ep_counter += n_episodes
            self._fail_streak = 0
            return {
                "episodes_generated": n_added,
                "command_outcomes": [],
                "episode_summaries": summaries,
                "action_one_prob_counts": prob_counts,
                "overflow": overflow,
                "wall": wall,
            }
        except Exception as e:  # noqa: BLE001
            self._fail_streak += 1
            traceback.print_exc()
            return {"episodes_generated": 0, "command_outcomes": [],
                    "episode_summaries": [], "action_one_prob_counts": {},
                    "_factory_failed": f"{type(e).__name__}: {e}"}

    # ---- Phase 3: 指令付き rollout -------------------------------------------
    def run_commands(self, commands: Sequence) -> Dict[str, Any]:
        """pre_fetched_commands 互換: [(desired_return, horizon[, base_return]), ...]。

        同質チャンク運用（Phase 3.5 の知見: 指令を cost 目標でソートして chunk 化すると
        資源別ビュー幅が縮み +25%）。command_outcomes は既存 Actor :1193-1201 と同形式。
        """
        try:
            import ray

            from factory_jax import episodes_to_transitions

            # 重み同期: 既存 Actor.run 冒頭と同じ順序点（Learner actor キュー経由の共有 ref）。
            # FIFO キューで直前 learn の後ろに並ぶ＝learn 完了後の重みで rollout（Actor と同順序点）。
            wref = ray.get(self.learner.get_weights_ref.remote())
            state = ray.get(wref) if isinstance(wref, ray.ObjectRef) else wref
            # [PCN_GPU_RAW_P3] ロックステップ経路は torch state_dict を直接使う(JAX params 不要)。
            _use_raw_p3 = self._use_raw_phase3()
            params = None if _use_raw_p3 else self._state_to_params(state)

            trip = []
            for c in commands:
                dr = np.asarray(c[0], dtype=np.float32)
                hz = float(c[1])
                base = np.asarray(c[2], dtype=np.float32) if (
                    isinstance(c, (list, tuple)) and len(c) >= 3) else dr.copy()
                trip.append((dr, hz, base))
            # 同質チャンク: cost 目標 dr[1] 昇順（安い端→高い端でまとまる）
            order = np.argsort([t[0][1] for t in trip], kind="stable")
            n_scale = max(1, self.n_jobs)
            outcomes = []
            n_added = 0
            overflow = False
            wall = 0.0
            n_used = 0
            for i0 in range(0, len(order), self.chunk_b):
                sel = order[i0:i0 + self.chunk_b]
                # [PCN_MIX_REGIMES/PCN_MIX_JOBS] エピソード毎一様抽選（OFF は単一グループ=従来）
                for _jobs_g, _tags, _poss, _hz_ov in self._mix_groups(len(sel), "Phase3"):
                    sel_g = [sel[k] for k in _poss]
                    # [PCN_MIX_JOBS] horizon をそのエピソードの実ジョブ数へ上書き
                    # (actor :1473-1474 と同じ規約。指令 desired_return はそのまま)。
                    cmds = [(trip[j][0], trip[j][1] if _hz_ov is None else _hz_ov)
                            for j in sel_g]
                    eid0 = self._ep_counter + n_used
                    if _use_raw_p3:
                        res = self._produce_cmd_raw(cmds, state, eid0, jobs=_jobs_g)
                    else:
                        res = self._produce_cmd(cmds, params, eid0, jobs=_jobs_g)
                    n_used += len(sel_g)
                    overflow = overflow or bool(res.get("overflow"))
                    wall += float(res["stats"]["wall"])
                    if self.array_mode:
                        from src.distributed.factory_episode import build_factory_array_episodes

                        _uids = [
                            f"gpufactory_cmd:{res['seed0']}:{int(res['episode_ids'][k])}"
                            + _tags[k]
                            for k in range(len(sel_g))
                        ]
                        _cmd_rets = [res["commands0"][k] for k in range(len(sel_g))]
                        _wall_per_ep = float(res["stats"]["wall"]) / max(1, len(sel_g))
                        episodes = build_factory_array_episodes(
                            res, uids=_uids, command_returns=_cmd_rets,
                            solution_execution_time=_wall_per_ep)
                    else:
                        episodes = episodes_to_transitions(res)
                    n_added += int(ray.get(self.buffer.add_batch.remote(episodes)))
                    if self._collect_outcomes:
                        ach_r = res["rewards"].sum(axis=1)          # γ=1 累積 (B,2)
                        for k, j in enumerate(sel_g):
                            dr, _, base = trip[j]
                            cost, _, avg_wait = (
                                episodes[k][0].objective_values
                            )
                            outcomes.append({
                                "base_return": base.tolist(),
                                "command_return": dr.tolist(),
                                "achieved_return": ach_r[k].astype(np.float32).tolist(),
                                "command_values": [float(-dr[1]), float(-dr[0] / n_scale)],
                                "achieved_values": [float(cost), float(avg_wait)],
                            })
            self._ep_counter += len(order)
            self._fail_streak = 0
            return {
                "episodes_generated": n_added,
                "command_outcomes": outcomes,
                "episode_summaries": [],
                "action_one_prob_counts": {},
                "overflow": overflow,
                "wall": wall,
            }
        except Exception as e:  # noqa: BLE001
            self._fail_streak += 1
            traceback.print_exc()
            return {"episodes_generated": 0, "command_outcomes": [],
                    "episode_summaries": [], "action_one_prob_counts": {},
                    "_factory_failed": f"{type(e).__name__}: {e}"}

    # ---- 学習中 eval（greedy 格子）を O(N) 工場で ----------------------------
    def eval_commands(self, commands, use_eval_weights: bool = True) -> Dict[str, Any]:
        """(desired_return, horizon) 格子を greedy(決定的)で評価。CPU env の O(N^2) 格子評価を置換。

        戻り: values=(N,2)[cost, avg_wait] / episode_returns=(N,2)[-Σwait,-Σcost]。
        _run_uniform_grid_commands の (N,2)[cost,wait] と ingest_distributed_eval_results の
        (episode_return, value) の両方を満たす。重みは eval 用（EMA 有効時 EMA）。
        """
        try:
            import ray

            if use_eval_weights:
                wref = ray.get(self.learner.get_eval_weights_ref.remote())
            else:
                wref = ray.get(self.learner.get_weights_ref.remote())
            state = ray.get(wref) if isinstance(wref, ray.ObjectRef) else wref
            params = self._state_to_params(state)
            n_cmds = len(commands)
            values = np.zeros((n_cmds, 2), dtype=np.float64)
            ep_returns = np.zeros((n_cmds, 2), dtype=np.float64)
            overflow = False
            t0 = time.perf_counter()
            # [PCN_MIX_REGIMES] 学習中 eval のレジーム(既定=リスト先頭, PCN_EVAL_REGIME で変更可)
            # (actor evaluate_episode/eval_uniform_grid_batch :1681,1729 と同じ規約)。
            _jobs_eval = (self._regime_jobs[getattr(self, "_eval_regime", self._regimes[0])]
                          if self._regimes else None)
            # [PCN_GPU_EVAL_LOCKSTEP] ロックステップ経路は本数を増やしても時間がほぼ増えない
            # ので、eval のまとめ数はこちらだけ別に大きく取れる(既定は従来と同じ chunk_b)。
            _ls_eval = self._use_lockstep_eval()
            _eval_chunk = int(os.environ.get("PCN_GPU_EVAL_CHUNK", "0")) or self.chunk_b
            if not _ls_eval:
                _eval_chunk = self.chunk_b
            for i0 in range(0, n_cmds, _eval_chunk):
                chunk = commands[i0:i0 + _eval_chunk]
                cmds = [(np.asarray(c[0], dtype=np.float32), float(c[1])) for c in chunk]
                if _ls_eval:
                    res = self._produce_cmd_raw(cmds, state, 0, greedy=True, jobs=_jobs_eval)
                else:
                    res = self._produce_cmd(cmds, params, 0, greedy=True, jobs=_jobs_eval)
                overflow = overflow or bool(res.get("overflow"))
                ach = res["achieved"]                # (b,3) [cost, makespan, avg_wait]
                er = res["rewards"].sum(axis=1)      # (b,2) [-Σwait, -Σcost]
                for k in range(len(chunk)):
                    values[i0 + k] = (float(ach[k, 0]), float(ach[k, 2]))
                    ep_returns[i0 + k] = er[k].astype(np.float64)
            self._fail_streak = 0
            return {"values": values.tolist(), "episode_returns": ep_returns.tolist(),
                    "overflow": overflow, "n": int(n_cmds),
                    "wall": float(time.perf_counter() - t0)}
        except Exception as e:  # noqa: BLE001
            self._fail_streak += 1
            traceback.print_exc()
            return {"_factory_failed": f"{type(e).__name__}: {e}"}

    # ---- 内部 ----------------------------------------------------------------
    def _summaries(self, episodes, ep_index0: int, probs_chunk) -> List[Dict[str, Any]]:
        """_summarize_episode_for_log（distributed_pcn.py:361-384）互換の要約。
        actor_id=9000 を工場の識別子とする。"""
        out = []
        maxlen = self._summary_maxlen
        for k, ep in enumerate(episodes):
            n = len(ep)
            # 単一パスで action 集計 + reward 累積（list-of-arrays np.sum を避ける）。
            counts: Dict[int, int] = {}
            r0 = 0.0
            r1 = 0.0
            for t in ep:
                a = int(t.action)
                counts[a] = counts.get(a, 0) + 1
                rw = t.reward
                r0 += float(rw[0])
                r1 += float(rw[1])
            first = ep[0]
            out.append({
                "actor_id": 9000,
                "actor_episode_index": int(ep_index0 + k),
                "random_action_prob": float(probs_chunk[k]),
                "episode_length": int(n),
                "total_reward": [r0, r1],
                "objective_values": (
                    np.asarray(first.objective_values, dtype=np.float64).ravel().tolist()
                    if hasattr(first, "objective_values") else None
                ),
                "action_counts": {str(a): int(c) for a, c in counts.items()},
                # 長エピソードは per-step actions 列を省く（JSONL/Ray 直列化コスト回避）。
                "actions": [int(t.action) for t in ep] if n <= maxlen else [],
            })
        return out

    def _summaries_from_res(self, res, ep_index0: int, probs_chunk,
                            tags=None) -> List[Dict[str, Any]]:
        """配列モード用: res(obs/actions/rewards/achieved) から要約を**ベクトル化**で生成。
        per-transition Python ループを一切回さない（40960 で要約だけで数分かかっていた主因）。
        tags: [PCN_MIX_REGIMES/PCN_MIX_JOBS] 各エピソードの uid サフィックス
        （":r0.5"→arrival_scale / ":n512"→mix_n を要約に付記。None/空=従来と同一出力）。
        """
        actions = np.asarray(res["actions"])        # (B, T) i8
        rewards = np.asarray(res["rewards"])         # (B, T, 2)
        achieved = np.asarray(res["achieved"], dtype=np.float64)
        lengths = res.get("lengths")                 # defer 工場: (B,) 実エピソード長
        maxlen = self._summary_maxlen
        out = []
        B, T = actions.shape
        tot = rewards.sum(axis=1)                    # (B, 2) 生reward累積（padding は 0）
        for k in range(B):
            L = int(lengths[k]) if lengths is not None else T
            a = actions[k][:L]
            uniq, cnt = np.unique(a, return_counts=True)
            s = {
                "actor_id": 9000,
                "actor_episode_index": int(ep_index0 + k),
                "random_action_prob": float(probs_chunk[k]),
                "episode_length": int(L),
                "total_reward": [float(tot[k, 0]), float(tot[k, 1])],
                "objective_values": [
                    float(achieved[k, 0]), float(achieved[k, 1]), float(achieved[k, 2])],
                "action_counts": {str(int(x)): int(c) for x, c in zip(uniq, cnt)},
                "actions": a.astype(np.int64).tolist() if L <= maxlen else [],
            }
            if tags is not None and tags[k]:
                if tags[k].startswith(":r"):
                    s["arrival_scale"] = float(tags[k][2:])   # [PCN_MIX_REGIMES] 監査用
                elif tags[k].startswith(":n"):
                    s["mix_n"] = int(tags[k][2:])             # [PCN_MIX_JOBS] 監査用
            out.append(s)
        return out
