import numpy as np
import torch as th
import yaml
from tqdm import tqdm
import time
import torch.nn.functional as F
import gzip
import heapq
import json
import os
import pickle
import shutil
import sys
from typing import Any, Dict, List, Tuple, Optional, Union


def _apply_cli_env_overrides(argv: List[str]) -> None:
    """Map common ``--`` options to the existing environment-variable controls.

    Some switches, such as observation mode, are read during module import, so this
    must run before those globals and before importing ``pcn_agent``.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    value_options = {
        "workdir": "DISTRIBUTED_PCN_WORKDIR",
        "output_dir": "DISTRIBUTED_PCN_OUTPUT_DIR",
        "run_dir": "DISTRIBUTED_PCN_RUN_DIR",
        "config": "DISTRIBUTED_PCN_CONFIG",
        "jobs": "DISTRIBUTED_PCN_JOBS",
        "onprem": "DISTRIBUTED_PCN_ONPREM",
        "cloud": "DISTRIBUTED_PCN_CLOUD",
        "n_iterations": "DISTRIBUTED_PCN_N_ITERATIONS",
        "n_actors": "DISTRIBUTED_PCN_N_ACTORS",
        "initial_episodes": "DISTRIBUTED_PCN_INITIAL_EPISODES",
        "eval_interval": "DISTRIBUTED_PCN_EVAL_INTERVAL",
        "supervised_epochs": "DISTRIBUTED_PCN_SUPERVISED_EPOCHS",
        "supervised_updates_per_epoch": "DISTRIBUTED_PCN_SUPERVISED_UPDATES_PER_EPOCH",
        "phase2_importance_samples": "DISTRIBUTED_PCN_PHASE2_IMPORTANCE_SAMPLES",
        "initial_action_sweep_probs": "DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP_PROBS",
        "initial_episode_cache_path": "DISTRIBUTED_PCN_INITIAL_EPISODE_CACHE_PATH",
        "initial_episode_log_path": "DISTRIBUTED_PCN_INITIAL_EPISODE_LOG_PATH",
        "initial_episode_log_limit": "DISTRIBUTED_PCN_INITIAL_EPISODE_LOG_LIMIT",
        "mo_hv_export": "DISTRIBUTED_PCN_MO_HV_EXPORT",
        "eval_samples": "DISTRIBUTED_PCN_EVAL_SAMPLES",
        "desired_return_clip": "PCN_DESIRED_RETURN_CLIP",
        "desired_return_scale": "PCN_DESIRED_RETURN_SCALE",
        "command_alpha": "PCN_COMMAND_ALPHA",
        "train_pf_weight": "PCN_TRAIN_PF_WEIGHT",
        "train_endpoint_weight": "PCN_TRAIN_ENDPOINT_WEIGHT",
        "train_recent_weight": "PCN_TRAIN_RECENT_WEIGHT",
        "train_cost_endpoint_weight": "PCN_TRAIN_COST_ENDPOINT_WEIGHT",
        "return_norm_ema": "PCN_RETURN_NORM_EMA",
        "return_norm_min_scale": "PCN_RETURN_NORM_MIN_SCALE",
    }
    bool_options = {
        "profile": "DISTRIBUTED_PCN_PROFILE",
        "quick": "DISTRIBUTED_PCN_QUICK",
        "async_overlap": "DISTRIBUTED_PCN_ASYNC_OVERLAP",
        "phase3_gpu_cache": "DISTRIBUTED_PCN_PHASE3_GPU_CACHE",
        "phase2_importance": "DISTRIBUTED_PCN_PHASE2_IMPORTANCE",
        "initial_action_sweep": "DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP",
        "fast": "DISTRIBUTED_PCN_FAST",
        "use_jax": "DISTRIBUTED_PCN_USE_JAX",
        "use_event_obs": "DISTRIBUTED_PCN_USE_EVENT_OBS",
        "use_event_native": "DISTRIBUTED_PCN_USE_EVENT_NATIVE",
        "enable_visualization": "DISTRIBUTED_PCN_ENABLE_VISUALIZATION",
        "log_gpu_mem": "DISTRIBUTED_PCN_LOG_GPU_MEM",
        "log_ray_transfer": "DISTRIBUTED_PCN_LOG_RAY_TRANSFER",
        "eval_diag": "DISTRIBUTED_PCN_EVAL_DIAG",
        "archive_vis_all": "DISTRIBUTED_PCN_ARCHIVE_VIS_ALL",
        "vis_reward_plot": "DISTRIBUTED_PCN_VIS_REWARD_PLOT",
        "vis_command_arrows": "DISTRIBUTED_PCN_VIS_COMMAND_ARROWS",
        "skip_final_eval": "DISTRIBUTED_PCN_SKIP_FINAL_EVAL",
        "learner_bitmap": "SCHEDULER_LEARNER_BITMAP",
        "event_to_bitmap": "DISTRIBUTED_PCN_EVENT_TO_BITMAP",
        "adaptive_return_normalization": "PCN_ADAPTIVE_RETURN_NORMALIZATION",
        "empty_cache_every_update": "PCN_EMPTY_CACHE_EVERY_UPDATE",
        "use_torch_compile": "DISTRIBUTED_PCN_USE_TORCH_COMPILE",
        "build_schedule_maps": "SCHEDULER_BUILD_SCHEDULE_MAPS",
    }

    for dest in value_options:
        parser.add_argument(f"--{dest.replace('_', '-')}", dest=dest)
    for dest in bool_options:
        parser.add_argument(
            f"--{dest.replace('_', '-')}",
            dest=dest,
            action=argparse.BooleanOptionalAction,
            default=None,
        )

    args, _ = parser.parse_known_args(argv)
    for dest, env_name in value_options.items():
        value = getattr(args, dest, None)
        if value is not None:
            os.environ[env_name] = str(value)
    for dest, env_name in bool_options.items():
        value = getattr(args, dest, None)
        if value is not None:
            os.environ[env_name] = "1" if value else "0"


_apply_cli_env_overrides(sys.argv[1:])

# Singularity/コンテナ環境向け: ヘッドレスでmatplotlibを使用（ディスプレイ不要）
if os.environ.get('MPLBACKEND') != 'Agg':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import ray
import copy
import warnings
import gc  # ガベージコレクション用
import psutil  # メモリ情報取得用
# CUDAが利用できない場合の警告を抑制
warnings.filterwarnings('ignore', message="Can't initialize NVML")
warnings.filterwarnings('ignore', message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available")
# TF32 を有効化（A40 等で高速化）、警告も抑制
if th.cuda.is_available():
    th.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore', message="TensorFloat32 tensor cores")

# =========================
# 0. ハイパーパラメータ設定
# =========================

DEBUG = False
TIME_DEBUG = True  # 各フェーズの経過時間を表示
ENABLE_VISUALIZATION = True

N_ITERATIONS = 100  # フェーズ3の学習イテレーション数（config.yml で上書き可）
N_ACTORS = 32      # 並列実行するActorの数
N_JOBS = 24 # ジョブ数

EVAL_INTERVAL = 5  # 評価を実行する間隔（イテレーション数）
# 複数 Actor で評価エピソードを並列実行（順序・割当は従来の round-robin と同じ）
USE_DISTRIBUTED_EVAL = os.environ.get("DISTRIBUTED_PCN_DISTRIBUTED_EVAL", "1") == "1"

BATCH_SIZE = 2048  # 実験規模を変えない（GPU 時も同じ）
N_UPDATES = 100  # 学習更新回数（3に減らすと高速化、精度はやや低下の可能性）
LEARNING_RATE = 1e-2

# 学習率減衰(Phase3限定): 続学習の後半ノイズ蓄積で効率方策が壊れるのを、終端で lr を下げて抑える。
# iteration割合ベース=スケール非依存。"0"/"off"=OFF。Learner内部カウンタで learn 実行順に減衰(async index ずれ回避)。
_LR_DECAY = os.environ.get("PCN_LR_DECAY", "0")
_LR_DECAY_ON = _LR_DECAY not in ("0", "", "off")
_LR_DECAY_FINAL = float(os.environ.get("PCN_LR_DECAY_FINAL", "0.1"))


def _lr_scale_for_iter(it, n_iter):
    """Phase3 iteration(0-based) に対する lr 倍率(1.0 → _LR_DECAY_FINAL)。"""
    if not _LR_DECAY_ON or n_iter <= 1:
        return 1.0
    frac = min(1.0, max(0.0, it / (n_iter - 1)))
    f0, f1 = 1.0, _LR_DECAY_FINAL
    if _LR_DECAY == "linear":
        return f0 + (f1 - f0) * frac
    import math
    return f1 + 0.5 * (f0 - f1) * (1.0 + math.cos(math.pi * frac))  # cosine(既定)


# PF点学習重みの段階的スケジュール: 前半は安定に広く学び、後半でPF点(膝=希薄点)に学習を集中。
# 質(膝深さ)と安定(崩壊回避)のトレードオフを時間分離で突破する狙い。"off"で従来挙動。
_PF_WEIGHT_SCHED = os.environ.get("PCN_TRAIN_PF_WEIGHT_SCHED", "off")
_PF_WEIGHT_PEAK_MUL = float(os.environ.get("PCN_TRAIN_PF_WEIGHT_PEAK_MUL", "1.0"))      # 後半の倍率
_PF_WEIGHT_SCHED_START = float(os.environ.get("PCN_TRAIN_PF_WEIGHT_SCHED_START", "0.5"))  # 集中開始frac


def _pf_weight_mul_for_iter(it, n_iter):
    """Phase3 iter(0-based) のPF点重み倍率: 前半1.0(安定), START以降 1→PEAK_MUL(膝に集中)。"""
    if _PF_WEIGHT_SCHED in ("off", "", "0") or n_iter <= 1 or _PF_WEIGHT_PEAK_MUL <= 1.0:
        return 1.0
    frac = min(1.0, max(0.0, it / (n_iter - 1)))
    if frac < _PF_WEIGHT_SCHED_START:
        return 1.0
    if _PF_WEIGHT_SCHED == "step":
        return _PF_WEIGHT_PEAK_MUL
    f = (frac - _PF_WEIGHT_SCHED_START) / max(1e-9, 1.0 - _PF_WEIGHT_SCHED_START)
    return 1.0 + (_PF_WEIGHT_PEAK_MUL - 1.0) * f  # ramp(既定)


EARLY_STOPPING_PATIENCE = 5  # 改善が見られないイテレーション数
EARLY_STOPPING_THRESHOLD = 0.0001  # 改善とみなす最小変化量
MIN_ITERATIONS = 5  # 最低限実行するイテレーション数


INITIAL_EPISODES = 100  # フェーズ1: 各Actorあたりのランダム収集エピソード数（config.yml で上書き可）

USE_ENHANCED_MODEL = False  # True: EnhancedPCNModel, False: DiscreteActionsDefaultModel (3層NLPモデル)


SUPERVISED_LEARNING_EPOCHS = 200
SUPERVISED_BATCH_SIZE = 2048    
SUPERVISED_UPDATES_PER_EPOCH = 100
SUPERVISED_LEARNING_RATE = 1e-3

VISUALIZATION_INTERVAL = 5  # 可視化を実行する間隔（イテレーション数）

EPISODES_PER_ITERATION = 1  # 各イテレーションで各Actorが生成するエピソード数

EVAL_SAMPLES = int(os.environ.get("DISTRIBUTED_PCN_EVAL_SAMPLES", "500"))  # 評価command数の上限（実実行はPF上のユニーク数まで）
EVAL_SAMPLES_DISTRIBUTED = 0  # 分散評価時に使用するサンプル数(最近使わない)
EVAL_SAMPLES_FINAL = int(os.environ.get("DISTRIBUTED_PCN_EVAL_SAMPLES_FINAL", str(EVAL_SAMPLES)))
EVAL_SAMPLES_VISUALIZATION = int(
    os.environ.get("DISTRIBUTED_PCN_EVAL_SAMPLES_VIS", str(EVAL_SAMPLES))
)  # 可視化専用の再評価（EVALと同intervalなら再利用するため通常は走らない）
# 可視化: Archive の全点は描かない（PFのみ）。報酬空間プロット・command矢印は既定OFFで高速化
_ARCHIVE_VIS_ALL = os.environ.get("DISTRIBUTED_PCN_ARCHIVE_VIS_ALL", "0") == "1"
_VIS_REWARD_PLOT = os.environ.get("DISTRIBUTED_PCN_VIS_REWARD_PLOT", "0") == "1"
_VIS_COMMAND_ARROWS = os.environ.get("DISTRIBUTED_PCN_VIS_COMMAND_ARROWS", "0") == "1"
_SKIP_FINAL_EVAL = os.environ.get("DISTRIBUTED_PCN_SKIP_FINAL_EVAL", "1") == "1"
_VIS_PLOT_DPI = int(os.environ.get("DISTRIBUTED_PCN_VIS_DPI", "90"))

# プロファイリング用: 環境変数で短時間実行モードを有効化
_PROFILE_MODE = os.environ.get('DISTRIBUTED_PCN_PROFILE', '0') == '1'
_QUICK_MODE = os.environ.get('DISTRIBUTED_PCN_QUICK', '0') == '1'
_ABLATION_MODE = os.environ.get('DISTRIBUTED_PCN_ABLATION', '0') == '1'
# Actor-Learner非同期オーバーラップ（Learner(i)とActor(i+1)を並列実行して待ち時間を隠蔽）
_ASYNC_OVERLAP = os.environ.get('DISTRIBUTED_PCN_ASYNC_OVERLAP', '1') == '1'
# Phase3 ロールアウト重み配布を共有ObjectRef化（"1"=ref / "0"=従来 materialize）。
# Learner actor のメソッドキュー順序(learn→重み取得)は変えず、materialize した state_dict の代わりに
# 同一中身の ObjectRef を1回putして全Actorで共有受信する（転送量を Actor 数分→1回に削減）。
_ACTOR_WEIGHTS_REF = os.environ.get('PCN_ACTOR_WEIGHTS_REF', '1') == '1'
_PHASE3_GPU_CACHE = os.environ.get('DISTRIBUTED_PCN_PHASE3_GPU_CACHE', '1') == '1'
_PHASE2_IMPORTANCE = os.environ.get('DISTRIBUTED_PCN_PHASE2_IMPORTANCE', '1') == '1'
_PHASE2_IMPORTANCE_SAMPLES = int(os.environ.get('DISTRIBUTED_PCN_PHASE2_IMPORTANCE_SAMPLES', '1024'))
# Phase2の教師あり学習は初期エピソードの行動を模倣するため、完全なIIDランダム行動だけだと
# ラベルに学習可能な構造がなく、2値NLLがlog(2)付近で止まりやすい。既存の比率スイープを既定にする。
_INITIAL_ACTION_SWEEP = os.environ.get('DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP', '1') == '1'
_INITIAL_ACTION_SWEEP_PROBS = tuple(
    float(x.strip())
    for x in os.environ.get(
        'DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP_PROBS',
        '0,0.1,0.25,0.5,0.75,0.9,1',
    ).split(',')
    if x.strip()
)
# 高速化モード: N_UPDATESを3に削減（本番でも有効、DISTRIBUTED_PCN_FAST=1）
_FAST_MODE = os.environ.get('DISTRIBUTED_PCN_FAST', '0') == '1'
_USE_JAX_LEARNER = os.environ.get('DISTRIBUTED_PCN_USE_JAX', '0') == '1'
# 既定: Cビットマップ観測（環境もNN入力も SchedulingEnvCacheOptimized）
# イベント観測へ切替: import 前に DISTRIBUTED_PCN_USE_EVENT_OBS=1、または python -m src.distributed.distributed_pcn_event
_USE_EVENT_OBS = os.environ.get('DISTRIBUTED_PCN_USE_EVENT_OBS', '0') == '1'
# イベント観測 ON 時は step もイベントネイティブ（time_transition ループなし）を既定
_USE_EVENT_NATIVE = os.environ.get(
    'DISTRIBUTED_PCN_USE_EVENT_NATIVE',
    '1' if _USE_EVENT_OBS else '0',
) == '1'
if _QUICK_MODE:
    # 短時間デバッグ用（本番実験規約の 100/100 ではない）
    N_ITERATIONS = 20
    N_ACTORS = 12
    INITIAL_EPISODES = 30
    EPISODES_PER_ITERATION = 2
    EVAL_INTERVAL = 5
    SUPERVISED_LEARNING_EPOCHS = 10
    # N_UPDATESは変更しない（ハイパラメータ変更は高速化ではない）
    ENABLE_VISUALIZATION = True
    print("[PROFILE] クイックモード（デバッグ）: N_ITERATIONS=5, N_ACTORS=12, INITIAL_EPISODES=100")
elif _ABLATION_MODE:
    # conditioning アブレーション用（本番より短いが QUICK より Phase2/3 を確保）
    N_ITERATIONS = int(os.environ.get("DISTRIBUTED_PCN_ABLATION_N_ITER", "25"))
    N_ACTORS = int(os.environ.get("DISTRIBUTED_PCN_ABLATION_N_ACTORS", "8"))
    INITIAL_EPISODES = int(os.environ.get("DISTRIBUTED_PCN_ABLATION_INITIAL_EP", "40"))
    EPISODES_PER_ITERATION = 1
    EVAL_INTERVAL = 5
    SUPERVISED_LEARNING_EPOCHS = int(os.environ.get("DISTRIBUTED_PCN_ABLATION_SUP_EPOCHS", "40"))
    SUPERVISED_UPDATES_PER_EPOCH = int(os.environ.get("DISTRIBUTED_PCN_ABLATION_SUP_UPDATES", "50"))
    EVAL_SAMPLES = int(os.environ.get("DISTRIBUTED_PCN_ABLATION_EVAL_SAMPLES", "64"))
    ENABLE_VISUALIZATION = False
    print(
        f"[ABLATION] N_ITER={N_ITERATIONS} N_ACTORS={N_ACTORS} "
        f"INITIAL_EP={INITIAL_EPISODES} SUP_EPOCHS={SUPERVISED_LEARNING_EPOCHS} "
        f"EVAL_SAMPLES={EVAL_SAMPLES}"
    )
elif _FAST_MODE:
    # 削除: N_UPDATES変更はハイパラメータ変更のため高速化に含めない
    pass

# ベンチマーク等: DISTRIBUTED_PCN_ENABLE_VISUALIZATION=0 で最終可視化・一部プロットを省略
_viz_env = os.environ.get("DISTRIBUTED_PCN_ENABLE_VISUALIZATION", "").strip().lower()
if _viz_env in ("0", "false", "no"):
    ENABLE_VISUALIZATION = False
elif _viz_env in ("1", "true", "yes"):
    ENABLE_VISUALIZATION = True

from src.agents.pcn_agent import (
    PCN, 
    Transition, 
    get_non_dominated_inds, 
    get_non_dominated_inds_minimize,
    crowding_distance,
    hypervolume
)


def _make_timestamped_run_dir(root_dir: str, prefix: str = "") -> str:
    """Create a lexicographically sortable run directory under root_dir."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}{prefix}"
    run_dir = os.path.join(root_dir, base_name)
    if not os.path.exists(run_dir):
        return run_dir

    suffix = 1
    while True:
        candidate = os.path.join(root_dir, f"{base_name}_{suffix:02d}")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


class _TeeStream:
    """Mirror stdout/stderr to a run log while keeping terminal output."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        written = self._stream.write(data)
        self._log_file.write(data)
        return written

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._stream.isatty()

    @property
    def encoding(self):
        return self._stream.encoding

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _to_json_list(value) -> List[float]:
    return np.asarray(value, dtype=np.float64).ravel().tolist()


def _summarize_episode_for_log(actor_id: int, episode_index: int, episode, random_action_prob=None) -> Dict[str, Any]:
    actions = [int(t.action) for t in episode]
    action_counts = {
        str(int(action)): int(count)
        for action, count in zip(*np.unique(actions, return_counts=True))
    } if actions else {}
    rewards = [np.asarray(t.reward, dtype=np.float64) for t in episode]
    total_reward = np.sum(rewards, axis=0) if rewards else np.zeros(2, dtype=np.float64)
    first = episode[0] if episode else None
    objective_values = (
        _to_json_list(first.objective_values)
        if first is not None and hasattr(first, "objective_values")
        else None
    )
    return {
        "actor_id": int(actor_id),
        "actor_episode_index": int(episode_index),
        "random_action_prob": None if random_action_prob is None else float(random_action_prob),
        "episode_length": int(len(episode)),
        "total_reward": _to_json_list(total_reward),
        "objective_values": objective_values,
        "action_counts": action_counts,
        "actions": actions,
    }


def _find_initial_episode_log(output_base: str, execution_dir: str) -> Optional[str]:
    candidates = []
    current_dir = os.path.abspath(execution_dir)
    if not os.path.isdir(output_base):
        return None
    for root, _, files in os.walk(output_base):
        if "initial_episodes_first100.jsonl" not in files:
            continue
        path = os.path.join(root, "initial_episodes_first100.jsonl")
        if os.path.abspath(root) == current_dir:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _find_initial_episode_cache(output_base: str, execution_dir: str) -> Optional[str]:
    candidates = []
    current_dir = os.path.abspath(execution_dir)
    if not os.path.isdir(output_base):
        return None
    for root, _, files in os.walk(output_base):
        if "initial_episodes_cache.pkl.gz" not in files:
            continue
        path = os.path.join(root, "initial_episodes_cache.pkl.gz")
        if os.path.abspath(root) == current_dir:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_initial_episode_log(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_loaded_initial_episode_summary(execution_dir: str, source_path: str, rows: List[Dict[str, Any]]) -> str:
    lengths = [int(row.get("episode_length", 0)) for row in rows]
    action0 = sum(int(row.get("action_counts", {}).get("0", 0)) for row in rows)
    action1 = sum(int(row.get("action_counts", {}).get("1", 0)) for row in rows)
    summary_path = os.path.join(execution_dir, "initial_episodes_loaded_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_path": source_path,
                "loaded_episodes": len(rows),
                "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
                "episode_length_min": int(np.min(lengths)) if lengths else 0,
                "episode_length_max": int(np.max(lengths)) if lengths else 0,
                "action_counts": {"0": int(action0), "1": int(action1)},
            },
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    return summary_path


from src.envs.scheduling_env import SchedulingEnv
from src.envs.scheduling_variants.bitmap_c_env import SchedulingEnvCacheOptimized
from src.envs.scheduling_variants.event_c_env import (
    EVENT_FEATURES,
    JOB_QUEUE_SIZE,
    N_EVENTS_OBS,
    SchedulingEnvEventObs,
)
from src.envs.scheduling_variants.event_native_env import SchedulingEnvEventNative
from src.utils.event_obs_bitmap_adapter import (
    learner_bitmap_enabled,
    apply_learner_bitmap_to_event_env,
    bitmap_flat_dim_from_event_env,
    event_obs_to_bitmap_observation,
)
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.anchor_residual import get_anchor_set
from src.utils.algorithm_compare_config import get_param_algorithm_compare


def _hypervolume_2d_min(nd_pts, ref):
    """2目的 minimize 空間の達成front HV(参照点 ref からの体積)。nd_pts=非支配点[N,2], ref=nadir。
    early-stop で「広さ＋効率」を1値で測り、達成HV最良の学習中ckptを選ぶために使う。"""
    front = [(float(c), float(w)) for c, w in nd_pts if c < ref[0] and w < ref[1]]
    if not front:
        return 0.0
    hv = 0.0
    prev = float(ref[0])
    for c, w in sorted(front, key=lambda x: -x[0]):
        hv += (prev - c) * (float(ref[1]) - w)
        prev = c
    return hv

# 使用する環境クラス（イベント観測: EventNative=イベント駆動 step / EventObs=旧ビットマップ step）
if _USE_EVENT_OBS:
    _EnvClass = (
        SchedulingEnvEventNative
        if _USE_EVENT_NATIVE
        else SchedulingEnvEventObs
    )
else:
    _EnvClass = SchedulingEnvCacheOptimized

# イベント観測環境使用時のみ、NN入力をラーナー側でビットマップへ復元可能（SCHEDULER_LEARNER_BITMAP / DISTRIBUTED_PCN_EVENT_TO_BITMAP）
if _USE_EVENT_OBS:
    _step_mode = (
        "イベントネイティブ step（time_transition なし）"
        if _USE_EVENT_NATIVE
        else "ビットマップ step（SchedulingEnvCacheOptimized 継承）"
    )
    if learner_bitmap_enabled():
        print(
            f"[ENV] イベント観測 + {_step_mode}: Actor→ReplayBuffer はイベントベクトル / "
            "Learner が学習データをビットマップへ復元（Ray 転送はイベント側が細い）"
        )
    else:
        print(
            f"[ENV] イベント観測 + {_step_mode}（NNはイベントベクトルのみ、ビットマップ復元OFF）"
        )
else:
    print("[ENV] Cビットマップ観測（環境・ラーナーともbitmap）")

_LOG_GPU_MEM = os.environ.get("DISTRIBUTED_PCN_LOG_GPU_MEM", "0") == "1"
_REPLAY_ZERO_COPY = os.environ.get("DISTRIBUTED_PCN_REPLAY_ZERO_COPY", "1") == "1"
_ACTOR_RAY_PUT = os.environ.get("DISTRIBUTED_PCN_ACTOR_RAY_PUT", "1") == "1"
_LOG_RAY_TRANSFER = os.environ.get("DISTRIBUTED_PCN_LOG_RAY_TRANSFER", "0") == "1"
_GC_EACH_ITER = os.environ.get("DISTRIBUTED_PCN_GC_EACH_ITER", "0") == "1"
_VIZ_LIGHT = os.environ.get("DISTRIBUTED_PCN_VIZ_LIGHT", "1") == "1"
_COLLECT_CMD_OUTCOMES = (
    os.environ.get("DISTRIBUTED_PCN_CMD_OUTCOMES", "").strip() == "1"
    or (ENABLE_VISUALIZATION and not _VIZ_LIGHT)
)
_EVAL_QUIET = os.environ.get("DISTRIBUTED_PCN_EVAL_QUIET", "1") == "1"
# [元PCN復元] desired_return の上限クリップ(元実装の max_return)。報酬負だと dr が使うほど0へ→0を突き抜けて
# 正へ漂流し「達成不能な指令」になる。元PCNは np.clip(dr-reward, None, max_return) で上限を達成可能域に制限。
# 私の実装はこれを削除(dr-=reward のみ)していた。PCN_DESIRED_RETURN_UB="0" で上限0(達成可能=報酬負の上限)。
# 空=無効(従来=クリップ無し=ビット一致)。
_DESIRED_RETURN_UB_RAW = os.environ.get("PCN_DESIRED_RETURN_UB", "")
_DESIRED_RETURN_UB = float(_DESIRED_RETURN_UB_RAW) if _DESIRED_RETURN_UB_RAW != "" else None
# [anti-ration] cost成分の desired_return を decrement せず初期目標で一定保持(actor探索側)。既定0=ビット一致。
# trace高cost端飽和の真因 rationing(残予算↓でcloud停止)対策。eval側 pcn_agent._run_episode にも同名gate。
_COST_HOLD = os.environ.get("PCN_COST_HOLD", "0") == "1"


def _estimate_episodes_numpy_bytes(episodes) -> int:
    """Transition 内 numpy の nbytes 合計（Actor→Buffer→Learner の Ray 転送ペイロード目安。pickle オーバーヘッドは含まない）。"""
    total = 0
    if not episodes:
        return 0
    for episode in episodes:
        if not episode:
            continue
        for t in episode:
            for attr in ("observation", "next_observation", "reward"):
                v = getattr(t, attr, None)
                if isinstance(v, np.ndarray):
                    total += int(v.nbytes)
                elif hasattr(v, "nbytes"):
                    total += int(v.nbytes)
            act = getattr(t, "action", None)
            if isinstance(act, np.ndarray):
                total += int(act.nbytes)
            elif isinstance(act, (int, np.integer)):
                total += 8
    return total


def _log_gpu_memory_snapshot(tag: str) -> None:
    """PyTorch CUDA と nvidia-smi の実測値を標準出力へ（bitmap / event 比較用）。"""
    if not _LOG_GPU_MEM:
        return
    if not th.cuda.is_available():
        print(f"[GPU_MEM] {tag} | CUDA 利用不可（CPU 実行）")
        return
    dev = 0
    try:
        th.cuda.synchronize(dev)
    except Exception:
        pass
    alloc = th.cuda.memory_allocated(dev) / (1024 ** 2)
    reserved = th.cuda.memory_reserved(dev) / (1024 ** 2)
    peak = th.cuda.max_memory_allocated(dev) / (1024 ** 2)
    print(
        f"[GPU_MEM] {tag} | torch.cuda allocated={alloc:.1f} MB "
        f"reserved={reserved:.1f} MB peak_allocated={peak:.1f} MB"
    )
    try:
        import subprocess

        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        lines = (proc.stdout or "").strip().splitlines()
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 2:
                print(
                    f"[GPU_MEM] {tag} | nvidia-smi gpu_used={parts[0]} MiB "
                    f"gpu_total={parts[1]} MiB"
                )
    except Exception:
        pass


def _enable_event_bitmap_adapter(env):
    """イベント観測環境の get_observation をビットマップ復元版へ差し替える（共通ユーティリティ）。"""
    if not _USE_EVENT_OBS:
        return env
    return apply_learner_bitmap_to_event_env(env)


def _eval_max_return(reward_dim: int = 2) -> np.ndarray:
    """評価時の desired_return 上限。学習 Actor はクリップしないため inf（旧既定 100 は桁が合わず command が潰れる）。"""
    return np.full(reward_dim, np.inf, dtype=np.float32)


def _distributed_evaluate_episodes(learner, actors, n: int):
    """Driver 上で分散評価（Learner 内から Actor→Learner.get_weights を呼ぶとデッドロックするため）。"""
    max_return = _eval_max_return()
    targets = ray.get(learner.get_eval_targets.remote(n))
    if not targets:
        return [], [], [], None
    weights_ref = ray.get(learner.get_eval_weights_ref.remote())
    futures = []
    for i, (desired_return, desired_horizon) in enumerate(targets):
        actor = actors[i % len(actors)]
        futures.append(
            actor.evaluate_episode.remote(
                desired_return, desired_horizon, max_return, weights_ref
            )
        )
    results = ray.get(futures)
    return ray.get(learner.ingest_distributed_eval_results.remote(results))


def _driver_eval_gap_feedback(learner, actors, training_iteration, plot_dir, n_jobs: int):
    """Eval ギャップ FB を Driver で実行（格子評価は Actor のみ、Learner は準備と boost 適用のみ）。"""
    from src.utils.pf_eval_gap import (
        _uniform_grid_commands,
        _run_uniform_grid_commands,
        summarize_band_gaps,
        gap_band_boosts,
        gap_bands_from_env,
        _maybe_write_live_uniform_pf,
        get_non_dominated_inds_minimize,
    )

    g = int(os.environ.get("PCN_EVAL_GAP_FEEDBACK_GRID", "12"))
    prep = ray.get(learner.prepare_uniform_grid_prep.remote(n_jobs, g))
    ref_pts = np.asarray(prep.get("ref_pts", []), dtype=np.float64)
    exploration = np.asarray(prep.get("exploration", []), dtype=np.float64)
    commands = [
        (np.asarray(dr, dtype=np.float32), float(hz))
        for dr, hz in prep.get("commands", [])
    ]
    weights_ref = ray.get(learner.get_eval_weights_ref.remote())
    pts = _run_uniform_grid_commands(
        None, None, commands, actors=actors, weights_ref=weights_ref
    )
    nd = get_non_dominated_inds_minimize(pts)
    eval_pf = pts[nd] if len(nd) else pts
    band_list = gap_bands_from_env()
    summary = summarize_band_gaps(eval_pf, ref_pts, band_list)
    boosts = gap_band_boosts(summary, band_list)
    _maybe_write_live_uniform_pf(
        plot_dir=plot_dir,
        plot_iteration=training_iteration,
        plot_label=os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LABEL", "live"),
        all_pts=pts,
        eval_pf=eval_pf,
        archive_pf=ref_pts,
        exploration=exploration,
    )
    ray.get(learner.apply_eval_gap_boosts.remote(boosts))
    if boosts:
        parts = [f"{lo:.0g}-{hi:.0g}x{mult:.2f}" for lo, hi, mult in boosts]
        print(f"[EVAL_GAP] iter={training_iteration} boost_bands=[{', '.join(parts)}]")
    return summary


def _driver_live_uniform_pf_plot(learner, actors, training_iteration, plot_dir, n_jobs: int):
    """LIVE PF 図のみ（Eval ギャップ FB なし）を Driver 経由で保存。"""
    from src.utils.pf_eval_gap import (
        _uniform_grid_commands,
        _run_uniform_grid_commands,
        _maybe_write_live_uniform_pf,
        get_non_dominated_inds_minimize,
    )
    from src.utils.pf_uniform_plot import live_uniform_pf_grid

    g = live_uniform_pf_grid()
    prep = ray.get(learner.prepare_uniform_grid_prep.remote(n_jobs, g))
    ref_pts = np.asarray(prep.get("ref_pts", []), dtype=np.float64)
    exploration = np.asarray(prep.get("exploration", []), dtype=np.float64)
    commands = [
        (np.asarray(dr, dtype=np.float32), float(hz))
        for dr, hz in prep.get("commands", [])
    ]
    weights_ref = ray.get(learner.get_eval_weights_ref.remote())
    pts = _run_uniform_grid_commands(
        None, None, commands, actors=actors, weights_ref=weights_ref
    )
    nd = get_non_dominated_inds_minimize(pts)
    eval_pf = pts[nd] if len(nd) else pts
    return _maybe_write_live_uniform_pf(
        plot_dir=plot_dir,
        plot_iteration=training_iteration,
        plot_label=os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LABEL", "live"),
        all_pts=pts,
        eval_pf=eval_pf,
        archive_pf=ref_pts,
        exploration=exploration,
    )

# =========================
# 1. Replay Buffer (Ray Actor)
# =========================
@ray.remote
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.episode_hashes = set()  # 重複検出用のハッシュセット
        self._hash_cache = {}  # エピソードのハッシュ値キャッシュ（idをキーとして使用）
        if DEBUG:
            print(f"ReplayBuffer initialized with max_size={max_size}")

    def add(self, episode):
        # エピソードのハッシュ値を計算
        episode_hash = self._compute_episode_hash(episode)
        
        # 重複チェック
        if episode_hash in self.episode_hashes:
            if DEBUG:
                print(f"ReplayBuffer: 重複エピソードをスキップしました。ハッシュ: {episode_hash}")
            return
        
        # バッファサイズチェック
        if len(self.buffer) >= self.max_size:
            # 最も古いエピソードのハッシュを削除
            oldest_episode = self.buffer.pop(0)
            oldest_hash = self._compute_episode_hash(oldest_episode)
            self.episode_hashes.discard(oldest_hash)
            # キャッシュからも削除
            oldest_episode_id = id(oldest_episode)
            self._hash_cache.pop(oldest_episode_id, None)
        
        # 新しいエピソードを追加
        self.buffer.append(episode)
        self.episode_hashes.add(episode_hash)
        
        # ログ出力を簡潔にする（100エピソードごとに表示）
        if DEBUG and len(self.buffer) % 100 == 0:
            print(f"ReplayBuffer: episode added, current size={len(self.buffer)}")
    
    def add_batch(self, episodes):
        """複数のエピソードを一度に追加（シリアライゼーション最適化）"""
        added_count = 0
        skipped_count = 0
        
        for episode in episodes:
            # エピソードのハッシュ値を計算
            episode_hash = self._compute_episode_hash(episode)
            
            # 重複チェック
            if episode_hash in self.episode_hashes:
                skipped_count += 1
                continue
            
            # バッファサイズチェック
            if len(self.buffer) >= self.max_size:
                # 最も古いエピソードのハッシュを削除
                oldest_episode = self.buffer.pop(0)
                oldest_hash = self._compute_episode_hash(oldest_episode)
                self.episode_hashes.discard(oldest_hash)
                # キャッシュからも削除
                oldest_episode_id = id(oldest_episode)
                self._hash_cache.pop(oldest_episode_id, None)
            
            # 新しいエピソードを追加
            self.buffer.append(episode)
            self.episode_hashes.add(episode_hash)
            added_count += 1
        
        if DEBUG:
            print(f"ReplayBuffer: バッチ追加完了 - 追加: {added_count}, スキップ: {skipped_count}, 現在のサイズ: {len(self.buffer)}")
        
        return added_count

    def _compute_episode_hash(self, episode):
        """エピソードの内容に基づくハッシュ値を計算（軽量版）"""
        import hashlib
        
        if not episode:
            return 0
        
        # キャッシュチェック（エピソードのidをキーとして使用）
        episode_id = id(episode)
        if episode_id in self._hash_cache:
            return self._hash_cache[episode_id]
        
        # エピソードを一意に識別する要約情報のみを使用
        hasher = hashlib.md5()
        
        # 1. エピソードの長さ
        episode_len = len(episode)
        hasher.update(episode_len.to_bytes(8, byteorder='big'))

        episode_uid = getattr(episode[0], '_pcn_episode_uid', None)
        if episode_uid is not None:
            hasher.update(str(episode_uid).encode())
        
        # 2. 最初の観測の要約（最初の数要素のみ、またはハッシュ）
        first_obs = episode[0].observation
        if hasattr(first_obs, 'tobytes'):
            # 観測が大きい場合は最初の一部のみを使用
            obs_summary = first_obs.flatten()[:min(100, first_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(first_obs).encode())
        
        # 3. 行動のシーケンス（効率的にバイト列として結合）
        actions = np.array([t.action for t in episode], dtype=np.int32)
        hasher.update(actions.tobytes())
        
        # 4. 報酬の要約（合計と平均）
        rewards = np.array([t.reward for t in episode])
        if rewards.size > 0:
            reward_summary = np.array([rewards.sum(), rewards.mean()], dtype=np.float32)
            hasher.update(reward_summary.tobytes())
        
        # 5. 最後の観測の要約
        last_obs = episode[-1].next_observation
        if hasattr(last_obs, 'tobytes'):
            obs_summary = last_obs.flatten()[:min(100, last_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(last_obs).encode())
        
        # 6. ターミナル状態の情報
        terminal_info = np.array([t.terminal for t in episode], dtype=bool)
        hasher.update(terminal_info.tobytes())
        
        # ハッシュ値を計算（intに変換）
        hash_value = int(hasher.hexdigest(), 16)
        
        # キャッシュに保存（エピソードのidをキーとして使用）
        self._hash_cache[episode_id] = hash_value
        
        return hash_value

    def get_all_episodes(self):
        """全てのエピソードを取得してバッファをクリア（シリアライゼーション最適化）"""
        # 深いコピーを作成して、元のオブジェクトとの参照を完全に分離
        # ただし、観測データが既にfloat32の場合は変換しない（メモリコピーを避ける）
        result = []
        for episode in self.buffer:
            # エピソードの各Transitionを軽量化
            optimized_episode = []
            for t in episode:
                # 観測データをfloat32に変換（既にfloat32の場合は変換しない）
                obs = t.observation
                if hasattr(t.observation, 'dtype') and t.observation.dtype != np.float32:
                    obs = np.array(t.observation, dtype=np.float32, copy=True)
                elif hasattr(t.observation, 'copy'):
                    obs = t.observation.copy()  # 参照を分離するためコピー
                
                next_obs = t.next_observation
                if hasattr(t.next_observation, 'dtype') and t.next_observation.dtype != np.float32:
                    next_obs = np.array(t.next_observation, dtype=np.float32, copy=True)
                elif hasattr(t.next_observation, 'copy'):
                    next_obs = t.next_observation.copy()  # 参照を分離するためコピー
                
                reward = t.reward
                if hasattr(t.reward, 'dtype') and t.reward.dtype != np.float32:
                    reward = np.array(t.reward, dtype=np.float32, copy=True)
                elif hasattr(t.reward, 'copy'):
                    reward = t.reward.copy()  # 参照を分離するためコピー
                
                optimized_transition = Transition(
                    observation=obs,
                    action=t.action,
                    reward=reward,
                    next_observation=next_obs,
                    terminal=t.terminal
                )
                # 追加の属性もコピー
                if hasattr(t, 'objective_values'):
                    optimized_transition.objective_values = t.objective_values
                if hasattr(t, 'solution_execution_time'):
                    optimized_transition.solution_execution_time = t.solution_execution_time
                    optimized_transition.command_return = getattr(t, "command_return", None)  # [案1] 指令もコピー保持
                if hasattr(t, '_pcn_episode_uid'):
                    optimized_transition._pcn_episode_uid = t._pcn_episode_uid
                if hasattr(t, 'random_action_prob'):
                    optimized_transition.random_action_prob = t.random_action_prob
                
                optimized_episode.append(optimized_transition)
            result.append(optimized_episode)
        
        self.buffer.clear()
        self.episode_hashes.clear()  # ハッシュセットもクリア
        self._hash_cache.clear()  # ハッシュキャッシュもクリア
        if DEBUG:
            print(f"ReplayBuffer: retrieved all {len(result)} episodes and cleared buffer")
        return result

    def take_all_episodes(self):
        """バッファ内エピソードを所有権ごと返しクリア（深いコピーなし）。"""
        result = self.buffer
        self.buffer = []
        self.episode_hashes.clear()
        self._hash_cache.clear()
        if DEBUG:
            print(f"ReplayBuffer: took {len(result)} episodes (zero-copy) and cleared buffer")
        return result

    def add_batch_ref(self, episodes_ref):
        """Object Store 上のエピソード参照を1回だけ materialize して追加。"""
        if isinstance(episodes_ref, ray.ObjectRef):
            episodes = ray.get(episodes_ref)
        else:
            episodes = episodes_ref
        return self.add_batch(episodes)

    def save_to_file(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "episodes": self.buffer,
        }
        # compresslevel=6: 9→6で保存 ~4倍速・サイズ +2〜3%のみ（展開後の中身は同一）。
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        nbytes = _estimate_episodes_numpy_bytes(self.buffer)
        return {
            "path": path,
            "episodes": len(self.buffer),
            "transitions": sum(len(ep) for ep in self.buffer),
            "numpy_payload_bytes": int(nbytes),
        }

    def load_from_file(self, path: str):
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
        self.buffer.clear()
        self.episode_hashes.clear()
        self._hash_cache.clear()
        added = self.add_batch(episodes)
        return {
            "path": path,
            "episodes": len(self.buffer),
            "episodes_added": int(added),
            "transitions": sum(len(ep) for ep in self.buffer),
            "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
        }

    def size(self):
        return len(self.buffer)
    
    def get_stats(self):
        """バッファの統計情報を取得"""
        return {
            'buffer_size': len(self.buffer),
            'unique_episodes': len(self.episode_hashes),
            'max_size': self.max_size,
            'utilization': len(self.buffer) / self.max_size if self.max_size > 0 else 0
        }

# =========================
# 2. Actor (Ray Actor)
# =========================
@ray.remote
class Actor:
    def __init__(self, config, learner, buffer, actor_id=0):
        self.config = config
        self.learner = learner
        self.buffer = buffer
        self.actor_id = actor_id
        self.env = None
        self.agent = None
        self._weights_ref = None  # 重みのObjectRefを保持
        if DEBUG:
            print(f"Actor {actor_id} initialized")

    def _get_available_device(self, requested_device):
        """利用可能なデバイスを検出（CUDAが確実に存在する前提）"""
        import torch
        
        if requested_device == 'cuda':
            # CUDAが確実に存在する前提でCUDAを返す
            if DEBUG:
                print(f"Actor {self.actor_id}: Using CUDA device.")
            return 'cuda'
        else:
            if DEBUG:
                print(f"Actor {self.actor_id}: Using requested device: {requested_device}")
            return requested_device

    def _make_env(self):
        if self.env is None:
            n_jobs = self.config['param_env'].get('n_jobs', N_JOBS)
            job_generator = JobGenerator(
                0, 1,
                self.config['param_env']['n_window'],
                self.config['param_env']['n_on_premise_node'],
                self.config['param_env']['n_cloud_node'],
                self.config, n_jobs, 0.2, 0
            )
            jobs_set = job_generator.generate_jobs_set()
            self.env = _EnvClass(
                np.inf,
                self.config['param_env']['n_window'],
                self.config['param_env']['n_on_premise_node'],
                self.config['param_env']['n_cloud_node'],
                self.config['param_env']['n_job_queue_obs'],
                self.config['param_env']['n_job_queue_bck'],
                self.config['param_agent']['weight_wt'],
                self.config['param_agent']['weight_cost'],
                self.config['param_env']['penalty_not_allocate'],
                self.config['param_env']['penalty_invalid_action'],
                jobs_set,
                None, flag=0
            )
            # Learner 側でイベント→bitmap 復元する既定では、Actor は生イベント観測のまま収集（転送削減）
            if _USE_EVENT_OBS and learner_bitmap_enabled():
                self.env._pcn_raw_event_obs_for_transfer = True
            else:
                self.env = _enable_event_bitmap_adapter(self.env)
            # C実装が正しく使用されているか確認
            # if hasattr(self.env, '_cache_onpre_c'):
            #     print(f"[Actor {self.actor_id}] ✓ C実装環境が正しく初期化されました")
            # else:
            #     print(f"[Actor {self.actor_id}] ⚠️ C実装環境の初期化に問題があります")
            
            # Actorは常にCPUで実行（Learnerとの互換性のため）
            actual_device = 'cpu'
            
            if _USE_EVENT_OBS and learner_bitmap_enabled():
                _actor_state_dim = bitmap_flat_dim_from_event_env(self.env)
            else:
                _actor_state_dim = self.env.observation_space.shape[0]
            
            # PCNエージェントの初期化（CPUで実行）
            # horizon スケールは Learner と一致させる（重み同期で上書きされるが初期値も揃える）
            _h_scale = 1.0 / max(1, int(self.config['param_env'].get('n_jobs', N_JOBS)))
            self.agent = PCN(
                self.env,
                device=actual_device,
                state_dim=_actor_state_dim,
                scaling_factor=np.array([1.0, 1.0, _h_scale]),
                learning_rate=LEARNING_RATE,
                batch_size=512,
                hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
                project_name="temp",
                experiment_name="PCN",
                log=False,
                debug_mode=DEBUG,  # DEBUGフラグを追加
                use_enhanced_model=USE_ENHANCED_MODEL,  # モデル選択
            )
            if DEBUG:
                print(f"Actor {self.actor_id} environment and agent initialized with device: {actual_device}")
                print(f"Actor {self.actor_id} observation space: {self.env.observation_space.shape}")
                print(f"Actor {self.actor_id} action space: {self.env.action_space}")
                print(f"Actor {self.actor_id} reward space: {self.env.reward_space.shape}")
                print(f"Actor {self.actor_id} model: {'EnhancedPCNModel' if USE_ENHANCED_MODEL else 'DiscreteActionsDefaultModel'}")
        # remote の戻り値は pickle される。SchedulingEnvEventObs 内の C SchedulingEventBuffer は pickle 不可のため env は返さない。
        return True

    def run(self, n_episodes=10, random_actions=False, pre_fetched_commands=None, random_action_probs=None, heuristic_thresholds=None, fixed_action_seqs=None, giant_defer_thresholds=None):
        """pre_fetched_commands: list of (desired_return, desired_horizon), length n_actors*n_episodes.
        指定時は_choose_commandsのリモート呼び出しをスキップ（Learner負荷削減）。
        fixed_action_seqs: 固定行動列(NSGA-II遺伝子等)のリスト。actor間でround-robin分配して各列を1回ずつ再生する。
        giant_defer_thresholds: 占有量順位の閾値リスト。巨大ジョブ後回し種まき(汎化する行動則)用。"""
        if self.env is None:
            self._make_env()
        
        episodes_generated = 0
        collected_episodes = []  # 収集したエピソードを一時保存
        solution_execution_times = []  # 改良された解の実行時間を記録
        command_outcomes = []  # 可視化用: command -> achieved の対応
        episode_summaries = []  # 初期ランダム収集ログ用（観測配列は含めない）
        action_one_prob_counts = {}
        
        progress_interval = max(1, n_episodes // 10)
        
        if not random_actions:
            if _ACTOR_WEIGHTS_REF:
                # 共有ObjectRef経路: Learner actor キューに get_weights_ref を積む（get_weights と同じ順序点）。
                # 戻りは update_weights_ref が put した同一 state_dict の ObjectRef。中身は materialize 経路と bit一致。
                self._load_policy_weights(ray.get(self.learner.get_weights_ref.remote()))
            else:
                self.agent.model.load_state_dict(ray.get(self.learner.get_weights.remote()))
        
        for ep in range(n_episodes):
            try:
                cmd = None
                if pre_fetched_commands is not None:
                    idx = self.actor_id * n_episodes + ep
                    if idx < len(pre_fetched_commands):
                        cmd = pre_fetched_commands[idx]
                        # (desired_return, desired_horizon, base_return) 形式にも対応
                        # _run_episode には (desired_return, desired_horizon) だけ渡す
                random_action_prob = None
                heuristic_threshold = None
                fixed_actions = None
                giant_defer_threshold = None
                if random_actions and giant_defer_thresholds is not None and len(giant_defer_thresholds) > 0:
                    idx = self.actor_id * n_episodes + ep
                    giant_defer_threshold = float(giant_defer_thresholds[idx % len(giant_defer_thresholds)])
                    # 非巨大ジョブの配置に使う WaitTimeThreshold もスイープ(front を広げる)
                    if heuristic_thresholds is not None and len(heuristic_thresholds) > 0:
                        heuristic_threshold = float(heuristic_thresholds[idx % len(heuristic_thresholds)])
                    _gk = f"giantdefer>={giant_defer_threshold:g}"
                    action_one_prob_counts[_gk] = action_one_prob_counts.get(_gk, 0) + 1
                elif random_actions and fixed_action_seqs is not None and len(fixed_action_seqs) > 0:
                    idx = self.actor_id * n_episodes + ep
                    fixed_actions = fixed_action_seqs[idx % len(fixed_action_seqs)]
                    _fk = f"nsga_seed={idx % len(fixed_action_seqs)}"
                    action_one_prob_counts[_fk] = action_one_prob_counts.get(_fk, 0) + 1
                elif random_actions and heuristic_thresholds is not None and len(heuristic_thresholds) > 0:
                    idx = self.actor_id * n_episodes + ep
                    heuristic_threshold = float(heuristic_thresholds[idx % len(heuristic_thresholds)])
                    _hk = f"wtth={heuristic_threshold:g}"
                    action_one_prob_counts[_hk] = action_one_prob_counts.get(_hk, 0) + 1
                elif random_actions and random_action_probs is not None and len(random_action_probs) > 0:
                    idx = self.actor_id * n_episodes + ep
                    random_action_prob = float(random_action_probs[idx % len(random_action_probs)])
                    action_one_prob_counts[random_action_prob] = action_one_prob_counts.get(random_action_prob, 0) + 1
                episode = self._run_episode(
                    random_actions,
                    pre_fetched_command=cmd,
                    random_action_prob=random_action_prob,
                    heuristic_threshold=heuristic_threshold,
                    fixed_actions=fixed_actions,
                    giant_defer_threshold=giant_defer_threshold,
                )
                if random_actions and episode:
                    episode_uid = f"phase1:{self.actor_id}:{ep}:{random_action_prob}"
                    episode[0]._pcn_episode_uid = episode_uid
                    episode[0].random_action_prob = random_action_prob
                # print("done")
                
                # エピソードを一時保存
                collected_episodes.append(episode)
                episodes_generated += 1
                if random_actions:
                    episode_summaries.append(
                        _summarize_episode_for_log(
                            self.actor_id,
                            ep,
                            episode,
                            random_action_prob=random_action_prob,
                        )
                    )
                if (
                    _COLLECT_CMD_OUTCOMES
                    and not random_actions
                    and cmd is not None
                    and len(episode) > 0
                ):
                    # cmd: (desired_return, desired_horizon) or (desired_return, desired_horizon, base_return)
                    desired_return = np.array(cmd[0], dtype=np.float32)
                    base_return = np.array(cmd[2], dtype=np.float32) if (isinstance(cmd, (list, tuple)) and len(cmd) >= 3) else desired_return.copy()
                    # エピソードの割引累積報酬を計算（表示用、元データは変更しない）
                    rewards = [np.array(t.reward, dtype=np.float32, copy=True) for t in episode]
                    for i in reversed(range(len(rewards) - 1)):
                        rewards[i] = rewards[i] + self.agent.gamma * rewards[i + 1]
                    achieved_return = rewards[0] if rewards else np.zeros_like(desired_return)

                    # 実数値空間（値）も記録。
                    # current_e_values は [cost, avg_waiting_time] で描画しているため、command 側も同じ尺へ合わせる。
                    # desired_return[1] は -cost 累積、desired_return[0] は -waiting 累積なので、
                    # waiting は n_jobs で平均化して avg_waiting_time の近似尺度へ揃える。
                    n_jobs_for_scale = max(1, int(self.config['param_env'].get('n_jobs', N_JOBS)))
                    if hasattr(episode[0], 'objective_values') and episode[0].objective_values is not None:
                        # objective_values = [cost, _, avg_waiting_time]
                        obj = episode[0].objective_values
                        achieved_values = np.array([obj[0], obj[2]], dtype=np.float32)
                    else:
                        achieved_values = np.array(
                            [-achieved_return[1], (-achieved_return[0] / n_jobs_for_scale)],
                            dtype=np.float32
                        )
                    command_values = np.array(
                        [-desired_return[1], (-desired_return[0] / n_jobs_for_scale)],
                        dtype=np.float32
                    )
                    command_outcomes.append(
                        {
                            "base_return": base_return.tolist(),
                            "command_return": desired_return.tolist(),
                            "achieved_return": achieved_return.tolist(),
                            "command_values": command_values.tolist(),
                            "achieved_values": achieved_values.tolist(),
                        }
                    )
                
                # 改良された解の実行時間を記録（非ランダムアクションの場合）
                if not random_actions and len(episode) > 0 and hasattr(episode[0], 'solution_execution_time'):
                    solution_execution_times.append(episode[0].solution_execution_time)
                
                # 進捗表示（10分の1の間隔で）
                if (ep + 1) % progress_interval == 0 or (ep + 1) == n_episodes:
                    progress_percentage = ((ep + 1) / n_episodes) * 100
                    print(f"[Actor {self.actor_id}] 進捗: {ep+1}/{n_episodes} エピソード完了 ({progress_percentage:.1f}%)")
                
            except Exception as e:
                print(f"[Actor {self.actor_id}] エピソード {ep+1} でエラー: {e}")
        
        # 経験収集終了時に能動的にReplayBufferに詰め込む
        if DEBUG:
            print(f"[Actor {self.actor_id}] 経験収集完了。{len(collected_episodes)}エピソードをReplayBufferに追加中...")
        
        # 全てのエピソードをReplayBufferにバッチで追加（シリアライゼーション最適化）
        # Learner開始前にバッファに確実に反映させるため、完了を待機
        if len(collected_episodes) > 0:
            if _LOG_RAY_TRANSFER:
                nbytes = _estimate_episodes_numpy_bytes(collected_episodes)
                mode = "event_obs" if _USE_EVENT_OBS else "bitmap_c"
                n_tr = sum(len(ep) for ep in collected_episodes)
                print(
                    f"[RAY_TRANSFER] Actor→ReplayBuffer actor={self.actor_id} mode={mode} "
                    f"episodes={len(collected_episodes)} transitions={n_tr} "
                    f"numpy_payload≈{nbytes} B ({nbytes / 1024 ** 2:.2f} MiB)"
                )
            if _ACTOR_RAY_PUT:
                batch_ref = ray.put(collected_episodes)
                added_count = ray.get(self.buffer.add_batch_ref.remote(batch_ref))
            else:
                added_count = ray.get(self.buffer.add_batch.remote(collected_episodes))
            if added_count != len(collected_episodes):
                print(
                    f"[ReplayBuffer] actor={self.actor_id} added={added_count}/"
                    f"{len(collected_episodes)} episodes (duplicates skipped)"
                )
        
        # 改良された解の実行時間統計を表示（非ランダムアクションの場合）
        if not random_actions and solution_execution_times:
            avg_execution_time = np.mean(solution_execution_times)
            min_execution_time = np.min(solution_execution_times)
            max_execution_time = np.max(solution_execution_times)
            
            # print(f"[Actor {self.actor_id}] 改良された解の実行時間統計:")
            # print(f"  平均実行時間: {avg_execution_time:.4f}秒")
            # print(f"  最小実行時間: {min_execution_time:.4f}秒")
            # print(f"  最大実行時間: {max_execution_time:.4f}秒")
            # print(f"  実行時間記録数: {len(solution_execution_times)}エピソード")
        
        if DEBUG:
            print(f"[Actor {self.actor_id}] {episodes_generated} エピソードを生成し、ReplayBufferに追加しました")
        return {
            "episodes_generated": episodes_generated,
            "command_outcomes": command_outcomes,
            "episode_summaries": episode_summaries,
            "action_one_prob_counts": action_one_prob_counts,
        }

    def _waittime_threshold_action(self, threshold: float) -> int:
        """WaitTimeThreshold ヒューリスティック（reactive cloud overflow）:
        現ジョブのオンプレ予測待ち時間 >= threshold ならクラウド(1)、そうでなければオンプレ(0)。
        イベントネイティブ env の _find_event_allocation で予測する（副作用なしのクエリ）。
        早食い（コスト枠を序盤に使い切る）と逆に、混雑したジョブにだけクラウドを充てる→良質な低wait例。
        予測器を持たない env では 0 にフォールバック。"""
        env = self.env
        try:
            j = int(env.index_next_job)
            if j >= len(env.jobs):
                return 0
            raw_job = env.jobs[j]
            job = env._to_queue_job(raw_job)
            arrival = int(raw_job[0])
            _, onprem_start = env._find_event_allocation(job, False, arrival)
            predicted_wait = int(onprem_start) - arrival
            return 1 if predicted_wait >= float(threshold) else 0
        except Exception:
            return 0

    def _run_episode(self, random_actions=False, pre_fetched_command=None, random_action_prob=None, heuristic_threshold=None, fixed_actions=None, giant_defer_threshold=None):
        """pre_fetched_command: (desired_return, desired_horizon) が指定されていれば_choose_commandsをスキップ
        fixed_actions: ジョブ順の固定行動列(NSGA-II遺伝子等)。scheduled の時だけindexを進める(nsga2_agent._rolloutと同一規約)。
        giant_defer_threshold: 占有量パーセンタイル順位がこの値以上のジョブを後回し(action=2)するヒューリスティック。
          順位は相対値なので任意インスタンスで汎化(未知ジョブに崩れない種まき)。非巨大は heuristic_threshold で 0/1。"""
        obs = self.env.reset()
        done = False
        transitions = []
        _fa_idx = 0  # fixed_actions 用: スケジュール成立済みジョブ数
        # アンカー残差モード(get_anchor_set()がNoneでなければON)。OFF時は以下の分岐を一切通らずビット一致。
        _ar = get_anchor_set()
        _ar_gen_gene = None     # 生成時アンカー(方策分岐のXOR用; 達成値ベースの事後リアンカーとは別)
        _ar_job_idx = 0         # scheduled時のみ前進するジョブindex(_rollout規約)
        _ar_abs_actions = []    # 各transitionでenvに渡した絶対行動(事後リアンカーで残差化)
        _ar_job_idxs = []       # 各transitionのjob_idx

        solution_selection_start_time = None
        solution_execution_time = None
        _episode_command_return = None  # [案1] このエピソードの生成に使った指令(desired_return初期値)。指令追従lossが読む。

        if not random_actions:
            solution_selection_start_time = time.time()
            if pre_fetched_command is not None:
                if isinstance(pre_fetched_command, (list, tuple)) and len(pre_fetched_command) >= 2:
                    desired_return, desired_horizon = pre_fetched_command[0], pre_fetched_command[1]
                else:
                    desired_return, desired_horizon = pre_fetched_command
            else:
                t_choose_start = time.time()
                desired_return, desired_horizon = ray.get(self.learner._choose_commands.remote(50))
                if _PROFILE_MODE:
                    print(f"[PROFILE Actor {self.actor_id}] _choose_commands: {time.time()-t_choose_start:.3f}s")
            desired_return = np.array(desired_return, dtype=np.float32, copy=True)
            desired_horizon = np.float32(desired_horizon)
            # [案1] 初期指令を保存（1379で desired_return -= reward に上書きされる前）。
            # pre_fetched_command 経路でも _choose_commands 経路でも、ここで desired_return に揃うので両方捕捉。
            _episode_command_return = np.array(desired_return, dtype=np.float32, copy=True)
            self.agent.set_desired_return_and_horizon(desired_return, desired_horizon)
            if _ar is not None:
                # 方策の出力は「生成時アンカーからの残差」と解釈する
                _, _ar_gen_gene = _ar.select(desired_return)
            
            # print(f"[Actor {self.actor_id}] 改良された解の選択完了: 目標報酬={desired_return}, ホライズン={desired_horizon}")
        
        # ランダムアクションの場合、エピソードごとに異なるシードを設定
        if random_actions:
            # Actor ID、現在時刻、エピソードIDを組み合わせてユニークなシードを生成
            episode_seed = (int(time.time() * 1000000) + self.actor_id * 10000 + hash(obs.tobytes())) % 10000
            np.random.seed(episode_seed)
            if DEBUG:
                print(f"[Actor {self.actor_id}] ランダムアクション用シード設定: {episode_seed}")
        start_time = time.time()
        step_count = 0
        t_steps_start = time.time()
        while not done:
            if random_actions:
                if giant_defer_threshold is not None:
                    # 汎化する種まき: 占有量上位(相対順位>=閾値)の巨大ジョブは後回し(2)、
                    # それ以外は WaitTimeThreshold(heuristic_threshold or 150)で配置。
                    _lev = self.env._front_job_leverage() if hasattr(self.env, "_front_job_leverage") else 0.0
                    if _lev >= float(giant_defer_threshold):
                        action = 2
                    else:
                        action = self._waittime_threshold_action(
                            heuristic_threshold if heuristic_threshold is not None else 150.0
                        )
                elif fixed_actions is not None:
                    action = int(fixed_actions[_fa_idx]) if _fa_idx < len(fixed_actions) else 0
                elif heuristic_threshold is not None:
                    action = self._waittime_threshold_action(heuristic_threshold)
                elif random_action_prob is not None:
                    action = 1 if np.random.random() < random_action_prob else 0
                else:
                    # より多様なランダム行動を生成
                    if len(transitions) < 5:
                        # 最初の5ステップは完全ランダム
                        action = self.env.action_space.sample()
                    else:
                        # その後は少し偏りを持たせて多様性を確保
                        if np.random.random() < 0.7:
                            action = self.env.action_space.sample()
                        else:
                            # 30%の確率で前の行動と異なる行動を選択
                            if len(transitions) > 0:
                                prev_action = transitions[-1].action
                                if prev_action == 0:
                                    action = 1
                                else:
                                    action = 0
                            else:
                                action = self.env.action_space.sample()
                
                # 行動の多様性を確認するためのログ（最初の数ステップのみ）
                if DEBUG and len(transitions) < 3:
                    print(f"[Actor {self.actor_id}] ステップ {len(transitions)+1}: ランダム行動 = {action}")
            else:
                # PCN本体の _run_episode と同じく方策からサンプリング。イベント観測は _obs_for_policy で bitmap 次元へ。
                policy_obs = self.agent._obs_for_policy(self.env, obs)
                action = self.agent._act(
                    policy_obs, desired_return, desired_horizon, eval_mode=False
                )
                
            if _ar is not None:
                # 方策分岐の action は残差。生成時アンカーとXORして絶対行動を作る。
                # random/heuristic/fixed の action は既に絶対行動なのでそのまま。
                if (not random_actions) and _ar_gen_gene is not None:
                    _abit = int(_ar_gen_gene[_ar_job_idx]) if _ar_job_idx < len(_ar_gen_gene) else 0
                    env_action = _abit ^ int(action)
                else:
                    env_action = int(action)
            else:
                env_action = action
            n_obs, reward, scheduled, wt_step, done = self.env.step(env_action)
            step_count += 1
            if fixed_actions is not None and scheduled:
                _fa_idx += 1
            if _ar is not None:
                _ar_abs_actions.append(int(env_action))
                _ar_job_idxs.append(_ar_job_idx)
                if scheduled:
                    _ar_job_idx += 1

            # 観測データをfloat32に変換してメモリ使用量を削減（シリアライゼーション最適化）
            # 既にfloat32の場合は変換しない（メモリコピーを避ける）
            if hasattr(obs, 'dtype') and obs.dtype != np.float32:
                obs = np.array(obs, dtype=np.float32, copy=True)
            if hasattr(n_obs, 'dtype') and n_obs.dtype != np.float32:
                n_obs = np.array(n_obs, dtype=np.float32, copy=True)
            transitions.append(Transition(obs, env_action, np.float32(reward).copy(), n_obs, done))
            obs = n_obs
            if not random_actions:
                # 学習時の教師は「各時刻の残りreturn/horizon」なので、実行時commandも同じように更新する。
                # これを固定したままだと、学習時条件と実行時条件が別物になり、Cost軸追従が崩れる。
                desired_return = desired_return - np.array(reward, dtype=np.float32)
                if _COST_HOLD:
                    desired_return[1] = _episode_command_return[1]  # [anti-ration] cost目標を一定保持
                if _DESIRED_RETURN_UB is not None:
                    # [元PCN復元] 上限クリップ: 報酬負で dr が達成不能な正側へ漂流するのを防ぐ(元 max_return)。
                    desired_return = np.minimum(desired_return, _DESIRED_RETURN_UB).astype(np.float32)
                self.agent.set_desired_return_and_horizon(desired_return, desired_horizon)
                if scheduled:
                    desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
                    self.agent.set_desired_return_and_horizon(desired_return, desired_horizon)
            
        # エピソード完了時に実数値を計算
        if done:
            t_steps = time.time() - t_steps_start
            if _PROFILE_MODE and step_count > 0:
                print(f"[PROFILE Actor {self.actor_id}] env.step loop: {t_steps:.3f}s ({step_count} steps, {t_steps/step_count*1000:.1f}ms/step)")
            self.env.finalize_window_history()
            cost, _, avg_waiting_time = self.env.calc_objective_values()

            # 事後リアンカー: 達成値の最近傍アンカー基準で全transitionの行動を残差化する。
            # これにより policy / random / heuristic / fixed すべてが残差表現に統一され、
            # アーカイブ内で絶対行動と残差が混在する汚染を防ぐ(設計書§4)。
            if _ar is not None and len(transitions) > 0:
                _, _ach_gene = _ar.select_by_values(cost, avg_waiting_time)
                for _ti in range(len(transitions)):
                    _ji = _ar_job_idxs[_ti]
                    _abit = int(_ach_gene[_ji]) if _ji < len(_ach_gene) else 0
                    transitions[_ti].action = int(_ar_abs_actions[_ti]) ^ _abit

            solution_execution_time = time.time() - start_time
                
            # print(f"[Actor {self.actor_id}] 改良された解の実行完了")
            # print(f"  選択〜実行完了時間: {solution_execution_time:.4f}秒")
            # print(f"  最終コスト: {cost}")
            # print(f"  平均待機時間: {avg_waiting_time}")
            
            # 最初のTransitionに実数値を追加（後でアクセスできるように）
            if len(transitions) > 0:
                transitions[0].objective_values = [cost,_,avg_waiting_time]
                # [案1: 指令追従loss] 生成指令(desired_return)を先頭Transitionに保持する。
                # experience_replay 経由で学習側の _command_track_loss_from_replay が「obs・生成指令・
                # 達成obj」を揃えて読む（達成 objective_values と同じ動的属性方式。hasattr で安全フィルタ）。
                # 既存挙動は不変（属性が増えるだけ・loss は PCN_CMD_TRACK_WEIGHT=0 で読まないのでビット一致）。
                if (not random_actions) and _episode_command_return is not None:
                    transitions[0].command_return = np.asarray(_episode_command_return, dtype=np.float32)
                # 実行時間も追加（非ランダムアクションの場合）
                if not random_actions and solution_execution_time is not None:
                    transitions[0].solution_execution_time = solution_execution_time
        
        # エピソード完了時の統計を表示（ランダムアクションの場合）
        if DEBUG and random_actions:
            actions = [t.action for t in transitions]
            unique_actions, counts = np.unique(actions, return_counts=True)
            action_distribution = dict(zip(unique_actions, counts))
            print(f"[Actor {self.actor_id}] エピソード完了 - 行動分布: {action_distribution}")
            
            # 累積報酬を計算して表示（表示用のみ、元のデータは変更しない）
            episode_return = transitions[0].reward
            transitions_copy = []
            for t in transitions:
                transitions_copy.append(Transition(
                    observation=t.observation,
                    action=t.action,
                    reward=np.array(t.reward, copy=True),
                    next_observation=t.next_observation,
                    terminal=t.terminal
                ))
            
            for i in reversed(range(len(transitions_copy) - 1)):
                transitions_copy[i].reward += self.agent.gamma * transitions_copy[i + 1].reward
            
            final_return = transitions_copy[0].reward
            print(f"[Actor {self.actor_id}] エピソード完了 - 累積報酬: {final_return}")
            
            # 実数値も表示
            if hasattr(transitions[0], 'objective_values'):
                print(f"[Actor {self.actor_id}] エピソード完了 - 実数値: コスト={transitions[0].objective_values[0]}, 実行時間={transitions[0].objective_values[1]}")
        
        return transitions

    def _load_policy_weights(self, weights_ref=None) -> None:
        """ObjectRef または materialized state_dict のどちらでも重みをロード。"""
        if weights_ref is None:
            state = ray.get(self.learner.get_weights.remote())
        elif isinstance(weights_ref, dict):
            state = weights_ref
        else:
            state = ray.get(weights_ref)
        self.agent.model.load_state_dict(state)

    def evaluate_episode(self, desired_return, desired_horizon, max_return, weights_ref=None):
        """単一エピソードの評価を実行。weights_ref 指定時は Learner へ戻らない（デッドロック回避）。"""
        if self.env is None:
            self._make_env()
        self._load_policy_weights(weights_ref)
        
        # 評価エピソードの実行時間計測開始
        evaluation_start_time = time.time()
        
        # 目標値を設定
        self.agent.set_desired_return_and_horizon(desired_return, desired_horizon)
        
        # エピソード実行
        transitions, _, _, _, map_fin, value = self.agent._run_episode(
            self.env, desired_return, desired_horizon, max_return, eval_mode=True
        )
        
        # 評価エピソードの実行時間計測終了
        evaluation_end_time = time.time()
        evaluation_execution_time = evaluation_end_time - evaluation_start_time
        
        # 累積報酬を計算（表示用のみ、元のデータは変更しない）
        transitions_copy = []
        for t in transitions:
            transitions_copy.append(Transition(
                observation=t.observation,
                action=t.action,
                reward=np.array(t.reward, copy=True),
                next_observation=t.next_observation,
                terminal=t.terminal
            ))
        
        for i in reversed(range(len(transitions_copy) - 1)):
            transitions_copy[i].reward += self.agent.gamma * transitions_copy[i + 1].reward
        
        episode_return = transitions_copy[0].reward
        
        if DEBUG and not _EVAL_QUIET:
            print(f"[Actor {self.actor_id}] 評価エピソード完了: 報酬={episode_return}, 実数値={value}")
            print(f"  評価実行時間: {evaluation_execution_time:.4f}秒")
        
        return episode_return, value, transitions, map_fin

    def eval_uniform_grid_batch(self, commands, weights_ref=None):
        """均等格子 PF 用: (desired_return, horizon) のリストを同一 Actor で連続評価。"""
        if self.env is None:
            self._make_env()
        self._load_policy_weights(weights_ref)
        max_return = np.full(2, np.inf, dtype=np.float32)
        values = []
        for dr, hz in commands:
            desired_return = np.asarray(dr, dtype=np.float32)
            desired_horizon = np.float32(hz)
            _, _, _, _, _, val = self.agent._run_episode(
                self.env,
                desired_return,
                desired_horizon,
                max_return,
                eval_mode=True,
            )
            values.append([float(val[0]), float(val[1])])
        return values

# =========================
# 3. Learner (Ray Actor)
# =========================
# GPUリソースを条件付きで要求（RayがGPUを認識している場合のみ）
# 注意: Rayがクラスターモードで実行されている場合、num_gpusを指定すると
# autoscalerがGPUノードを探してしまう。そのため、Learnerの初期化時に
# GPUが利用可能かどうかを確認し、利用可能な場合のみGPUを使用する。
# Rayのリソース管理は行わず、PyTorchが直接GPUを使用する。
class Learner:
    def __init__(self, config, buffer, device='cuda'):
        self.config = config
        self.env = self._make_env()
        
        # より堅牢なデバイス検出
        self.actual_device = self._get_available_device(device)
        
        if self.actual_device == 'cuda':
            th.cuda.reset_peak_memory_stats(0)
        
        # PCNエージェントを正しいデバイスで初期化（GPU使用時はGPUで初期化、BATCH_SIZE使用）
        # horizon はエピソード長(≈n_jobs)で割って command 埋め込み内で return と桁を揃える
        # （正規化後の return は約[-1,0]。horizon を素のまま入れると桁が大きく条件付けを潰す）。
        _h_scale = 1.0 / max(1, int(self.config['param_env'].get('n_jobs', N_JOBS)))
        self.agent = PCN(
            self.env,
            device=self.actual_device,  # 検出したデバイスで初期化
            state_dim=self.env.observation_space.shape[0],
            scaling_factor=np.array([1.0, 1.0, _h_scale]),
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            hidden_dim=int(os.environ.get("PCN_HIDDEN_DIM", "512")),
            project_name="temp",
            experiment_name="PCN",
            log=False,
            debug_mode=DEBUG,  # DEBUGフラグを追加
            use_enhanced_model=USE_ENHANCED_MODEL,  # モデル選択
        )
        self.buffer = buffer
        self.global_step = 0
        self.experience_replay = []  # PCNエージェントの経験再生バッファ
        self.gamma = 1.0  # 割引率
        self.last_eval_step = 0  # 最後に評価を行ったステップ
        self._hash_cache = {}  # エピソードのハッシュ値キャッシュ（idをキーとして使用）
        self._episode_uids = set()  # _pcn_episode_uid による高速重複検出
        self._weights_ref = None  # 重みのObjectRefを保持（重みの共有用）
        self._cuda_perf_tuned = False
        self._use_jax = False
        self._mo_hv_wall0 = time.perf_counter()  # mo_benchmark_hv 用の壁時計起点
        if _USE_JAX_LEARNER and USE_ENHANCED_MODEL is False:
            try:
                from src.agents.pcn_jax import (
                    init_model, PCNModelJAX, jax_params_to_pytorch_state_dict,
                    JAX_AVAILABLE,
                )
                import jax
                import optax
                if JAX_AVAILABLE:
                    state_dim = self.env.observation_space.shape[0]
                    action_dim = self.env.action_space.n
                    reward_dim = self.env.reward_space.shape[0]
                    key = jax.random.PRNGKey(42)
                    self._jax_model, self._jax_params = init_model(
                        state_dim, action_dim, reward_dim, 512,
                        [1, 1, 1], key
                    )
                    self._jax_opt = optax.adam(LEARNING_RATE)
                    self._jax_opt_state = self._jax_opt.init(self._jax_params)
                    self._jax_key = key
                    self._use_jax = True
                    print("[Learner] JAX+CUDA 学習を有効化")
            except Exception as e:
                print(f"[Learner] JAX初期化失敗、PyTorchにフォールバック: {e}")
        if self.actual_device == 'cuda':
            _mode = "event_obs" if _USE_EVENT_OBS else "bitmap_c"
            _sdim = int(self.env.observation_space.shape[0])
            _log_gpu_memory_snapshot(
                f"Learner初期化完了 mode={_mode} state_dim={_sdim}"
            )
        if DEBUG:
            print(f"Learner initialized with device: {self.actual_device}")
            print(f"Learner model: {'EnhancedPCNModel' if USE_ENHANCED_MODEL else 'DiscreteActionsDefaultModel'}")
            if self.actual_device == 'cuda':
                import torch
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """既存 .pth から model_state_dict を読み込む（strict=False）。"""
        import torch as th

        state = th.load(path, map_location=self.actual_device, weights_only=False)
        sd = state.get("model_state_dict", state)
        target = self.agent.network if self.agent.use_enhanced_model else self.agent.model
        missing, unexpected = target.load_state_dict(sd, strict=False)
        self.agent.model.eval()
        if hasattr(self.agent, "network"):
            self.agent.network.eval()
        self.update_weights_ref()
        if hasattr(self.agent, "reinit_ema"):
            self.agent.reinit_ema()  # ロード重みからEMA再初期化(EMA無効時 no-op)
        info = {
            "path": path,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        print(
            f"[Learner] checkpoint loaded: {path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return info

    def _get_available_device(self, requested_device):
        """利用可能なデバイスを検出"""
        import torch
        
        if requested_device == 'cuda':
            if not torch.cuda.is_available():
                if DEBUG:
                    print("CUDAが利用できないため、CPUを使用します")
                return 'cpu'
            # Ray環境でのGPUリソース確認（オプション）
            gpu_ids = ray.get_gpu_ids() if hasattr(ray, 'get_gpu_ids') else []
            if gpu_ids:
                if DEBUG:
                    print(f"Ray GPU detected: {gpu_ids}")
                if len(gpu_ids) > 0:
                    torch.cuda.set_device(gpu_ids[0])
            if DEBUG:
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            if DEBUG:
                print(f"Using requested device: {requested_device}")
            return requested_device

    def _make_env(self):
        n_jobs = self.config['param_env'].get('n_jobs', N_JOBS)
        job_generator = JobGenerator(
            0, 1,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config, n_jobs, 0.2, 0
        )
        jobs_set = job_generator.generate_jobs_set()
        env = _EnvClass(
            np.inf,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config['param_env']['n_job_queue_obs'],
            self.config['param_env']['n_job_queue_bck'],
            self.config['param_agent']['weight_wt'],
            self.config['param_agent']['weight_cost'],
            self.config['param_env']['penalty_not_allocate'],
            self.config['param_env']['penalty_invalid_action'],
            jobs_set,
            None, flag=0
        )
        env = _enable_event_bitmap_adapter(env)
        # C実装が正しく使用されているか確認
        if hasattr(env, '_cache_onpre_c'):
            print("[Learner] ✓ C実装環境が正しく初期化されました")
        else:
            print("[Learner] ⚠️ C実装環境の初期化に問題があります")
        return env

    def _jax_update_step(self) -> float:
        """JAX で 1 ステップ学習更新。get_weights で PyTorch state_dict に変換して返す。"""
        import jax
        import jax.numpy as jnp
        import optax
        from src.agents.pcn_jax import JAX_AVAILABLE
        if not JAX_AVAILABLE:
            return 0.0
        obs, actions, desired_returns, desired_horizons = self.agent.get_training_batch()
        desired_horizons = desired_horizons[:, np.newaxis].astype(np.float32)
        obs_j = jnp.array(obs)
        dr_j = jnp.array(desired_returns)
        dh_j = jnp.array(desired_horizons)
        actions_j = jnp.array(actions)

        def loss_fn(params):
            logits = self._jax_model.apply(params, obs_j, dr_j, dh_j)
            one_hot = jax.nn.one_hot(actions_j, logits.shape[-1])
            nll = -jnp.sum(one_hot * logits, axis=-1)
            return jnp.mean(nll)

        loss_val, grads = jax.value_and_grad(loss_fn)(self._jax_params)
        updates, self._jax_opt_state = self._jax_opt.update(grads, self._jax_opt_state)
        self._jax_params = optax.apply_updates(self._jax_params, updates)
        return float(np.array(loss_val))

    def get_step_skip_stats(self):
        """Phase3凍結検知用: (skip累計, step成功累計)。skip率= skip/(skip+step)が高いと重みが動いていない。"""
        return (
            int(getattr(self.agent, "_nan_skip_total", 0)),
            int(getattr(self.agent, "_opt_step_total", 0)),
        )

    def get_weights(self):
        # CPUデバイスでモデルの重みを返す（ActorがCPUで実行されるため）
        if getattr(self, '_use_jax', False):
            from src.agents.pcn_jax import jax_params_to_pytorch_state_dict
            return jax_params_to_pytorch_state_dict(self._jax_params, scaling_factor=np.array([1, 1, 1]))
        if USE_ENHANCED_MODEL and hasattr(self.agent, 'network'):
            model_state = self.agent.network.state_dict()
        else:
            model_state = self.agent.model.state_dict()
        # torch.compile が _orig_mod. プレフィックスを付ける場合の除去
        def strip_orig_mod(d):
            return {k.replace('_orig_mod.', ''): v.cpu() for k, v in d.items()}
        return strip_orig_mod(model_state)
    
    def get_weights_ref(self):
        """モデルの重みのObjectRefを取得（重みの共有用）"""
        # 既存のObjectRefがある場合はそれを返す（重みの共有を最大化）
        # 重みが更新された場合は、update_weights_ref()が呼ばれるまで古いObjectRefを返す
        if self._weights_ref is None:
            # 初回のみ重みを取得してObject Storeに保存
            weights = self.get_weights()
            self._weights_ref = ray.put(weights)
        return self._weights_ref
    
    def update_weights_ref(self):
        """重みを更新してObjectRefを更新（学習後に呼び出す）"""
        weights = self.get_weights()
        self._weights_ref = ray.put(weights)
        return self._weights_ref

    def reinit_ema(self):
        """EMA shadow を現在(=Phase2学習済み)重みで初期化。Phase3突入直前に1回呼ぶ。EMA無効時はno-op。"""
        if hasattr(self.agent, "reinit_ema"):
            return self.agent.reinit_ema()
        return False

    def get_eval_weights_ref(self):
        """eval配布用の重みref。EMA有効時はEMA重みをput(rollout用onlineは温存)、無効時はonline refにフォールバック。
        rollout(get_weights_ref)とeval(これ)で重みを分離し、探索はonline・評価はEMAにする。"""
        if not getattr(self.agent, "_ema_shadow", None):
            return self.get_weights_ref()
        self.agent.swap_in_ema_weights()
        try:
            w = self.get_weights()   # EMA載った状態の state_dict(_orig_mod strip込み)
        finally:
            self.agent.restore_online_weights()
        return ray.put(w)

    def start_lr_schedule(self, n_iterations):
        """Phase3突入直前に呼ぶ: lr減衰スケジュールのカウンタを初期化(以降のlearnで減衰)。LR_DECAY OFF時 no-op。"""
        self._lr_sched_total = int(n_iterations)
        self._lr_sched_count = 0
        return True

    def _learner_event_transition_to_bitmap(self, t: Transition) -> Transition:
        """ReplayBuffer 経由のイベント生観測を、学習用ビットマップ観測へ復元する。"""
        target = int(self.env.observation_space.shape[0])
        o = np.asarray(t.observation, dtype=np.float32).reshape(-1)
        if o.size >= target:
            return t
        ow = int(getattr(self.env, "obs_window_size", 10))
        obs_b = event_obs_to_bitmap_observation(
            t.observation,
            int(self.env.n_window),
            int(self.env.n_on_premise_node),
            int(self.env.n_cloud_node),
            ow,
        )
        next_b = event_obs_to_bitmap_observation(
            t.next_observation,
            int(self.env.n_window),
            int(self.env.n_on_premise_node),
            int(self.env.n_cloud_node),
            ow,
        )
        t_new = Transition(
            observation=obs_b,
            action=t.action,
            reward=np.array(t.reward, copy=True),
            next_observation=next_b,
            terminal=t.terminal,
        )
        if hasattr(t, "objective_values"):
            t_new.objective_values = t.objective_values
        if hasattr(t, "solution_execution_time"):
            t_new.solution_execution_time = t.solution_execution_time
            t_new.command_return = getattr(t, "command_return", None)  # [案1] 指令もコピー保持
        if hasattr(t, "_pcn_episode_uid"):
            t_new._pcn_episode_uid = t._pcn_episode_uid
        if hasattr(t, "random_action_prob"):
            t_new.random_action_prob = t.random_action_prob
        return t_new

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> Optional[List[Transition]]:
        """エピソードを経験再生バッファに追加。追加した場合は格納したエピソードを返す。"""
        if not transitions:
            return None

        episode_uid = getattr(transitions[0], "_pcn_episode_uid", None)
        if _USE_EVENT_OBS and learner_bitmap_enabled():
            transitions_copy = []
            for t in transitions:
                t = self._learner_event_transition_to_bitmap(t)
                reward_copy = np.array(t.reward, copy=True)
                t_copy = Transition(
                    observation=t.observation,
                    action=t.action,
                    reward=reward_copy,
                    next_observation=t.next_observation,
                    terminal=t.terminal,
                )
                if hasattr(t, "objective_values"):
                    t_copy.objective_values = t.objective_values
                if hasattr(t, "solution_execution_time"):
                    t_copy.solution_execution_time = t.solution_execution_time
                    t_copy.command_return = getattr(t, "command_return", None)  # [案1] 指令もコピー保持
                if hasattr(t, "_pcn_episode_uid"):
                    t_copy._pcn_episode_uid = t._pcn_episode_uid
                if hasattr(t, "random_action_prob"):
                    t_copy.random_action_prob = t.random_action_prob
                transitions_copy.append(t_copy)
            stored = transitions_copy
        elif episode_uid is not None and _REPLAY_ZERO_COPY:
            if episode_uid in self._episode_uids:
                if DEBUG:
                    print(f"[Learner] 重複エピソード(uid)をスキップ: {episode_uid}")
                return None
            stored = transitions
        else:
            transitions_copy = []
            for t in transitions:
                reward_copy = np.array(t.reward, copy=True)
                t_copy = Transition(
                    observation=t.observation,
                    action=t.action,
                    reward=reward_copy,
                    next_observation=t.next_observation,
                    terminal=t.terminal,
                )
                if hasattr(t, "objective_values"):
                    t_copy.objective_values = t.objective_values
                if hasattr(t, "solution_execution_time"):
                    t_copy.solution_execution_time = t.solution_execution_time
                    t_copy.command_return = getattr(t, "command_return", None)  # [案1] 指令もコピー保持
                if hasattr(t, "_pcn_episode_uid"):
                    t_copy._pcn_episode_uid = t._pcn_episode_uid
                if hasattr(t, "random_action_prob"):
                    t_copy.random_action_prob = t.random_action_prob
                transitions_copy.append(t_copy)
            stored = transitions_copy

        for i in reversed(range(len(stored) - 1)):
            stored[i].reward = np.asarray(stored[i].reward, dtype=np.float32) + self.gamma * np.asarray(
                stored[i + 1].reward, dtype=np.float32
            )

        if episode_uid is not None and _REPLAY_ZERO_COPY:
            episode_hash = hash(episode_uid)
            self._episode_uids.add(episode_uid)
        else:
            episode_hash = self._compute_episode_hash(stored)
            if self._is_duplicate_episode(episode_hash):
                if DEBUG:
                    print(f"[Learner] 重複エピソードをスキップしました。ハッシュ: {episode_hash}")
                return None
            if not hasattr(self, "_episode_hashes"):
                self._episode_hashes = set()
            self._episode_hashes.add(episode_hash)

        was_at_capacity = len(self.agent.experience_replay) == max_size
        unique_step = (step, episode_hash)
        if was_at_capacity:
            heapq.heappushpop(self.agent.experience_replay, (1, unique_step, stored))
            self.agent.mark_training_batch_cache_stale()
        else:
            heapq.heappush(self.agent.experience_replay, (1, unique_step, stored))

        if DEBUG:
            print(f"[Learner] エピソードを追加しました。現在のバッファサイズ: {len(self.agent.experience_replay)}")
        return stored

    def _compute_episode_hash(self, transitions: List[Transition]) -> int:
        """エピソードの内容に基づくハッシュ値を計算（軽量版）"""
        import hashlib
        
        if not transitions:
            return 0
        
        # キャッシュチェック（transitionsリストのidをキーとして使用）
        transitions_id = id(transitions)
        if transitions_id in self._hash_cache:
            return self._hash_cache[transitions_id]
        
        # エピソードを一意に識別する要約情報のみを使用
        hasher = hashlib.md5()
        
        # 1. エピソードの長さ
        episode_len = len(transitions)
        hasher.update(episode_len.to_bytes(8, byteorder='big'))

        episode_uid = getattr(transitions[0], '_pcn_episode_uid', None)
        if episode_uid is not None:
            hasher.update(str(episode_uid).encode())
        
        # 2. 最初の観測の要約（最初の数要素のみ、またはハッシュ）
        first_obs = transitions[0].observation
        if hasattr(first_obs, 'tobytes'):
            # 観測が大きい場合は最初の一部のみを使用
            obs_summary = first_obs.flatten()[:min(100, first_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(first_obs).encode())
        
        # 3. 行動のシーケンス（効率的にバイト列として結合）
        actions = np.array([t.action for t in transitions], dtype=np.int32)
        hasher.update(actions.tobytes())
        
        # 4. 報酬の要約（合計と平均）
        rewards = np.array([t.reward for t in transitions])
        if rewards.size > 0:
            reward_summary = np.array([rewards.sum(), rewards.mean()], dtype=np.float32)
            hasher.update(reward_summary.tobytes())
        
        # 5. 最後の観測の要約
        last_obs = transitions[-1].next_observation
        if hasattr(last_obs, 'tobytes'):
            obs_summary = last_obs.flatten()[:min(100, last_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(last_obs).encode())
        
        # 6. ターミナル状態の情報
        terminal_info = np.array([t.terminal for t in transitions], dtype=bool)
        hasher.update(terminal_info.tobytes())
        
        # ハッシュ値を計算（intに変換）
        hash_value = int(hasher.hexdigest(), 16)
        
        # キャッシュに保存（transitionsリストのidをキーとして使用）
        self._hash_cache[transitions_id] = hash_value
        
        return hash_value
    
    def _is_duplicate_episode(self, episode_hash: int) -> bool:
        """エピソードが重複しているかチェック"""
        if not hasattr(self, '_episode_hashes'):
            self._episode_hashes = set()
        return episode_hash in self._episode_hashes

    def _nlargest(self, n: int, threshold: float = 0.1) -> List[Tuple[float, int, List[Transition]]]:
        """経験再生バッファから上位n個のエピソードを取得"""
        return self.agent._nlargest(n, threshold)

    def get_archive_pareto_snapshot(self, include_all_points: bool = None) -> Dict[str, Any]:
        """Replay archive の PF スナップショット（既定は PF のみ、Ray 転送を抑える）。

        Policy Eval PF（NN再実行）と Archive PF（収集済み経験）を分けて描画する。
        include_all_points=True のときのみ全 archive 点を返す（DISTRIBUTED_PCN_ARCHIVE_VIS_ALL=1）。
        """
        if include_all_points is None:
            include_all_points = _ARCHIVE_VIS_ALL
        n_jobs_for_value_scale = max(1, int(self.config['param_env'].get('n_jobs', N_JOBS)))
        returns_chunks = []
        values_chunks = []
        for _, _, episode in self.agent.experience_replay:
            if not episode:
                continue
            first = episode[0]
            r = np.asarray(first.reward, dtype=np.float64)
            returns_chunks.append(r)
            if hasattr(first, "objective_values") and first.objective_values is not None:
                obj = first.objective_values
                values_chunks.append(np.array([obj[0], obj[2]], dtype=np.float64))
            else:
                values_chunks.append(np.array([-r[1], -r[0] / n_jobs_for_value_scale], dtype=np.float64))

        if not returns_chunks:
            return {
                "all_returns": [],
                "all_values": [],
                "pareto_front_reward": [],
                "pareto_front_values": [],
                "n_archive": 0,
                "n_unique_values": 0,
                "n_unique_returns": 0,
            }

        returns_np = np.stack(returns_chunks, axis=0)
        values_np = np.stack(values_chunks, axis=0)
        nd_r = get_non_dominated_inds(returns_np)
        nd_v = get_non_dominated_inds_minimize(values_np)
        pf_returns = returns_np[nd_r]
        pf_values = values_np[nd_v]
        out: Dict[str, Any] = {
            "pareto_front_reward": pf_returns.tolist(),
            "pareto_front_values": pf_values.tolist(),
            "n_archive": int(len(returns_np)),
            "n_unique_values": int(len(np.unique(np.round(values_np, 6), axis=0))),
            "n_unique_returns": int(len(np.unique(np.round(returns_np, 6), axis=0))),
        }
        if include_all_points:
            out["all_returns"] = returns_np.tolist()
            out["all_values"] = values_np.tolist()
        else:
            out["all_returns"] = []
            out["all_values"] = []
        return out

    def _choose_commands(self, num_episodes: int) -> Tuple[np.ndarray, np.float32]:
        """次のエピソードの目標報酬とホライズンを選択"""
        return self.agent._choose_commands(num_episodes)

    def _choose_commands_batch(self, num_episodes: int, n_commands: int):
        """複数の異なる探索方向を一括で取得（Learner呼び出しを1回に削減、各Actorに異なる目標値）"""
        return self.agent._choose_commands_batch(num_episodes, n_commands)

    def _ensure_cuda_perf(self) -> None:
        if self.actual_device != "cuda" or self._cuda_perf_tuned:
            return
        import torch
        torch.backends.cudnn.benchmark = True
        self._cuda_perf_tuned = True

    def learn(self, batch_size: int = 100, n_updates: int = 2, use_training_cache: bool = False) -> float:
        self._ensure_cuda_perf()
        total_loss = []
        total_policy_acc = []
        total_true_prob = []
        total_cmd_track = []  # [案1] 指令追従loss の集計（出力確認・効き具合の可視化）
        
        # ReplayBufferから全てのエピソードを取得（サンプリングせずに全部）
        # buffer.size()は不要（get_all_episodes()の戻り値が空かどうかで判定できる）
        t_get_episodes_start = time.time()
        if _REPLAY_ZERO_COPY:
            all_episodes = ray.get(self.buffer.take_all_episodes.remote())
        else:
            all_episodes = ray.get(self.buffer.get_all_episodes.remote())
        t_get_episodes = time.time() - t_get_episodes_start
        if _LOG_RAY_TRANSFER and all_episodes:
            nbytes = _estimate_episodes_numpy_bytes(all_episodes)
            mode = "event_obs" if _USE_EVENT_OBS else "bitmap_c"
            n_tr = sum(len(ep) for ep in all_episodes)
            print(
                f"[RAY_TRANSFER] ReplayBuffer→Learner learn() mode={mode} "
                f"episodes={len(all_episodes)} transitions={n_tr} "
                f"numpy_payload≈{nbytes} B ({nbytes / 1024 ** 2:.2f} MiB)"
            )
        if _PROFILE_MODE and not hasattr(self, '_learn_timings'):
            self._learn_timings = {'get_episodes': [], 'add_episodes': [], 'update': []}
        if _PROFILE_MODE and hasattr(self, '_learn_timings'):
            self._learn_timings['get_episodes'].append(t_get_episodes)
        if not all_episodes:
            return 0.0

        if DEBUG:
            # バッチの内容を詳細に表示
            print(f"\n=== 学習時のエピソード内容 (サイズ: {len(all_episodes)}) ===")
            print(f"ReplayBufferから取得したエピソード数: {len(all_episodes)}")
            
            # バッチ全体の統計
            all_episode_lengths = [len(episode) for episode in all_episodes]
            print(f"エピソード長の統計:")
            print(f"  平均長: {np.mean(all_episode_lengths):.2f}")
            print(f"  標準偏差: {np.std(all_episode_lengths):.2f}")
            print(f"  最小長: {np.min(all_episode_lengths)}")
            print(f"  最大長: {np.max(all_episode_lengths)}")
            
            # 最初の5エピソードの詳細を表示
            for i in range(min(5, len(all_episodes))):
                episode = all_episodes[i]
                print(f"\nエピソード {i+1}:")
                print(f"  長さ: {len(episode)}")
                
                if len(episode) > 0:
                    # 最初と最後のTransitionの報酬を表示
                    first_reward = episode[0].reward
                    last_reward = episode[-1].reward
                    print(f"  最初の報酬: {first_reward}")
                    print(f"  最後の報酬: {last_reward}")
                    
                    # 全ての報酬の統計
                    all_rewards = [t.reward for t in episode]
                    rewards_array = np.array(all_rewards)
                    print(f"  報酬の平均: {np.mean(rewards_array, axis=0)}")
                    print(f"  報酬の標準偏差: {np.std(rewards_array, axis=0)}")
                    print(f"  報酬の最小値: {np.min(rewards_array, axis=0)}")
                    print(f"  報酬の最大値: {np.max(rewards_array, axis=0)}")
                    
                    # 行動の分布も確認
                    all_actions = [t.action for t in episode]
                    actions_array = np.array(all_actions)
                    unique_actions, counts = np.unique(actions_array, return_counts=True)
                    print(f"  行動の分布: {dict(zip(unique_actions, counts))}")
            
            if len(all_episodes) > 5:
                print(f"\n... 他 {len(all_episodes) - 5} エピソード")
            
            print("=" * 50)
        
        # 重複検出の統計
        initial_buffer_size = len(self.agent.experience_replay)
        added_episodes = 0
        skipped_episodes = 0
        
        # 全てのエピソードを経験再生バッファに追加
        # max_size は「総transition数」で律速する（長尺エピソード=1024job では episode数 10000
        # だと教師cacheが ~10M transitions=GPU OOM になるため）。短尺(24job)は従来通り上限10000。
        _tx_budget = int(os.environ.get("DISTRIBUTED_PCN_REPLAY_TX_BUDGET", "1200000"))
        _ep_len_est = max(1, len(all_episodes[0]) if all_episodes else 1)
        _replay_max = max(300, min(10000, _tx_budget // _ep_len_est))
        new_episodes_for_cache: List[List[Transition]] = []
        t_add_start = time.time()
        for episode in all_episodes:
            added_episode = self._add_episode(episode, max_size=_replay_max, step=self.global_step)
            if added_episode is not None:
                added_episodes += 1
                if not getattr(self.agent, "_training_batch_cache_stale", False):
                    new_episodes_for_cache.append(added_episode)
            else:
                skipped_episodes += 1
        t_add = time.time() - t_add_start
        if _PROFILE_MODE and hasattr(self, '_learn_timings'):
            self._learn_timings['add_episodes'].append(t_add)
        
        # 重複統計を表示
        if DEBUG:
            print(f"\n=== 重複検出統計 ===")
            print(f"処理したエピソード数: {len(all_episodes)}")
            print(f"追加されたエピソード数: {added_episodes}")
            print(f"スキップされたエピソード数: {skipped_episodes}")
            print(f"重複率: {skipped_episodes / len(all_episodes) * 100:.1f}%")
            print(f"初期バッファサイズ: {initial_buffer_size}")
            print(f"最終バッファサイズ: {len(self.agent.experience_replay)}")
            print(f"実効的な追加数: {len(self.agent.experience_replay) - initial_buffer_size}")
        
        # 修正: バッファ追加完了後にバッファサイズをチェック
        final_buffer_size = len(self.agent.experience_replay)
        self.agent.update_desired_return_normalization()
        # frozen-PF cloning: best-ever 非支配フロントを更新し、教師に常時含める（自己強化崩壊の遮断）
        self.agent.update_frozen_pf()
        # anchor-KL: イテレーション境界で方策スナップショットを更新（PCN_ANCHOR_KL_WEIGHT>0 のときのみ）
        self.agent.update_anchor_snapshot()
        if DEBUG:
            print(f"[Learner] バッファ追加完了後のサイズ: {final_buffer_size}")
        
        if final_buffer_size == 0:
            print("エラー: 経験再生バッファにエピソードが追加されていません。")
            return 0.0
        elif final_buffer_size < len(all_episodes):
            print(f"警告: 取得したエピソード数 {len(all_episodes)} に対して、Learnerのバッファには {final_buffer_size} 個しか追加されていません。")

        if (
            use_training_cache
            and _PHASE3_GPU_CACHE
            and not getattr(self, '_use_jax', False)
        ):
            sync = self.agent.sync_training_batch_cache(
                on_device=self.actual_device == "cuda",
                new_episodes=new_episodes_for_cache,
            )
            cache = sync.get("cache", {}) or {}
            cache_steps = int(sync.get("steps", 0))
            cache_mode = sync.get("mode", "reuse")
            cache_place = "GPU" if cache.get("on_device", False) else "CPU"
            cache_mb = float(cache.get("nbytes", 0)) / (1024 ** 2)
            n_new = int(sync.get("n_new_episodes", 0))
            mode_label = {
                "rebuild": "全件再構築",
                "extend": f"追記(+{n_new} ep)",
                "reuse": "再利用",
            }.get(cache_mode, cache_mode)
            print(
                f"Phase3教師データcache({mode_label}): {cache_steps} transitions "
                f"({cache_place}, {cache_mb:.1f} MB, PF重み対象={cache.get('pf_episode_count', 0)}, "
                f"端点重み対象={cache.get('endpoint_episode_count', 0)}, "
                f"Cost端重み対象={cache.get('cost_endpoint_episode_count', 0)}, "
                f"MidPF重み対象={cache.get('mid_pf_episode_count', 0)}, "
                f"LowWaitPF重み対象={cache.get('low_wait_pf_episode_count', 0)}, "
                f"Cost端action0率={cache.get('cost_endpoint_action0_rate', float('nan')):.3f}, "
                f"直近Achieved重み対象={cache.get('recent_episode_count', 0)})"
            )
        
        # 学習更新を実行
        t_update_start = time.time()
        if getattr(self, "_use_jax", False):
            for i in range(n_updates):
                try:
                    loss_value = self._jax_update_step()
                    if np.isnan(loss_value) or np.isinf(loss_value):
                        loss_value = 0.0
                    total_loss.append(loss_value)
                except Exception as e:
                    print(f"[Learner] エラー: JAX更新 {i}: {e}")
                    total_loss.append(0.0)
                self.global_step += 1
        else:
            try:
                # 段階的スケジュール(LR減衰 / PF重み)共通カウンタ。start_lr_schedule で _lr_sched_total を設定。
                _sched_tot = getattr(self, "_lr_sched_total", 0)
                _lr_override = None
                if _sched_tot > 0:
                    _sched_cnt = getattr(self, "_lr_sched_count", 0)
                    if _LR_DECAY_ON:
                        _lr_override = LEARNING_RATE * _lr_scale_for_iter(_sched_cnt, _sched_tot)
                    if _PF_WEIGHT_SCHED not in ("off", "", "0") and _PF_WEIGHT_PEAK_MUL > 1.0:
                        self.agent.set_pf_weight_mul(_pf_weight_mul_for_iter(_sched_cnt, _sched_tot))
                    self._lr_sched_count = _sched_cnt + 1
                mean_loss, batch_metrics, per_update_losses = self.agent.update_many(n_updates, learning_rate=_lr_override)
                total_loss.extend(per_update_losses)
                if isinstance(batch_metrics, dict):
                    if "policy_acc" in batch_metrics:
                        total_policy_acc.append(float(batch_metrics["policy_acc"]))
                    if "true_prob_mean" in batch_metrics:
                        total_true_prob.append(float(batch_metrics["true_prob_mean"]))
                    if "cmd_track_loss" in batch_metrics:
                        total_cmd_track.append(float(batch_metrics["cmd_track_loss"]))
                self.global_step += n_updates
            except Exception as e:
                print(f"[Learner] エラー: update_many 失敗: {e}")
                import traceback
                traceback.print_exc()
                total_loss.extend([0.0] * n_updates)
                self.global_step += n_updates
        t_update = time.time() - t_update_start
        if _PROFILE_MODE and hasattr(self, '_learn_timings'):
            self._learn_timings['update'].append(t_update)
            per_up = t_update / max(n_updates, 1)
            print(
                f"[PROFILE Learner] get={t_get_episodes:.3f}s add={t_add:.3f}s "
                f"update={t_update:.3f}s ({per_up:.4f}s/update, n_updates={n_updates})"
            )
        if total_policy_acc and total_true_prob:
            _ct_str = f", cmd_track_loss={np.mean(total_cmd_track):.4f}(n={len(total_cmd_track)})" if total_cmd_track else ", cmd_track_loss=NA"
            print(
                f"[Learner] policy_acc={np.mean(total_policy_acc):.4f}, "
                f"true_prob_mean={np.mean(total_true_prob):.4f}{_ct_str}"
            )
        
        # 学習後に重みのObjectRefを更新（全Actorで共有される）
        # 重みが更新された場合のみObjectRefを更新
        if total_loss and len(total_loss) > 0:
            self.update_weights_ref()
        
        return np.mean(total_loss) if total_loss else 0.0

    def get_learn_profile_summary(self) -> Dict[str, Any]:
        timings = getattr(self, "_learn_timings", None) or {}
        out: Dict[str, Any] = {}
        for key in ("get_episodes", "add_episodes", "update"):
            vals = timings.get(key) or []
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_p95"] = float(np.percentile(vals, 95))
                out[f"{key}_n"] = len(vals)
        return out

    def evaluate(self, max_return=None, n=10, training_iteration=None, eval_diag_path=None):
        """エージェントの評価を実行"""
        if max_return is None:
            max_return = _eval_max_return()
        if getattr(self, '_use_jax', False):
            from src.agents.pcn_jax import jax_params_to_pytorch_state_dict
            sd = jax_params_to_pytorch_state_dict(self._jax_params, scaling_factor=np.array([1, 1, 1]))
            # torch.compile 時は _orig_mod. プレフィックスが必要
            if any(k.startswith('_orig_mod.') for k in self.agent.model.state_dict().keys()):
                sd = {'_orig_mod.' + k: v for k, v in sd.items()}
            self.agent.model.load_state_dict(sd, strict=False)
        if DEBUG:
            print("評価を実行中...")
        eval_diag = None
        if eval_diag_path:
            eval_diag = {
                "path": eval_diag_path,
                "training_iteration": training_iteration,
                "lightweight": True,
            }
        _eval_save_history = os.environ.get("DISTRIBUTED_PCN_EVAL_SAVE_HISTORY", "1") == "1"
        _ema = self.agent.swap_in_ema_weights() if hasattr(self.agent, "swap_in_ema_weights") else False
        try:
            e_returns, e_value, distances, map_fin = self.agent.evaluate(
                self.env,
                max_return,
                n=n,
                save_history=_eval_save_history,
                eval_diag=eval_diag,
            )
        finally:
            if _ema:
                self.agent.restore_online_weights()
        
        # PCNエージェントのevaluate()で既に出力されているため、
        # ここでは追加の出力処理を行わず、結果のみを返す
        return e_returns, e_value, distances, map_fin  # 実数値はPCNエージェント側で処理済み

    def get_eval_targets(self, n: int):
        """分散評価用の (desired_return, horizon) リスト（Driver 側で Actor に配る）。"""
        episodes = self.agent._select_eval_target_episodes(n)
        return [
            (np.asarray(e[2][0].reward, dtype=np.float32), float(len(e[2])))
            for e in episodes
        ]

    def ingest_distributed_eval_results(self, results):
        """Actor 分散評価の結果を Learner 側で履歴へ反映。"""
        e_returns = []
        e_values = []
        for episode_return, value, _transitions, _map_fin in results:
            e_returns.append(episode_return)
            e_values.append(value)
        e_returns_np = np.array(e_returns, dtype=np.float64)
        e_values_np = np.array(e_values, dtype=np.float64)
        non_dominated_inds_reward = get_non_dominated_inds(e_returns_np)
        non_dominated_inds_values = get_non_dominated_inds_minimize(e_values_np)
        self.agent.evaluation_history.append({
            "all_returns": np.array(e_returns),
            "pareto_front_reward": e_returns_np[non_dominated_inds_reward],
            "pareto_front_values": e_values_np[non_dominated_inds_values],
            "values": e_values,
        })
        self.agent.evaluation_timestamps.append("1")
        self.agent.global_steps_at_evaluation.append(self.global_step)
        if not hasattr(self.agent, "wall_seconds_at_evaluation"):
            self.agent.wall_seconds_at_evaluation = []
        self.agent.wall_seconds_at_evaluation.append(
            float(time.perf_counter() - self._mo_hv_wall0)
        )
        return e_returns, e_values, [], None

    def prepare_uniform_grid_prep(self, n_jobs: int, grid: int):
        """均等格子の command 列と archive 点を Learner 上だけで準備（Actor は呼ばない）。"""
        from src.utils.pf_eval_gap import _uniform_grid_commands

        commands, ref_pts, exploration, _r0 = _uniform_grid_commands(
            self.agent, n_jobs, grid, 1.10
        )
        return {
            "commands": [(dr.tolist(), float(hz)) for dr, hz in commands],
            "ref_pts": ref_pts.tolist() if ref_pts.size else [],
            "exploration": exploration.tolist() if exploration.size else [],
        }

    def apply_eval_gap_boosts(self, boosts):
        self.agent.set_eval_gap_band_boosts(boosts if boosts else None)
        self.agent.mark_training_batch_cache_stale()

    def update_eval_gap_feedback(
        self,
        training_iteration: Optional[int] = None,
        plot_dir: Optional[str] = None,
        actors: Optional[List[Any]] = None,
    ):
        """均等 command Eval PF の弱点 cost 帯を検出し、次の教師 cache で replay 重みを増幅。"""
        if os.environ.get("PCN_EVAL_GAP_FEEDBACK", "0") != "1":
            return {}
        from src.utils.pf_eval_gap import compute_eval_gap_feedback

        n_jobs = int(getattr(self.env, "n_jobs", 1024))
        label = os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LABEL", "live")
        boosts, summary, _plot = compute_eval_gap_feedback(
            self.agent,
            self.env,
            n_jobs,
            plot_dir=plot_dir,
            plot_iteration=training_iteration,
            plot_label=label,
            actors=actors,
        )
        self.agent.set_eval_gap_band_boosts(boosts if boosts else None)
        self.agent.mark_training_batch_cache_stale()
        if boosts:
            parts = [f"{lo:.0g}-{hi:.0g}x{mult:.2f}" for lo, hi, mult in boosts]
            print(
                f"[EVAL_GAP] iter={training_iteration} boost_bands=[{', '.join(parts)}]"
            )
        return summary

    def save_live_uniform_pf_plot(
        self,
        training_iteration: Optional[int] = None,
        plot_dir: Optional[str] = None,
        actors: Optional[List[Any]] = None,
    ) -> Optional[str]:
        """Eval ギャップ FB なし時の均等格子 PF 図（DISTRIBUTED_PCN_LIVE_UNIFORM_PF=1）。"""
        from src.utils.pf_eval_gap import save_live_uniform_pf_plot

        n_jobs = int(getattr(self.env, "n_jobs", 1024))
        label = os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF_LABEL", "live")
        return save_live_uniform_pf_plot(
            self.agent,
            self.env,
            n_jobs,
            plot_dir or "",
            iteration=training_iteration,
            plot_label=label,
            actors=actors,
        )

    def evaluate_distributed(self, actors, max_return=None, n=10):
        """分散評価を実行"""
        if max_return is None:
            max_return = _eval_max_return()
        
        if DEBUG:
            print(f"分散評価を実行中... (n={n}, actors={len(actors)})")
        
        # 評価用の目標値を取得
        episodes = self.agent._nlargest(n)
        if len(episodes) == 0:
            print("警告: 評価用のエピソードが見つかりませんでした。")
            return [], [], [], None
        
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        
        # Actorに分散して評価を実行
        evaluation_futures = []
        for i, (desired_return, desired_horizon) in enumerate(zip(returns, horizons)):
            actor_id = i % len(actors)  # ラウンドロビンでActorに割り当て
            future = actors[actor_id].evaluate_episode.remote(desired_return, desired_horizon, max_return)
            evaluation_futures.append(future)
        
        # 全ての評価結果を収集
        results = ray.get(evaluation_futures)
        
        # 結果を整理
        e_returns = []
        e_values = []
        all_transitions = []
        
        for episode_return, value, transitions, map_fin in results:
            e_returns.append(episode_return)
            e_values.append(value)
            all_transitions.append(transitions)
        
        if DEBUG:
            print(f"分散評価完了: {len(e_returns)}エピソードを評価")
        
        # 非支配解を計算
        e_returns_np = np.array(e_returns, dtype=np.float64)
        e_values_np = np.array(e_values, dtype=np.float64)
        
        non_dominated_inds_reward = get_non_dominated_inds(e_returns_np)
        non_dominated_inds_values = get_non_dominated_inds_minimize(e_values_np)
        
        # 評価履歴に保存
        self.agent.evaluation_history.append({
            'all_returns': np.array(e_returns),
            'pareto_front_reward': e_returns_np[non_dominated_inds_reward],
            'pareto_front_values': e_values_np[non_dominated_inds_values],
            'values': e_values
        })
        self.agent.evaluation_timestamps.append("1")
        self.agent.global_steps_at_evaluation.append(self.global_step)
        if not hasattr(self.agent, "wall_seconds_at_evaluation"):
            self.agent.wall_seconds_at_evaluation = []
        self.agent.wall_seconds_at_evaluation.append(
            float(time.perf_counter() - self._mo_hv_wall0)
        )
        
        return e_returns, e_values, [], map_fin  # distancesは計算しない（分散評価では不要）

    def export_mo_hv_data(self) -> dict:
        """アルゴリズム比較用: 解空間パレート（pareto_front_values）と時系列メタデータを JSON 化可能な dict で返す。"""
        out = {
            "name": "pcn_distributed",
            "pareto_fronts_per_eval": [],
            "global_steps_at_evaluation": [],
            "wall_seconds_at_evaluation": [],
        }
        for h in self.agent.evaluation_history:
            v = h["pareto_front_values"]
            arr = np.asarray(v, dtype=np.float64)
            out["pareto_fronts_per_eval"].append(arr.tolist())
        out["global_steps_at_evaluation"] = [int(x) for x in self.agent.global_steps_at_evaluation]
        out["wall_seconds_at_evaluation"] = [
            float(x) for x in getattr(self.agent, "wall_seconds_at_evaluation", [])
        ]
        return out

    def _get_buffer_size(self) -> int:
        return len(self.agent.experience_replay)

    def log_gpu_memory_snapshot(self, tag: str) -> None:
        """Ray から呼び出し、Learner プロセス上の GPU メモリをログ。"""
        if self.actual_device != "cuda":
            print(f"[GPU_MEM] {tag} | Learner は CPU")
            return
        _log_gpu_memory_snapshot(tag)

    def update(self, learning_rate=None):
        """PCNエージェントのupdateメソッドを呼び出す（JAX時は_jax_update_step）"""
        if getattr(self, '_use_jax', False):
            loss_val = self._jax_update_step()
            loss = th.tensor(loss_val, dtype=th.float32) if loss_val is not None else None
            _ = None
        else:
            loss, _ = self.agent.update(learning_rate=learning_rate)
        if loss is not None:
            self.update_weights_ref()
        return loss, _

    def supervised_train_epoch(self, updates_per_epoch: int, learning_rate: float) -> Dict[str, Any]:
        """Phase2用: 1 epoch分の教師あり更新をLearner内でまとめて実行する。

        学習内容は従来と同じく PCN.update() の繰り返しだが、updateごとのRay往復と
        重みObjectRef更新を避け、epoch末に1回だけ共有重みを更新する。
        """
        self._ensure_cuda_perf()
        cache_steps = 0
        if getattr(self.agent, "_training_batch_cache", None) is None:
            cache_steps = self.agent.build_training_batch_cache(on_device=self.actual_device == "cuda")
        cache = getattr(self.agent, "_training_batch_cache", {}) or {}

        losses = []
        update_logs = []
        if getattr(self, "_use_jax", False):
            for update_i in range(updates_per_epoch):
                loss_value = float(self._jax_update_step())
                losses.append(loss_value)
                update_logs.append({"update": update_i + 1, "loss": loss_value, "metrics": {}})
        else:
            mean_loss, last_metrics, per_losses = self.agent.update_many(
                updates_per_epoch, learning_rate=learning_rate
            )
            losses = list(per_losses)
            for update_i, loss_value in enumerate(per_losses, start=1):
                update_logs.append({
                    "update": update_i,
                    "loss": float(loss_value),
                    "metrics": last_metrics if update_i == len(per_losses) else {},
                })

        if losses:
            self.update_weights_ref()

        observations = cache.get("observations")
        obs_dim = int(observations.shape[1]) if observations is not None and len(observations.shape) > 1 else 0
        return {
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "losses": losses,
            "updates": update_logs,
            "cached_steps": cache_steps,
            "cache_on_device": bool(cache.get("on_device", False)),
            "cache_mb": float(cache.get("nbytes", 0)) / (1024 ** 2),
            "obs_dim": obs_dim,
            "pf_episode_count": int(cache.get("pf_episode_count", 0)),
            "endpoint_episode_count": int(cache.get("endpoint_episode_count", 0)),
            "recent_episode_count": int(cache.get("recent_episode_count", 0)),
            "cost_endpoint_episode_count": int(cache.get("cost_endpoint_episode_count", 0)),
            "cost_endpoint_action0_rate": float(cache.get("cost_endpoint_action0_rate", float("nan"))),
        }

    def supervised_train_epochs(
        self,
        n_epochs: int,
        updates_per_epoch: int,
        learning_rate: float,
    ) -> Dict[str, Any]:
        """Phase2: 全 epoch を Learner 内で連続実行（epoch ごとの Ray 往復を削減）。"""
        self._ensure_cuda_perf()
        # 凍結runガード(PCN_FROZEN_RETRY>0で有効・既定OFF):
        # 初期NaN勾配ロックイン(ハズレ初期値→step不成立のまま完走=重み凍結)を Phase2 epoch1損失で検知し、
        # ネット再init→Phase2やり直し。正常損失0.3-5 vs 凍結44-50 なので閾値10で広いマージン。
        _frozen_retry_max = int(os.environ.get("PCN_FROZEN_RETRY", "0"))
        _frozen_loss_thr = float(os.environ.get("PCN_FROZEN_LOSS_THR", "10.0"))
        epoch_losses: List[float] = []
        epoch_summaries: List[Dict[str, Any]] = []
        cache_steps = 0
        cache: Dict[str, Any] = {}
        frozen_retries = 0

        for attempt in range(_frozen_retry_max + 1):
            epoch_losses = []
            epoch_summaries = []
            retry_triggered = False
            # skip率ベース検知(主検知器): 凍結=epoch内skip率~100% vs 健全~0%で完全分離・レシピ非依存。
            # 損失閾値(副検知器)はレシピ依存(PF重み/LSで健全runも5-9に底上げ)のため補助に格下げ。
            _sk0 = int(getattr(self.agent, "_nan_skip_total", 0))
            _st0 = int(getattr(self.agent, "_opt_step_total", 0))
            for epoch in range(n_epochs):
                epoch_result = self.supervised_train_epoch(
                    updates_per_epoch=updates_per_epoch,
                    learning_rate=learning_rate,
                )
                epoch_summaries.append(epoch_result)
                _eloss = float(epoch_result.get("avg_loss", 0.0))
                epoch_losses.append(_eloss)
                if epoch == 0:
                    cache_steps = int(epoch_result.get("cached_steps", 0) or 0)
                    cache = getattr(self.agent, "_training_batch_cache", {}) or {}
                    _d_sk = int(getattr(self.agent, "_nan_skip_total", 0)) - _sk0
                    _d_st = int(getattr(self.agent, "_opt_step_total", 0)) - _st0
                    _skip_rate = _d_sk / max(1, _d_sk + _d_st)
                    if (attempt < _frozen_retry_max
                            and (_skip_rate > 0.5
                                 or not np.isfinite(_eloss) or _eloss > _frozen_loss_thr)):
                        frozen_retries += 1
                        print(
                            f"[FROZEN_RETRY] Phase2 epoch1 skip率={_skip_rate:.1%} (skip={_d_sk}/step={_d_st}) "
                            f"損失={_eloss:.2f} → 凍結初期値と判定。ネット再init→Phase2再試行 ({frozen_retries}/{_frozen_retry_max})"
                        )
                        self.agent.reinit_network()
                        retry_triggered = True
                        break
            if not retry_triggered:
                break

        obs_dim = 0
        observations = cache.get("observations")
        if observations is not None and len(observations.shape) > 1:
            obs_dim = int(observations.shape[1])

        return {
            "n_epochs": n_epochs,
            "epoch_losses": epoch_losses,
            "avg_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "best_loss": float(np.min(epoch_losses)) if epoch_losses else 0.0,
            "epochs": epoch_summaries,
            "cached_steps": cache_steps,
            "cache_on_device": bool(cache.get("on_device", False)),
            "cache_mb": float(cache.get("nbytes", 0)) / (1024 ** 2),
            "obs_dim": obs_dim,
            "frozen_retries": frozen_retries,
        }

    def _phase2_importance_groups(self, obs_dim: int) -> List[Dict[str, Any]]:
        """Phase2重要度可視化用の解釈しやすい入力グループを作る。"""
        urgency_extra = 1 if os.environ.get("SCHEDULER_OBS_URGENCY", "0") == "1" else 0
        occupancy_extra = 1 if os.environ.get("SCHEDULER_OBS_OCCUPANCY", "0") == "1" else 0
        event_obs_dim = N_EVENTS_OBS * EVENT_FEATURES + JOB_QUEUE_SIZE + urgency_extra + occupancy_extra
        is_event_vector = _USE_EVENT_OBS and not learner_bitmap_enabled() and obs_dim == event_obs_dim

        if is_event_vector:
            return self._phase2_event_importance_groups(obs_dim)

        obs_window_size = int(getattr(self.env, "obs_window_size", 10))
        onpre_size = int(self.env.n_on_premise_node) * obs_window_size
        cloud_size = int(self.env.n_cloud_node) * obs_window_size
        map_total = onpre_size + cloud_size

        groups: List[Dict[str, Any]] = []

        def add_obs_group(name: str, start: int, end: int, description: str) -> None:
            start = max(0, min(int(start), obs_dim))
            end = max(start, min(int(end), obs_dim))
            if end > start:
                groups.append({
                    "name": name,
                    "kind": "observation",
                    "start": start,
                    "end": end,
                    "n_features": end - start,
                    "description": description,
                })

        add_obs_group("on_premise_map", 0, onpre_size, "オンプレミス資源マップ")
        add_obs_group("cloud_map", onpre_size, map_total, "クラウド資源マップ")
        add_obs_group("job_queue_all", map_total, obs_dim, "ジョブキュー全体")

        job_dim = 8
        for job_i in range(5):
            start = map_total + job_i * job_dim
            add_obs_group(
                f"job_queue_{job_i}",
                start,
                start + job_dim,
                f"ジョブキュー位置{job_i}の8特徴量",
            )

        groups.extend([
            {
                "name": "desired_return_wait",
                "kind": "desired_return",
                "index": 0,
                "n_features": 1,
                "description": "条件入力: 待ち時間側の残り目標報酬",
            },
            {
                "name": "desired_return_cost",
                "kind": "desired_return",
                "index": 1,
                "n_features": 1,
                "description": "条件入力: コスト側の残り目標報酬",
            },
            {
                "name": "desired_horizon",
                "kind": "desired_horizon",
                "n_features": 1,
                "description": "条件入力: 残りホライゾン",
            },
        ])
        return groups

    def _phase2_event_importance_groups(self, obs_dim: int) -> List[Dict[str, Any]]:
        """イベント観測ベクトル用のPhase2重要度グループを作る。"""
        groups: List[Dict[str, Any]] = []

        def add_range_group(name: str, start: int, end: int, description: str) -> None:
            start = max(0, min(int(start), obs_dim))
            end = max(start, min(int(end), obs_dim))
            if end > start:
                groups.append({
                    "name": name,
                    "kind": "observation",
                    "start": start,
                    "end": end,
                    "n_features": end - start,
                    "description": description,
                })

        def add_index_group(name: str, indices: List[int], description: str) -> None:
            valid = [int(i) for i in indices if 0 <= int(i) < obs_dim]
            if valid:
                groups.append({
                    "name": name,
                    "kind": "observation",
                    "indices": valid,
                    "n_features": len(valid),
                    "description": description,
                })

        events_size = N_EVENTS_OBS * EVENT_FEATURES
        job_start = events_size
        add_range_group("events_all", 0, events_size, "イベント列全体")

        event_feature_names = [
            ("event_start_time_all", 0, "全イベントの開始時刻"),
            ("event_end_time_all", 1, "全イベントの終了時刻"),
            ("event_duration_all", 2, "全イベントの実行時間"),
            ("event_use_cloud_all", 3, "全イベントのクラウド利用フラグ"),
            ("event_start_node_all", 4, "全イベントの開始ノード"),
            ("event_job_height_all", 5, "全イベントのジョブ高さ"),
        ]
        for name, offset, description in event_feature_names:
            add_index_group(
                name,
                [event_i * EVENT_FEATURES + offset for event_i in range(N_EVENTS_OBS)],
                description,
            )

        for event_i in range(N_EVENTS_OBS):
            start = event_i * EVENT_FEATURES
            add_range_group(
                f"event_slot_{event_i:02d}",
                start,
                start + EVENT_FEATURES,
                f"イベントスロット{event_i}の6特徴量",
            )

        add_range_group("job_queue_all", job_start, job_start + JOB_QUEUE_SIZE, "ジョブキュー全体")
        job_dim = 8
        for job_i in range(5):
            start = job_start + job_i * job_dim
            add_range_group(
                f"job_queue_{job_i}",
                start,
                start + job_dim,
                f"ジョブキュー位置{job_i}の8特徴量",
            )

        # urgency obs (SCHEDULER_OBS_URGENCY=1) は events+job_queue の後ろに +1 次元付く
        urgency_start = events_size + JOB_QUEUE_SIZE
        if urgency_start < obs_dim:
            add_range_group("urgency_pred_wait", urgency_start, obs_dim, "予測待ち時間(urgency, 末尾+1)")

        groups.extend([
            {
                "name": "desired_return_wait",
                "kind": "desired_return",
                "index": 0,
                "n_features": 1,
                "description": "条件入力: 待ち時間側の残り目標報酬",
            },
            {
                "name": "desired_return_cost",
                "kind": "desired_return",
                "index": 1,
                "n_features": 1,
                "description": "条件入力: コスト側の残り目標報酬",
            },
            {
                "name": "desired_horizon",
                "kind": "desired_horizon",
                "n_features": 1,
                "description": "条件入力: 残りホライゾン",
            },
        ])
        return groups

    def _policy_action1_prob(self, observations, desired_returns, desired_horizons):
        desired_horizons = desired_horizons.unsqueeze(1)
        if USE_ENHANCED_MODEL:
            output = self.agent.network(observations, desired_returns, desired_horizons)
            logits = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logits, dim=-1)
        else:
            log_probs = self.agent.model(observations, desired_returns, desired_horizons)
            probs = th.exp(log_probs)
        if probs.shape[-1] < 2:
            return probs.squeeze(-1), th.zeros(probs.shape[0], dtype=th.long, device=probs.device)
        return probs[:, 1], th.argmax(probs, dim=-1)

    def export_phase2_feature_importance(self, save_dir: str, max_samples: int = 1024) -> Dict[str, Any]:
        """Phase2終了時点のモデルで、入力グループ置換による方策感度を出力する。

        決定木のfeature importanceとは異なり、NNの事後解析として各入力群を
        バッチ平均に置換し、action=1確率がどれだけ変わるかを測る。
        """
        if getattr(self, "_use_jax", False):
            return {"enabled": False, "reason": "JAX learner is not supported for this analysis"}

        os.makedirs(save_dir, exist_ok=True)
        cache = getattr(self.agent, "_training_batch_cache", None)
        if cache is None:
            self.agent.build_training_batch_cache(on_device=self.actual_device == "cuda")
            cache = getattr(self.agent, "_training_batch_cache", None)
        if not cache:
            return {"enabled": False, "reason": "training cache is empty"}

        n_steps = int(cache["episode_offsets"][-1] + cache["episode_lengths"][-1])
        n_samples = max(1, min(int(max_samples), n_steps))
        sample_indices = self.agent.np_random.choice(n_steps, size=n_samples, replace=n_steps < n_samples)

        def take_cache(name: str, dtype, device):
            value = cache[name]
            if th.is_tensor(value):
                idx_t = th.as_tensor(sample_indices, dtype=th.long, device=value.device)
                return value.index_select(0, idx_t).to(device=device, dtype=dtype)
            return th.as_tensor(value[sample_indices], dtype=dtype, device=device)

        device = th.device(self.actual_device)
        observations = take_cache("observations", th.float32, device)
        actions = take_cache("actions", th.long, device)
        desired_returns = take_cache("desired_returns", th.float32, device)
        desired_horizons = take_cache("desired_horizons", th.float32, device)

        model = self.agent.network if USE_ENHANCED_MODEL else self.agent.model
        was_training = model.training
        model.eval()

        rows: List[Dict[str, Any]] = []
        with th.no_grad():
            base_prob, base_action = self._policy_action1_prob(observations, desired_returns, desired_horizons)
            base_prob_mean = float(base_prob.mean().item())
            base_acc = float((base_action == actions).float().mean().item())

            obs_mean = observations.mean(dim=0, keepdim=True)
            return_mean = desired_returns.mean(dim=0, keepdim=True)
            horizon_mean = desired_horizons.mean().view(1)

            for group in self._phase2_importance_groups(observations.shape[1]):
                obs_mod = observations
                ret_mod = desired_returns
                horizon_mod = desired_horizons

                if group["kind"] == "observation":
                    obs_mod = observations.clone()
                    if "indices" in group:
                        indices = th.as_tensor(group["indices"], dtype=th.long, device=observations.device)
                        obs_mod.index_copy_(1, indices, obs_mean.index_select(1, indices).expand(n_samples, -1))
                    else:
                        obs_mod[:, group["start"]:group["end"]] = obs_mean[:, group["start"]:group["end"]]
                elif group["kind"] == "desired_return":
                    ret_mod = desired_returns.clone()
                    idx = int(group["index"])
                    if idx < ret_mod.shape[1]:
                        ret_mod[:, idx] = return_mean[:, idx]
                elif group["kind"] == "desired_horizon":
                    horizon_mod = desired_horizons.clone()
                    horizon_mod[:] = horizon_mean

                ablated_prob, ablated_action = self._policy_action1_prob(obs_mod, ret_mod, horizon_mod)
                delta = base_prob - ablated_prob
                rows.append({
                    "group": group["name"],
                    "kind": group["kind"],
                    "n_features": int(group["n_features"]),
                    "description": group["description"],
                    "baseline_action1_prob": base_prob_mean,
                    "ablated_action1_prob": float(ablated_prob.mean().item()),
                    "mean_delta_action1_prob": float(delta.mean().item()),
                    "mean_abs_delta_action1_prob": float(delta.abs().mean().item()),
                    "action_flip_rate": float((base_action != ablated_action).float().mean().item()),
                })

        if was_training:
            model.train()

        total_abs = sum(row["mean_abs_delta_action1_prob"] for row in rows)
        for row in rows:
            row["relative_importance"] = (
                row["mean_abs_delta_action1_prob"] / total_abs if total_abs > 0 else 0.0
            )

        rows_sorted = sorted(rows, key=lambda x: x["mean_abs_delta_action1_prob"], reverse=True)
        csv_path = os.path.join(save_dir, "phase2_feature_importance.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            columns = [
                "group", "kind", "n_features", "description",
                "baseline_action1_prob", "ablated_action1_prob",
                "mean_delta_action1_prob", "mean_abs_delta_action1_prob",
                "relative_importance", "action_flip_rate",
            ]
            f.write(",".join(columns) + "\n")
            for row in rows_sorted:
                values = []
                for col in columns:
                    value = row[col]
                    if isinstance(value, str):
                        values.append('"' + value.replace('"', '""') + '"')
                    elif isinstance(value, float):
                        values.append(f"{value:.8g}")
                    else:
                        values.append(str(value))
                f.write(",".join(values) + "\n")

        txt_path = os.path.join(save_dir, "phase2_feature_importance_readme.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Phase2 feature importance\n")
            f.write("=========================\n")
            f.write("各入力グループを教師データサンプル内の平均値へ置換し、P(action=1)の変化を測定した事後解析です。\n")
            f.write("mean_abs_delta_action1_prob が大きいほど、その入力群がPhase2終了時点の方策判断を強く動かしています。\n")
            f.write("mean_delta_action1_prob > 0 は、その入力群が元の状態では平均的に action=1 確率を上げていたことを示します。\n")
            f.write(f"samples={n_samples}, baseline_action1_prob={base_prob_mean:.6f}, baseline_acc={base_acc:.6f}\n")

        def save_importance_plot(path: str, plot_rows: List[Dict[str, Any]], title: str) -> None:
            labels = [row["group"] for row in plot_rows][::-1]
            values = [row["mean_abs_delta_action1_prob"] for row in plot_rows][::-1]
            colors = [
                "#2f7ed8" if row["mean_delta_action1_prob"] >= 0 else "#d94f45"
                for row in plot_rows
            ][::-1]
            plt.figure(figsize=(9, max(4, 0.42 * len(labels) + 1.5)))
            plt.barh(labels, values, color=colors, alpha=0.85)
            plt.xlabel("mean abs change in P(action=1)")
            plt.title(title)
            plt.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

        png_path = os.path.join(save_dir, "phase2_feature_importance.png")
        png_all_path = os.path.join(save_dir, "phase2_feature_importance_all.png")
        save_importance_plot(
            png_path,
            rows_sorted[:12],
            "Phase2 Policy Input Group Importance (Top 12)",
        )
        save_importance_plot(
            png_all_path,
            rows_sorted,
            "Phase2 Policy Input Group Importance (All Groups)",
        )

        print(
            f"Phase2入力重要度を保存: {png_path} "
            f"(samples={n_samples}, baseline_action1_prob={base_prob_mean:.3f}, acc={base_acc:.3f})"
        )
        if rows_sorted:
            top = rows_sorted[0]
            print(
                f"Phase2入力重要度Top: {top['group']} "
                f"abs_delta={top['mean_abs_delta_action1_prob']:.4f}, "
                f"delta={top['mean_delta_action1_prob']:.4f}"
            )

        return {
            "enabled": True,
            "csv_path": csv_path,
            "png_path": png_path,
            "png_all_path": png_all_path,
            "txt_path": txt_path,
            "n_samples": n_samples,
            "baseline_action1_prob": base_prob_mean,
            "baseline_acc": base_acc,
            "top_group": rows_sorted[0]["group"] if rows_sorted else None,
        }

    def clear_training_batch_cache(self) -> None:
        self.agent.clear_training_batch_cache()

    def warmup_training_batch_cache(self) -> Dict[str, Any]:
        """Phase3開始前: 既存 replay から教師 cache を1回だけ整える（以降は差分追記）。"""
        if getattr(self, "_use_jax", False) or not _PHASE3_GPU_CACHE:
            return {"steps": 0, "mode": "skip"}
        sync = self.agent.sync_training_batch_cache(
            on_device=self.actual_device == "cuda",
            new_episodes=[],
            force_rebuild=bool(getattr(self.agent, "_training_batch_cache_stale", False)),
        )
        cache = sync.get("cache", {}) or {}
        cache_place = "GPU" if cache.get("on_device", False) else "CPU"
        cache_mb = float(cache.get("nbytes", 0)) / (1024 ** 2)
        print(
            f"Phase3教師データcacheウォームアップ({sync.get('mode', 'reuse')}): "
            f"{sync.get('steps', 0)} transitions ({cache_place}, {cache_mb:.1f} MB)"
        )
        return sync

    def get_global_step(self) -> int:
        """グローバルステップを取得"""
        return self.global_step

    def get_experience_replay(self):
        """experience replayの内容を取得（コピーを返す）"""
        # experience_replayの内容をコピーして返す
        replay_copy = []
        for priority, unique_step, transitions in self.agent.experience_replay:
            # transitionsのコピーを作成
            transitions_copy = []
            for t in transitions:
                reward_copy = np.array(t.reward, copy=True)
                t_copy = Transition(
                    observation=t.observation,
                    action=t.action,
                    reward=reward_copy,
                    next_observation=t.next_observation,
                    terminal=t.terminal
                )
                # objective_values属性もコピー
                if hasattr(t, 'objective_values'):
                    t_copy.objective_values = t.objective_values
                transitions_copy.append(t_copy)
            replay_copy.append((priority, unique_step, transitions_copy))
        
        if DEBUG:
            print(f"[Learner] experience_replayの内容を取得: {len(replay_copy)} エピソード")
        
        return replay_copy

    def save_replay_snapshot(self, path: str) -> Dict[str, Any]:
        """Learner replay を gzip pickle で保存（オフライン PF 評価用）。"""
        episodes = []
        for _prio, _step, transitions in self.agent.experience_replay:
            if transitions:
                episodes.append(transitions)
        n_jobs = int(self.config["param_env"].get("n_jobs", N_JOBS))
        payload = {
            "metadata": {
                "n_jobs": n_jobs,
                "n_episodes": len(episodes),
                "use_event_obs": bool(_USE_EVENT_OBS),
                "learner_bitmap": bool(learner_bitmap_enabled()),
            },
            "episodes": episodes,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # compresslevel=6: end-of-run 1回の後処理。9→6で保存 ~4倍速・サイズ +2〜3%のみ（展開後は同一）。
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        nbytes = _estimate_episodes_numpy_bytes(episodes)
        print(
            f"[Learner] replay スナップショット保存: {path} "
            f"({len(episodes)} episodes, n_jobs={n_jobs})"
        )
        return {
            "path": path,
            "n_episodes": len(episodes),
            "n_jobs": n_jobs,
            "numpy_payload_bytes": int(nbytes),
        }

    def save_learning_data_to_file(self, filename="learning_data_debug.txt", sample_size=100):
        """学習データの詳細をファイルに書き込む（リモートメソッド）"""
        try:
            return self.agent.save_learning_data_to_file(filename, sample_size)
        except Exception as e:
            print(f"[Learner] 学習データ保存エラー: {e}")
            return None

    def export_learning_samples_to_csv(self, filename="learning_samples.csv", num_samples=1000):
        """学習サンプルをCSVファイルにエクスポート（リモートメソッド）"""
        try:
            return self.agent.export_learning_samples_to_csv(filename, num_samples)
        except Exception as e:
            print(f"[Learner] CSVエクスポートエラー: {e}")
            return None

    def save_model(self, save_path):
        """モデルを指定パスに保存（リモートメソッド）。EMA有効時はEMA重みを保存(eval重みと一致させる)。"""
        _ema = self.agent.swap_in_ema_weights() if hasattr(self.agent, "swap_in_ema_weights") else False
        try:
            import torch
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if getattr(self, '_use_jax', False):
                from src.agents.pcn_jax import jax_params_to_pytorch_state_dict
                model_state_dict = jax_params_to_pytorch_state_dict(self._jax_params, scaling_factor=np.array([1, 1, 1]))
            elif USE_ENHANCED_MODEL and hasattr(self.agent, 'network'):
                model_state_dict = self.agent.network.state_dict()
            else:
                model_state_dict = self.agent.model.state_dict()
            model_state = {
                'model_state_dict': model_state_dict,
                'global_step': self.global_step,
                'config': self.config,
                'model_type': 'EnhancedPCNModel' if USE_ENHANCED_MODEL else 'DiscreteActionsDefaultModel',
                'device': self.actual_device,
                'experience_replay_size': len(self.agent.experience_replay)
            }
            
            # ターゲットモデルがある場合は保存
            if hasattr(self.agent, 'target_model') and self.agent.target_model is not None:
                model_state['target_model_state_dict'] = self.agent.target_model.state_dict()
            
            torch.save(model_state, save_path)
            
            if DEBUG:
                print(f"[Learner] モデルを保存しました: {save_path}")
                print(f"  グローバルステップ: {self.global_step}")
                print(f"  モデルタイプ: {model_state['model_type']}")
                print(f"  経験再生バッファサイズ: {model_state['experience_replay_size']}")
            
            return save_path
            
        except Exception as e:
            print(f"[Learner] モデル保存エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if _ema:
                self.agent.restore_online_weights()

# =========================
# 4. ユーティリティ関数
# =========================

_VIS_PLOT_DEDUPE_DECIMALS = 6


def _unique_row_mask_for_plot(points: np.ndarray, decimals: int = _VIS_PLOT_DEDUPE_DECIMALS) -> np.ndarray:
    """可視化用: 座標が重なる行のうち先頭のみ残すマスク（学習・評価データは変更しない）。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    keep = np.ones(len(arr), dtype=bool)
    seen = set()
    for i, row in enumerate(arr):
        key = tuple(np.round(row, decimals))
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    return keep


def _dedupe_points_for_plot(points: np.ndarray, decimals: int = _VIS_PLOT_DEDUPE_DECIMALS) -> np.ndarray:
    """可視化用: 重複座標の点を除いた配列を返す。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[_unique_row_mask_for_plot(arr, decimals)]


def _dedupe_aligned_points_for_plot(
    *arrays: np.ndarray,
    decimals: int = _VIS_PLOT_DEDUPE_DECIMALS,
) -> Tuple[np.ndarray, ...]:
    """可視化用: 先頭配列の座標重複に合わせ、複数系列の行を同期して間引く。"""
    if not arrays:
        return tuple()
    ref = np.asarray(arrays[0], dtype=np.float64)
    if ref.size == 0:
        return tuple(np.asarray(a) for a in arrays)
    mask = _unique_row_mask_for_plot(ref, decimals)
    return tuple(np.asarray(a)[mask] for a in arrays)


def visualize_initial_pareto_front(initial_batch, save_dir="pareto_front_visualization"):
    """ 
    初期経験収集後のパレートフロントを可視化する関数
    
    Args:
        initial_batch: 初期エピソードのリスト（Learnerの経験再生バッファ形式）
        save_dir: 保存ディレクトリ
        
    Returns:
        dict: 軸範囲の情報を含む辞書
    """
    # 初期エピソードから累積報酬と実数値を計算
    initial_e_returns = []
    initial_e_values = []

    for episode in initial_batch:
        # episode[2]が遷移のリスト
        transitions = episode[2]
        if len(transitions) > 0:
            # エピソードの累積報酬を計算
            episode_return = np.sum([t.reward for t in transitions], axis=0)
            initial_e_returns.append(episode_return)
            
            # エピソードの実数値（コストと実行時間）を取得
            # objective_values属性が存在するかチェック
            if hasattr(transitions[0], 'objective_values') and transitions[0].objective_values is not None:
                # Transitionオブジェクトから実数値を取得
                cost,_,avg_waiting_time = transitions[0].objective_values
                initial_e_values.append([cost,avg_waiting_time])
            else:
                # objective_valuesが存在しない場合は、報酬から推定値を計算
                # 報酬の累積値を実数値として使用（仮の対応）
                episode_return = np.sum([t.reward for t in transitions], axis=0)
                # 報酬を負の値に変換して最小化問題として扱う
                initial_e_values.append([-episode_return[0], -episode_return[1]])
                
                # デバッグ情報を表示（最初の数エピソードのみ）
                if len(initial_e_returns) <= 3:
                    print(f"エピソード {len(initial_e_returns)}: objective_valuesが見つかりません")
                    print(f"  報酬: {episode_return}")
                    print(f"  推定実数値: {[-episode_return[0], -episode_return[1]]}")

    # 軸範囲を計算するための辞書
    axis_ranges = {}
    
    if len(initial_e_returns) > 0 and len(initial_e_values) > 0:
        # 非支配解の計算
        initial_non_dominated_inds = get_non_dominated_inds(np.array(initial_e_returns))
        initial_non_dominated_inds_values = get_non_dominated_inds_minimize(np.array(initial_e_values))
        
        # 報酬空間の軸範囲を計算
        all_returns = np.array(initial_e_returns)
        reward_x_min, reward_x_max = all_returns[:, 0].min(), all_returns[:, 0].max()
        reward_y_min, reward_y_max = all_returns[:, 1].min(), all_returns[:, 1].max()
        
        # マージンを追加（10%）
        reward_x_margin = (reward_x_max - reward_x_min) * 0.1
        reward_y_margin = (reward_y_max - reward_y_min) * 0.1
        
        axis_ranges['rewards'] = {
            'x_min': reward_x_min - reward_x_margin,
            'x_max': reward_x_max + reward_x_margin,
            'y_min': reward_y_min - reward_y_margin,
            'y_max': reward_y_max + reward_y_margin
        }
        
        # 実数値空間の軸範囲を計算
        all_values = np.array(initial_e_values)
        values_x_min, values_x_max = all_values[:, 0].min(), all_values[:, 0].max()
        values_y_min, values_y_max = all_values[:, 1].min(), all_values[:, 1].max()
        
        # マージンを追加（10%）
        values_x_margin = (values_x_max - values_x_min) * 0.1
        values_y_margin = (values_y_max - values_y_min) * 0.1
        
        axis_ranges['values'] = {
            'x_min': values_x_min - values_x_margin,
            'x_max': values_x_max + values_x_margin,
            'y_min': values_y_min - values_y_margin,
            'y_max': values_y_max + values_y_margin
        }
        
        # 1. 報酬空間でのパレートフロント（最大化目的）
        plt.figure(figsize=(8, 6))
        
        all_returns_vis = _dedupe_points_for_plot(all_returns)
        non_dominated_inds_vis = get_non_dominated_inds(all_returns_vis)
        pareto_front_returns_vis = all_returns_vis[non_dominated_inds_vis]
        plt.scatter(all_returns_vis[:, 0], all_returns_vis[:, 1], c='lightblue', alpha=0.6, label='All Solutions', s=50)
        plt.scatter(
            pareto_front_returns_vis[:, 0], pareto_front_returns_vis[:, 1],
            c='red', s=100, label='Pareto Front', zorder=5,
        )
        
        if len(pareto_front_returns_vis) > 1:
            sorted_indices = np.lexsort((pareto_front_returns_vis[:, 1], pareto_front_returns_vis[:, 0]))
            sorted_pareto = pareto_front_returns_vis[sorted_indices]
            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
        
        plt.xlim(axis_ranges['rewards']['x_min'], axis_ranges['rewards']['x_max'])
        plt.ylim(axis_ranges['rewards']['y_min'], axis_ranges['rewards']['y_max'])
        
        plt.title(
            f'Initial Random Experience - Pareto Front (Reward)\n'
            f'Non-dominated: {len(non_dominated_inds_vis)} (unique points)',
            fontsize=12,
        )
        plt.xlabel('Reward 1', fontsize=10)
        plt.ylabel('Reward 2', fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        reward_plot_path = f"{save_dir}/pareto_front_rewards_initial_random.png"
        plt.savefig(reward_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        if DEBUG:
            print(f"初期ランダム経験の報酬空間パレートフロントを保存: {reward_plot_path}")
        
        # 2. 実数値空間でのパレートフロント（最小化目的）
        plt.figure(figsize=(8, 6))
        
        all_values_vis = _dedupe_points_for_plot(all_values)
        non_dominated_inds_values_vis = get_non_dominated_inds_minimize(all_values_vis)
        pareto_front_values_vis = all_values_vis[non_dominated_inds_values_vis]
        plt.scatter(all_values_vis[:, 0], all_values_vis[:, 1], c='lightgreen', alpha=0.6, label='All Solutions', s=50)
        plt.scatter(
            pareto_front_values_vis[:, 0], pareto_front_values_vis[:, 1],
            c='red', s=100, label='Pareto Front', zorder=5,
        )
        
        if len(pareto_front_values_vis) > 1:
            sorted_indices = np.lexsort((pareto_front_values_vis[:, 1], pareto_front_values_vis[:, 0]))
            sorted_pareto = pareto_front_values_vis[sorted_indices]
            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
        
        plt.xlim(axis_ranges['values']['x_min'], axis_ranges['values']['x_max'])
        plt.ylim(axis_ranges['values']['y_min'], axis_ranges['values']['y_max'])
        
        plt.title(
            f'Initial Random Experience - Pareto Front (Value)\n'
            f'Non-dominated: {len(non_dominated_inds_values_vis)} (unique points)',
            fontsize=12,
        )
        plt.xlabel('Cost (Minimize)', fontsize=10)
        plt.ylabel('Execution Time (Minimize)', fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        values_plot_path = f"{save_dir}/pareto_front_values_initial_random.png"
        plt.savefig(values_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        if DEBUG:
            print(f"初期ランダム経験の実数値空間パレートフロントを保存: {values_plot_path}")
        
        # 3. 詳細データの保存
        details_path = f"{save_dir}/pareto_front_details_initial_random.txt"
        with open(details_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 初期ランダム経験パレートフロント詳細 ===\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"評価サンプル数: {len(initial_e_returns)}\n")
            f.write(f"報酬空間非支配解数: {len(initial_non_dominated_inds)}\n")
            f.write(f"実数値空間非支配解数: {len(initial_non_dominated_inds_values)}\n")
            f.write(f"初期エピソード数: {len(initial_batch)}\n")
            f.write(f"報酬空間軸範囲: X[{axis_ranges['rewards']['x_min']:.4f}, {axis_ranges['rewards']['x_max']:.4f}], Y[{axis_ranges['rewards']['y_min']:.4f}, {axis_ranges['rewards']['y_max']:.4f}]\n")
            f.write(f"実数値空間軸範囲: X[{axis_ranges['values']['x_min']:.4f}, {axis_ranges['values']['x_max']:.4f}], Y[{axis_ranges['values']['y_min']:.4f}, {axis_ranges['values']['y_max']:.4f}]\n")
            
            # 報酬空間の非支配解を詳細に記録
            f.write(f"\n=== 報酬空間の非支配解 ===\n")
            for i, idx in enumerate(initial_non_dominated_inds):
                f.write(f"解{i+1}: {initial_e_returns[idx]}\n")
            
            # 実数値空間の非支配解を詳細に記録
            f.write(f"\n=== 実数値空間の非支配解 ===\n")
            for i, idx in enumerate(initial_non_dominated_inds_values):
                f.write(f"解{i+1}: {initial_e_values[idx]}\n")
        
        if DEBUG:
            print(f"初期ランダム経験の詳細データを保存: {details_path}")
            print(f"=== 初期経験収集後の可視化完了 ===")
    
    else:
        if DEBUG:
            print("初期経験が不足しているため、可視化をスキップします")
    
    return axis_ranges

# =========================
# 5. 実行スクリプト
# =========================
def main():
    
    import matplotlib.pyplot as plt
    import os

    save_phase1_cache = "--save-phase1" in sys.argv[1:]
    load_phase1_cache = "--load-phase1" in sys.argv[1:]
    
    # Singularity等: 作業ディレクトリを環境変数で指定可能（相対パス解決の基準）
    workdir = os.environ.get('DISTRIBUTED_PCN_WORKDIR')
    if workdir and os.path.isdir(workdir):
        os.chdir(workdir)
        if DEBUG:
            print(f"[DISTRIBUTED_PCN] 作業ディレクトリ: {os.getcwd()}")
    
    # 実行用のディレクトリを作成
    # 通常は保存ルートだけを指定し、その下にプログラムが日時順のrunディレクトリを作る。
    # 例: experiments/distributed_pcn/20260527_153012
    # 完全に固定したい場合だけ DISTRIBUTED_PCN_RUN_DIR を指定する。
    explicit_run_dir = os.environ.get('DISTRIBUTED_PCN_RUN_DIR')
    output_base = os.environ.get('DISTRIBUTED_PCN_OUTPUT_DIR', 'experiments/distributed_pcn')
    if explicit_run_dir:
        execution_dir = explicit_run_dir
    else:
        execution_dir = _make_timestamped_run_dir(output_base)
    os.makedirs(execution_dir, exist_ok=True)
    run_log_file = open(os.path.join(execution_dir, "pcn_run.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, run_log_file)
    sys.stderr = _TeeStream(sys.stderr, run_log_file)

    initial_episode_cache_path = None
    if load_phase1_cache:
        initial_episode_cache_path = os.environ.get("DISTRIBUTED_PCN_INITIAL_EPISODE_CACHE_PATH")
        if not initial_episode_cache_path:
            initial_episode_cache_path = _find_initial_episode_cache(output_base, execution_dir)
    if initial_episode_cache_path:
        print(f"[INITIAL_EPISODES] 学習用キャッシュ候補: {initial_episode_cache_path}")
    elif not load_phase1_cache:
        print("[INITIAL_EPISODES] --load-phase1 未指定のため、学習用キャッシュは読み込みません")
    else:
        if save_phase1_cache:
            print("[INITIAL_EPISODES] 学習用キャッシュは見つかりませんでした。Phase1で収集して保存します")
        else:
            print("[INITIAL_EPISODES] 学習用キャッシュは見つかりませんでした。Phase1で収集します（保存なし）")

    initial_episode_log_path = None
    if load_phase1_cache:
        initial_episode_log_path = os.environ.get("DISTRIBUTED_PCN_INITIAL_EPISODE_LOG_PATH")
        if not initial_episode_log_path:
            initial_episode_log_path = _find_initial_episode_log(output_base, execution_dir)
    if initial_episode_log_path:
        try:
            loaded_initial_episode_rows = _load_initial_episode_log(initial_episode_log_path)
            loaded_summary_path = _write_loaded_initial_episode_summary(
                execution_dir,
                initial_episode_log_path,
                loaded_initial_episode_rows,
            )
            print(
                f"[INITIAL_EPISODES] 既存の要約ログを読み込み: {initial_episode_log_path} "
                f"({len(loaded_initial_episode_rows)} episodes)"
            )
            print(f"[INITIAL_EPISODES] 要約ログの読み込みサマリーを保存: {loaded_summary_path}")
        except Exception as e:
            print(f"[INITIAL_EPISODES] 既存の要約ログの読み込みに失敗: {initial_episode_log_path}: {e}")
    elif not load_phase1_cache:
        print("[INITIAL_EPISODES] --load-phase1 未指定のため、既存ログは読み込みません")
    else:
        print("[INITIAL_EPISODES] 読み込む既存ログは見つかりませんでした")

    EVAL_DIAG = os.environ.get("DISTRIBUTED_PCN_EVAL_DIAG", "0") == "1"
    eval_diag_path = os.path.join(execution_dir, "pcn_eval_diag.jsonl") if EVAL_DIAG else None
    if EVAL_DIAG:
        print(f"[EVAL_DIAG] 各評価の統計を追記: {eval_diag_path}")
    print(
        f"[EVAL/VIS] samples={EVAL_SAMPLES} final_reuse={_SKIP_FINAL_EVAL} "
        f"archive_all={_ARCHIVE_VIS_ALL} reward_plot={_VIS_REWARD_PLOT} "
        f"command_arrows={_VIS_COMMAND_ARROWS} dpi={_VIS_PLOT_DPI}"
    )
    _main_wall_t0 = time.perf_counter()
    
    if TIME_DEBUG:
        overall_start_time = time.time()
        print(f"\n{'='*60}")
        print("分散PCN学習開始")
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"実行ディレクトリ: {execution_dir}")
        print(f"{'='*60}")
    
    # 設定ファイルの読み込み
    # 環境変数 DISTRIBUTED_PCN_CONFIG で設定ファイルパスを指定可能（Singularity等でマウント先を指定）
    config_path = os.environ.get('DISTRIBUTED_PCN_CONFIG', 'config/config.yml')
    if not os.path.isabs(config_path):
        # 相対パスの場合、プロジェクトルート基準
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(repo_root, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path} (DISTRIBUTED_PCN_CONFIG でパスを指定してください)")
    with open(config_path, 'r') as yml:
        config = yaml.safe_load(yml)
    
    # スケーリングベンチマーク用: 環境変数で上書き
    if os.environ.get('DISTRIBUTED_PCN_JOBS'):
        config['param_env']['n_jobs'] = int(os.environ['DISTRIBUTED_PCN_JOBS'])
    else:
        config['param_env']['n_jobs'] = N_JOBS
    if os.environ.get('DISTRIBUTED_PCN_ONPREM'):
        config['param_env']['n_on_premise_node'] = int(os.environ['DISTRIBUTED_PCN_ONPREM'])
    if os.environ.get('DISTRIBUTED_PCN_CLOUD'):
        config['param_env']['n_cloud_node'] = int(os.environ['DISTRIBUTED_PCN_CLOUD'])
    if _PROFILE_MODE or _QUICK_MODE:
        print(f"[SCALE] N_JOBS={config['param_env']['n_jobs']}, onprem={config['param_env']['n_on_premise_node']}, cloud={config['param_env']['n_cloud_node']}")

    # param_algorithm_compare.distributed_pcn（config.yml）で学習規模を上書きし、続けて QUICK があれば短縮
    global N_ITERATIONS, N_ACTORS, INITIAL_EPISODES, EPISODES_PER_ITERATION, EVAL_INTERVAL, SUPERVISED_LEARNING_EPOCHS, SUPERVISED_UPDATES_PER_EPOCH, LEARNING_RATE, SUPERVISED_LEARNING_RATE, N_UPDATES
    _dpc = get_param_algorithm_compare(config).get("distributed_pcn") or {}
    N_ITERATIONS = int(_dpc.get("n_iterations", N_ITERATIONS))
    N_ACTORS = int(_dpc.get("n_actors", N_ACTORS))
    INITIAL_EPISODES = int(_dpc.get("initial_episodes", INITIAL_EPISODES))
    if _dpc.get("quick") is True and "DISTRIBUTED_PCN_QUICK" not in os.environ:
        os.environ["DISTRIBUTED_PCN_QUICK"] = "1"
    elif _dpc.get("quick") is False and "DISTRIBUTED_PCN_QUICK" not in os.environ:
        os.environ["DISTRIBUTED_PCN_QUICK"] = "0"
    if _dpc.get("profile") is True and "DISTRIBUTED_PCN_PROFILE" not in os.environ:
        os.environ["DISTRIBUTED_PCN_PROFILE"] = "1"
    elif _dpc.get("profile") is False and "DISTRIBUTED_PCN_PROFILE" not in os.environ:
        os.environ["DISTRIBUTED_PCN_PROFILE"] = "0"
    if os.environ.get("DISTRIBUTED_PCN_QUICK", "0") == "1":
        N_ITERATIONS = 5
        N_ACTORS = 12
        INITIAL_EPISODES = 100
        EPISODES_PER_ITERATION = 1
        EVAL_INTERVAL = 5
        SUPERVISED_LEARNING_EPOCHS = 10

    # 環境変数で学習規模を最終上書き（ジョブ数スイープ等）
    if os.environ.get("DISTRIBUTED_PCN_N_ITERATIONS"):
        N_ITERATIONS = int(os.environ["DISTRIBUTED_PCN_N_ITERATIONS"])
    if os.environ.get("DISTRIBUTED_PCN_N_ACTORS"):
        N_ACTORS = int(os.environ["DISTRIBUTED_PCN_N_ACTORS"])
    if os.environ.get("DISTRIBUTED_PCN_INITIAL_EPISODES"):
        INITIAL_EPISODES = int(os.environ["DISTRIBUTED_PCN_INITIAL_EPISODES"])
    if os.environ.get("DISTRIBUTED_PCN_EVAL_INTERVAL"):
        EVAL_INTERVAL = int(os.environ["DISTRIBUTED_PCN_EVAL_INTERVAL"])
    if _dpc.get("left_tail") is True and os.environ.get("PCN_LEFT_TAIL_PROFILE") != "0":
        from src.distributed.distributed_pcn_cli import apply_left_tail_training_env

        apply_left_tail_training_env()
        print("[LEFT_TAIL] config.yml distributed_pcn.left_tail により学習プロファイルを適用")
    if _dpc.get("workload_adaptive") is True and os.environ.get("PCN_WORKLOAD_ADAPTIVE", "1") != "0":
        from src.distributed.workload_pcn_profile import apply_workload_adaptive_training_env

        apply_workload_adaptive_training_env(config)
        print("[WORKLOAD_ADAPT] ジョブセットからスケール推定 + 中域 PF / Evalギャップ FB を適用")
    if _dpc.get("conditioning") is True:
        from src.distributed.distributed_pcn_cli import apply_distributed_pcn_cli_env

        apply_distributed_pcn_cli_env(
            conditioning=True,
            mid_core=_dpc.get("mid_core") is True,
        )
        print("[CONFIG] distributed_pcn.conditioning (+ mid_core) プロファイルを適用")
    from src.agents.pcn_agent import refresh_train_env_weights

    refresh_train_env_weights()
    if os.environ.get("DISTRIBUTED_PCN_SUPERVISED_EPOCHS"):
        SUPERVISED_LEARNING_EPOCHS = int(os.environ["DISTRIBUTED_PCN_SUPERVISED_EPOCHS"])
    if os.environ.get("DISTRIBUTED_PCN_SUPERVISED_UPDATES_PER_EPOCH"):
        SUPERVISED_UPDATES_PER_EPOCH = int(os.environ["DISTRIBUTED_PCN_SUPERVISED_UPDATES_PER_EPOCH"])
    if os.environ.get("DISTRIBUTED_PCN_N_UPDATES"):
        # Phase3 1iterあたりの更新回数。Phase3で段階的に発見される効率点(PF点)の反復学習量を制御。
        # 大きいほど新発見効率点をその iter で多く学習(archiveに埋もれて忘れる前に定着)。
        N_UPDATES = int(os.environ["DISTRIBUTED_PCN_N_UPDATES"])
    if os.environ.get("DISTRIBUTED_PCN_LEARNING_RATE"):
        LEARNING_RATE = float(os.environ["DISTRIBUTED_PCN_LEARNING_RATE"])
    # 教師あり(Phase3)学習率の上書き。条件埋め込みの数値発散→NaN→nan_to_num=0→command無視
    # 崩壊（iter後半）を抑えるアブレーション用。既定 1e-3。
    if os.environ.get("DISTRIBUTED_PCN_SUPERVISED_LR"):
        SUPERVISED_LEARNING_RATE = float(os.environ["DISTRIBUTED_PCN_SUPERVISED_LR"])

    # Rayの初期化時にGPUリソースを明示的に指定
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if DEBUG:
        print(f"Ray初期化: GPU数={num_gpus}")
    
    # Rayの初期化（ローカルモードでGPUが利用可能な場合のみGPUリソースを指定）
    # 注意: クラスターモードで実行されている場合、num_gpusを指定するとautoscalerがGPUノードを探してしまう
    # そのため、ローカルモードで実行されている場合のみnum_gpusを指定する
    ray_init_kwargs = {
        'ignore_reinit_error': True
    }
    
    # ローカルモードで実行されている場合のみGPUリソースを指定
    # Rayが既に初期化されている場合は、クラスターモードで実行されている可能性がある
    if not ray.is_initialized() and num_gpus > 0:
        # ローカルモードで実行されている場合、GPUリソースを指定
        ray_init_kwargs['num_gpus'] = num_gpus
        if DEBUG:
            print(f"Ray初期化: ローカルモードでGPUリソースを指定 (num_gpus={num_gpus})")
    else:
        if DEBUG:
            if ray.is_initialized():
                print("Ray初期化: 既に初期化されているため、GPUリソースを指定しません（クラスターモードの可能性）")
            else:
                print("Ray初期化: GPUが利用できないため、GPUリソースを指定しません")
    
    # Rayのシリアライゼーション設定を最適化
    # object_store_memoryを増やしてシリアライゼーションのオーバーヘッドを削減
    if 'object_store_memory' not in ray_init_kwargs:
        # エピソードデータが大きいため、object_store_memoryを増やす
        # 1エピソードあたり約7MB、32Actor × 5エピソード = 約1120MB
        # バッファサイズとスピルを考慮して16GBに設定（メモリが利用可能な場合）
        # システムメモリの30%を上限とする
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # 最小8GB、最大16GB、利用可能メモリの30%のうち最小値
        suggested_memory = min(16 * 1024 * 1024 * 1024, int(available_memory_gb * 0.3 * 1024 * 1024 * 1024))
        suggested_memory = max(8 * 1024 * 1024 * 1024, suggested_memory)  # 最低8GB
        ray_init_kwargs['object_store_memory'] = suggested_memory
        if DEBUG:
            print(f"Ray object_store_memory設定: {suggested_memory / (1024**3):.1f}GB (利用可能メモリ: {available_memory_gb:.1f}GB)")
    
    # Rayのcompressionを有効化（シリアライゼーション時のデータサイズを削減）
    # 環境変数で設定（ray.init()の前に設定する必要がある）
    import os
    if 'RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE' not in os.environ:
        os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
    # スピルログを抑制（必要に応じて）
    if 'RAY_verbose_spill_logs' not in os.environ:
        os.environ['RAY_verbose_spill_logs'] = '0'
    
    ray.init(**ray_init_kwargs)
    
    # Rayのシリアライゼーション設定を最適化（ray.init()後に設定）
    # 注意: _system_configはray.init()の引数として直接渡すことはできないため、
    # 環境変数またはrayの設定ファイルで設定する必要があります
    # ここでは、object_store_memoryの増加のみを実装しています
    
    # RayがGPUリソースを認識しているかどうかを確認
    # クラスターモードで実行されている場合、GPUリソースを要求しない
    cluster_resources = ray.cluster_resources()
    has_gpu_in_cluster = 'GPU' in cluster_resources and cluster_resources['GPU'] > 0
    
    # ローカルモードでGPUが利用可能な場合のみ、GPUリソースを要求
    # クラスターモードで実行されている場合、num_gpusを指定しない
    # 注意: Rayがクラスターモードで実行されている場合、num_gpusを指定すると
    # autoscalerがGPUノードを探してしまう。そのため、RayがGPUリソースを
    # 認識している場合のみGPUリソースを要求する。
    if has_gpu_in_cluster and num_gpus > 0:
        # RayがGPUリソースを認識している場合、GPUリソースを要求
        LearnerActor = ray.remote(num_gpus=1)(Learner)
        if DEBUG:
            print(f"Learner: GPUリソースを要求 (RayがGPUを認識しています)")
            print(f"  クラスタリソース: {cluster_resources}")
    else:
        # RayがGPUリソースを認識していない場合、GPUリソースを要求しない
        # PyTorchが直接GPUを使用するため、Rayのリソース管理は不要
        LearnerActor = ray.remote(Learner)
        if DEBUG:
            if num_gpus > 0:
                print(f"Learner: GPUリソースを要求しません（RayがGPUを認識していない可能性があります）")
                print(f"  クラスタリソース: {cluster_resources}")
                print(f"  PyTorchが直接GPUを使用します（Rayのリソース管理は行いません）")
            else:
                print(f"Learner: GPUリソースを要求しません（GPUが利用できません）")

    # Replay Buffer（メモリ使用量を削減するためサイズを調整）
    # メモリスピルを防ぐため、max_sizeを5000に削減
    REPLAY_BUFFER_MAX_SIZE = 5000  # 10000から5000に削減
    buffer = ReplayBuffer.remote(max_size=REPLAY_BUFFER_MAX_SIZE)
    if DEBUG:
        print(f"ReplayBuffer初期化: max_size={REPLAY_BUFFER_MAX_SIZE}")
    initial_cache_loaded = False
    initial_cache_stats = None
    if initial_episode_cache_path and os.path.exists(initial_episode_cache_path):
        initial_cache_stats = ray.get(buffer.load_from_file.remote(initial_episode_cache_path))
        initial_cache_loaded = initial_cache_stats.get("episodes", 0) > 0
        if initial_cache_loaded:
            print(
                f"[INITIAL_EPISODES] 学習用キャッシュをReplayBufferへ復元: "
                f"{initial_cache_stats['episodes']} episodes, "
                f"{initial_cache_stats['transitions']} transitions"
            )
    elif initial_episode_cache_path:
        print(f"[INITIAL_EPISODES] 指定された学習用キャッシュが存在しません: {initial_episode_cache_path}")

    learner = LearnerActor.remote(config, buffer, device='cuda')

    actors = [Actor.remote(config, learner, buffer, actor_id=i) for i in range(N_ACTORS)]
    
    init_futures = [actor._make_env.remote() for actor in actors]


    # =========================
    # フェーズ1: 初期エピソードの収集
    # =========================
    if DEBUG or TIME_DEBUG:
        print("\n" + "="*60)
        print("フェーズ1: 初期エピソードの収集")
        print("="*60)
    
    # フェーズ1の開始時間を記録
    if TIME_DEBUG:
        phase1_start_time = time.time()
        print(f"フェーズ1開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if DEBUG:
        print(f"各Actorで{INITIAL_EPISODES}エピソードを実行します...")
    
    # 進捗表示の間隔を計算（INITIAL_EPISODESの10分の1）
    progress_interval = max(1, INITIAL_EPISODES // 10)
    if DEBUG:
        print(f"進捗表示間隔: {progress_interval}エピソードごと")

    total_episodes = 0
    completed_actors = 0
    phase1_episode_summaries = []
    if initial_cache_loaded:
        total_episodes = int(initial_cache_stats["episodes"])
        print("[INITIAL_EPISODES] Phase1のランダム収集をスキップしました（学習用キャッシュ使用）")
    else:
        initial_action_probs = None
        if _INITIAL_ACTION_SWEEP:
            if not _INITIAL_ACTION_SWEEP_PROBS:
                raise ValueError("DISTRIBUTED_PCN_INITIAL_ACTION_SWEEP_PROBS が空です")
            for p in _INITIAL_ACTION_SWEEP_PROBS:
                if p < 0.0 or p > 1.0:
                    raise ValueError(f"初期action sweep確率は[0,1]で指定してください: {p}")
            total_initial_episodes = N_ACTORS * INITIAL_EPISODES
            initial_action_probs = [
                _INITIAL_ACTION_SWEEP_PROBS[i % len(_INITIAL_ACTION_SWEEP_PROBS)]
                for i in range(total_initial_episodes)
            ]
            counts = {
                p: initial_action_probs.count(p)
                for p in _INITIAL_ACTION_SWEEP_PROBS
            }
            print(
                "初期ランダム収集にaction=1比率スイープを適用: "
                + ", ".join(f"p={p:g}: {counts[p]}eps" for p in _INITIAL_ACTION_SWEEP_PROBS)
            )

        # 各Actorで初期エピソードを実行（ランダム行動）
        # 全ての初期化を並列で待つ（for文を避けて並列化を最大化）
        try:
            ray.get(init_futures)  # 全ての初期化を並列で待つ
        except Exception as e:
            print(f"一部のActorの初期化でエラーが発生: {e}")

        # 全てのエピソード生成を並列で開始（for文を避けて並列化を最大化）
        simulation_futures = [
            actor.run.remote(
                n_episodes=INITIAL_EPISODES,
                random_actions=True,
                random_action_probs=initial_action_probs,
            )
            for actor in actors
        ]

        # 全てのエピソード生成を並列で待つ
        try:
            results = ray.get(simulation_futures)  # 全ての結果を並列で取得
            for i, result in enumerate(results):
                episodes_generated = result["episodes_generated"] if isinstance(result, dict) else int(result)
                total_episodes += episodes_generated
                if isinstance(result, dict):
                    phase1_episode_summaries.extend(result.get("episode_summaries", []))
                if isinstance(result, dict) and result.get("action_one_prob_counts"):
                    if "phase1_action_prob_counts" not in locals():
                        phase1_action_prob_counts = {}
                    for p, c in result["action_one_prob_counts"].items():
                        phase1_action_prob_counts[float(p)] = phase1_action_prob_counts.get(float(p), 0) + int(c)
                completed_actors += 1
                # 進捗を表示
                progress_percentage = (completed_actors / N_ACTORS) * 100
                if DEBUG:
                    print(f"Actor {i} の初期エピソード生成完了: {episodes_generated} エピソード (進捗: {progress_percentage:.1f}%)")
        except Exception as e:
            print(f"一部のActorのエピソード生成でエラーが発生: {e}")

        # --- Phase-1 ヒューリスティック種まき（reactive cloud-overflow で良質な低wait例を生成）---
        # ランダムBernoulli配置は「コストは合うがwaitは最適より悪い」点しか生まない。
        # WaitTimeThreshold（オンプレ予測待ち>=閾値でクラウド）で賢く配置された低wait例を追加投入する。
        _heur_th_env = os.environ.get("DISTRIBUTED_PCN_PHASE1_HEURISTIC_THRESHOLDS", "").strip()
        if _heur_th_env:
            heur_thresholds = [float(x) for x in _heur_th_env.split(",") if x.strip()]
            heur_eps = int(os.environ.get("DISTRIBUTED_PCN_PHASE1_HEURISTIC_EPISODES", str(max(1, INITIAL_EPISODES // 2))))
            print(f"[PHASE1_HEURISTIC] WaitTimeThreshold種まき: thresholds={heur_thresholds} eps/actor={heur_eps}")
            try:
                heur_futures = [
                    actor.run.remote(n_episodes=heur_eps, random_actions=True, heuristic_thresholds=heur_thresholds)
                    for actor in actors
                ]
                heur_results = ray.get(heur_futures)
                _heur_added = 0
                for result in heur_results:
                    eg = result["episodes_generated"] if isinstance(result, dict) else int(result)
                    total_episodes += eg
                    _heur_added += eg
                    if isinstance(result, dict):
                        phase1_episode_summaries.extend(result.get("episode_summaries", []))
                print(f"[PHASE1_HEURISTIC] 種まき完了: +{_heur_added} episodes（合計 {total_episodes}）")
            except Exception as e:
                print(f"[PHASE1_HEURISTIC] 種まきでエラー: {e}")

        # --- Phase-1 巨大ジョブ後回し(giant-defer)種まき（汎化する行動則: 未知ジョブに崩れない）---
        # 「占有量上位の巨大ジョブは後回し(defer)、他は WaitTimeThreshold で配置」というヒューリスティックを
        # デモとして投入。占有量順位は相対値なので任意インスタンスで有効＝NSGA種まき(s0特化)と違い汎化する。
        # defer 行動が要るので SCHEDULER_ALLOW_DEFER=1 と併用。OFF時(env未設定)は既存経路。
        _gdef_env = os.environ.get("DISTRIBUTED_PCN_PHASE1_GIANT_DEFER", "").strip()
        if _gdef_env:
            try:
                gdef_thr = [float(x) for x in _gdef_env.split(",") if x.strip()]
                gdef_eps = int(os.environ.get("DISTRIBUTED_PCN_PHASE1_GIANT_DEFER_EPISODES", str(max(1, INITIAL_EPISODES // 2))))
                gdef_wtth = [float(x) for x in os.environ.get("DISTRIBUTED_PCN_PHASE1_GIANT_DEFER_WTTH", "0,50,150,500,999999").split(",") if x.strip()]
                print(f"[PHASE1_GIANT_DEFER] 巨大ジョブ後回し種まき: 占有量順位閾値={gdef_thr} 非巨大wtth={gdef_wtth} eps/actor={gdef_eps}")
                gdef_futures = [
                    actor.run.remote(n_episodes=gdef_eps, random_actions=True,
                                     giant_defer_thresholds=gdef_thr, heuristic_thresholds=gdef_wtth)
                    for actor in actors
                ]
                gdef_results = ray.get(gdef_futures)
                _gdef_added = 0
                for result in gdef_results:
                    eg = result["episodes_generated"] if isinstance(result, dict) else int(result)
                    total_episodes += eg
                    _gdef_added += eg
                    if isinstance(result, dict):
                        phase1_episode_summaries.extend(result.get("episode_summaries", []))
                print(f"[PHASE1_GIANT_DEFER] 種まき完了: +{_gdef_added} episodes（合計 {total_episodes}）")
            except Exception as e:
                print(f"[PHASE1_GIANT_DEFER] 種まきでエラー: {e}")

        # --- Phase-1 NSGA-II 種まき（PCN_SEED_CHROMOSOMES=npz: 探索器が見つけたPF遺伝子をエピソード化して投入）---
        # ランダム/wtth種まきでは膝(効率域)のエピソードが宝くじでしか入らず、Phase2出発点の質が
        # run間分散の根になる。NSGA-IIのPF遺伝子(訓練インスタンス上の探索結果=eval非接触)を再生して
        # 膝域を確実にarchiveへ入れる。OFF時(env未設定)は完全に既存経路。
        _nsga_npz = os.environ.get("PCN_SEED_CHROMOSOMES", "").strip()
        if _nsga_npz:
            try:
                if not os.path.exists(_nsga_npz):
                    raise FileNotFoundError(_nsga_npz)
                _nd = np.load(_nsga_npz, allow_pickle=True)
                _ch = np.array(_nd["chromosomes"], dtype=np.int8)
                _npf = np.array(_nd["pf"], dtype=np.float64)
                # 重複遺伝子を除去し cost 昇順に → K 本へ等間引き（両端は必ず残す）
                _, _uidx = np.unique(_ch, axis=0, return_index=True)
                _uidx = np.sort(_uidx)
                _ch, _npf = _ch[_uidx], _npf[_uidx]
                _ord = np.argsort(_npf[:, 0])
                _ch, _npf = _ch[_ord], _npf[_ord]
                _k = int(os.environ.get("PCN_SEED_CHROMO_K", "40"))
                if len(_ch) > _k:
                    _sel = np.unique(np.linspace(0, len(_ch) - 1, _k).round().astype(int))
                    _ch, _npf = _ch[_sel], _npf[_sel]
                _seqs = [c.tolist() for c in _ch]
                # ε摂動種まき(アンカー残差方策の前哨戦): 各アンカー遺伝子のεビット反転コピーを追加し、
                # アンカー「近傍」の多様性を注入する(正解の丸暗記でなく周辺地形を教える=過剰模倣の緩和)。
                _pert = float(os.environ.get("PCN_SEED_PERTURB", "0"))
                _pcopies = int(os.environ.get("PCN_SEED_PERTURB_COPIES", "2"))
                if _pert > 0:
                    _prng = np.random.default_rng(20260613)
                    for _c in list(_seqs):
                        _arr = np.array(_c, dtype=np.int8)
                        for _ in range(_pcopies):
                            _mask = _prng.random(_arr.shape[0]) < _pert
                            _v = _arr.copy()
                            _v[_mask] = 1 - _v[_mask]
                            _seqs.append(_v.tolist())
                    print(f"[PHASE1_NSGA] ε摂動: p={_pert} copies={_pcopies} → 計{len(_seqs)}系列")
                _eps_per_actor = max(1, (len(_seqs) + N_ACTORS - 1) // N_ACTORS)
                print(f"[PHASE1_NSGA] NSGA種まき: {_nsga_npz} → {len(_seqs)}遺伝子 "
                      f"(cost {_npf[:, 0].min():.0f}–{_npf[:, 0].max():.0f}) eps/actor={_eps_per_actor}")
                nsga_futures = [
                    actor.run.remote(n_episodes=_eps_per_actor, random_actions=True, fixed_action_seqs=_seqs)
                    for actor in actors
                ]
                nsga_results = ray.get(nsga_futures)
                _nsga_added = 0
                for result in nsga_results:
                    eg = result["episodes_generated"] if isinstance(result, dict) else int(result)
                    total_episodes += eg
                    _nsga_added += eg
                    if isinstance(result, dict):
                        phase1_episode_summaries.extend(result.get("episode_summaries", []))
                print(f"[PHASE1_NSGA] 種まき完了: +{_nsga_added} episodes（合計 {total_episodes}）")
            except Exception as e:
                print(f"[PHASE1_NSGA] 種まきでエラー: {e}")

    if DEBUG:
        print(f"合計生成エピソード数: {total_episodes}")
        print("=== 初期経験収集完了 ===")
    if _INITIAL_ACTION_SWEEP and "phase1_action_prob_counts" in locals():
        print(
            "初期action=1比率スイープ実績: "
            + ", ".join(
                f"p={p:g}: {phase1_action_prob_counts[p]}eps"
                for p in sorted(phase1_action_prob_counts)
            )
        )
    initial_episode_log_limit = int(os.environ.get("DISTRIBUTED_PCN_INITIAL_EPISODE_LOG_LIMIT", "100"))
    if initial_episode_log_limit > 0 and phase1_episode_summaries:
        phase1_episode_summaries.sort(key=lambda x: (x["actor_id"], x["actor_episode_index"]))
        logged_summaries = phase1_episode_summaries[:initial_episode_log_limit]
        initial_log_path = os.path.join(execution_dir, "initial_episodes_first100.jsonl")
        initial_summary_path = os.path.join(execution_dir, "initial_episodes_first100_summary.json")
        with open(initial_log_path, "w", encoding="utf-8") as f:
            for row in logged_summaries:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        lengths = [row["episode_length"] for row in logged_summaries]
        action0 = sum(row["action_counts"].get("0", 0) for row in logged_summaries)
        action1 = sum(row["action_counts"].get("1", 0) for row in logged_summaries)
        with open(initial_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "logged_episodes": len(logged_summaries),
                    "total_phase1_episodes": len(phase1_episode_summaries),
                    "episode_log": initial_log_path,
                    "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
                    "episode_length_min": int(np.min(lengths)) if lengths else 0,
                    "episode_length_max": int(np.max(lengths)) if lengths else 0,
                    "action_counts": {"0": int(action0), "1": int(action1)},
                },
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        print(f"[INITIAL_EPISODES] 先頭{len(logged_summaries)}エピソードの要約を保存: {initial_log_path}")
    if save_phase1_cache and not initial_cache_loaded and total_episodes > 0:
        current_initial_cache_path = os.path.join(execution_dir, "initial_episodes_cache.pkl.gz")
        cache_stats = ray.get(
            buffer.save_to_file.remote(
                current_initial_cache_path,
                metadata={
                    "n_actors": int(N_ACTORS),
                    "initial_episodes_per_actor": int(INITIAL_EPISODES),
                    "total_episodes": int(total_episodes),
                    "use_event_obs": bool(_USE_EVENT_OBS),
                    "event_to_bitmap": bool(_USE_EVENT_OBS and learner_bitmap_enabled()),
                    "n_jobs": int(config["param_env"].get("n_jobs", N_JOBS)),
                },
            )
        )
        print(
            f"[INITIAL_EPISODES] 学習用キャッシュを保存: {cache_stats['path']} "
            f"({cache_stats['episodes']} episodes, {cache_stats['transitions']} transitions)"
        )
    elif not save_phase1_cache and not initial_cache_loaded and total_episodes > 0:
        print("[INITIAL_EPISODES] --save-phase1 未指定のため、学習用キャッシュは保存しません")

    if os.environ.get("DISTRIBUTED_PCN_STOP_AFTER_PHASE1", "0") == "1":
        print("[INITIAL_EPISODES] DISTRIBUTED_PCN_STOP_AFTER_PHASE1=1 のため Phase1 後に終了します")
        ray.shutdown()
        return
    
    # フェーズ1の完了時間を記録
    if TIME_DEBUG:
        phase1_end_time = time.time()
        phase1_duration = phase1_end_time - phase1_start_time
        print(f"\n{'='*40}")
        print(f"フェーズ1完了: 初期エピソード収集")
        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {phase1_duration:.2f}秒 ({phase1_duration/60:.2f}分)")
        print(f"生成エピソード数: {total_episodes}")
        print(f"{'='*40}")


    if DEBUG:
        print("初期エピソードをLearnerの経験再生バッファに追加中...")

    # まずLearnerの学習を実行（これによりReplayBufferからエピソードが取得され、Learnerの経験再生バッファに追加される）
    initial_loss = ray.get(learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES))
    print(f"初期学習の損失: {initial_loss}")

    if DEBUG:
        print(f"初期学習の損失: {initial_loss}")
        print("=== 初期学習完了 ===")
        
        # バッファの統計情報を表示
        buffer_stats = ray.get(buffer.get_stats.remote())
        print(f"\n=== ReplayBuffer統計 ===")
        print(f"バッファサイズ: {buffer_stats['buffer_size']}")
        print(f"ユニークエピソード数: {buffer_stats['unique_episodes']}")
        print(f"最大サイズ: {buffer_stats['max_size']}")
        print(f"利用率: {buffer_stats['utilization']:.2%}")
        print("=" * 30)

    initial_axis_ranges = None

    # =========================
    # フェーズ1終了時の学習データ分析（DEBUG時のみ：get_experience_replayは重い）
    # =========================
    if DEBUG:
        print("\n" + "="*60)
        print("フェーズ1終了: 学習データの分析と保存")
        print("="*60)
        # try:
        #     experience_replay = ray.get(learner.get_experience_replay.remote())
        #     if len(experience_replay) > 0:
        #         print(f"✓ 学習データの分析を開始します...")
        #         analysis_file = ray.get(learner.save_learning_data_to_file.remote(
        #             filename=f"phase1_learning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        #             sample_size=500
        #         ))
        #         print(f"✓ フェーズ1学習データの分析完了! 詳細: {analysis_file}")
        #         total_transitions = sum(len(episode[2]) for episode in experience_replay)
        #         print(f"総エピソード数: {len(experience_replay)}, 総遷移数: {total_transitions}")
        #         all_actions = []
        #         for episode in experience_replay:
        #             for transition in episode[2]:
        #                 all_actions.append(transition.action)
        #         unique_actions, action_counts = np.unique(all_actions, return_counts=True)
        #         print(f"行動分布: {dict(zip(unique_actions, action_counts))}")
        #     else:
        #         print("⚠️  学習データが空です。")
        # except Exception as e:
        #     print(f"❌ 学習データ分析中にエラー: {e}")
        #     import traceback
        #     traceback.print_exc()

    # =========================
    # フェーズ2: 教師あり学習（初期エピソードを使用）
    # =========================
    if DEBUG or TIME_DEBUG:
        print("\n" + "="*60)
        print("フェーズ2: 教師あり学習（初期エピソードを使用）")
        print("="*60)
    
    # フェーズ2の開始時間を記録
    if TIME_DEBUG:
        phase2_start_time = time.time()
        print(f"フェーズ2開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    if DEBUG:
        print("初期エピソードを使用して教師あり学習を開始します...")
    
    # 初期エピソードの統計を表示
    initial_buffer_size = ray.get(learner._get_buffer_size.remote())
    if DEBUG:
        print(f"教師あり学習開始時のバッファサイズ: {initial_buffer_size}")
    
    # 初期データの質を詳細に分析
    if DEBUG:
        print("\n=== 初期データの質分析 ===")
        experience_replay = ray.get(learner.get_experience_replay.remote())
        
        # 全エピソードの統計
        total_transitions = 0
        action_distribution = {}
        reward_stats = []
        episode_lengths = []
        episode_returns = []
        
        for priority, unique_step, transitions in experience_replay:
            if len(transitions) > 0:
                total_transitions += len(transitions)
                episode_lengths.append(len(transitions))
                
                # 行動の分布を集計
                for t in transitions:
                    action = t.action
                    action_distribution[action] = action_distribution.get(action, 0) + 1
                
                # 報酬の統計を集計
                episode_rewards = [t.reward for t in transitions]
                reward_stats.extend(episode_rewards)
                
                # エピソードの累積報酬を計算
                episode_return = transitions[0].reward  # 累積報酬
                episode_returns.append(episode_return)
        
        print(f"総遷移数: {total_transitions}")
        print(f"エピソード数: {len(episode_lengths)}")
        print(f"エピソード長の統計:")
        print(f"  平均: {np.mean(episode_lengths):.1f}")
        print(f"  標準偏差: {np.std(episode_lengths):.1f}")
        print(f"  最小: {np.min(episode_lengths)}")
        print(f"  最大: {np.max(episode_lengths)}")
        
        print(f"行動分布: {action_distribution}")
        if len(action_distribution) > 0:
            total_actions = sum(action_distribution.values())
            for action, count in action_distribution.items():
                percentage = (count / total_actions) * 100
                print(f"  行動{action}: {count}回 ({percentage:.1f}%)")
        
        if reward_stats:
            reward_array = np.array(reward_stats)
            print(f"報酬の統計:")
            print(f"  平均: {np.mean(reward_array, axis=0)}")
            print(f"  標準偏差: {np.std(reward_array, axis=0)}")
            print(f"  最小値: {np.min(reward_array, axis=0)}")
            print(f"  最大値: {np.max(reward_array, axis=0)}")
        
        if episode_returns:
            returns_array = np.array(episode_returns)
            print(f"エピソード累積報酬の統計:")
            print(f"  平均: {np.mean(returns_array, axis=0)}")
            print(f"  標準偏差: {np.std(returns_array, axis=0)}")
            print(f"  最小値: {np.min(returns_array, axis=0)}")
            print(f"  最大値: {np.max(returns_array, axis=0)}")
            
            # 報酬の多様性をチェック
            reward_variance = np.var(returns_array, axis=0)
            print(f"  報酬の分散: {reward_variance}")
            if np.any(reward_variance < 0.01):
                print("⚠️  警告: 報酬の分散が小さすぎます。データの多様性が不足している可能性があります。")
        
        # 行動の多様性をチェック
        if len(action_distribution) < 2:
            print("⚠️  警告: 行動の多様性が不足しています。ランダム行動の質を確認してください。")
        else:
            print(f"✓ 行動の多様性: {len(action_distribution)}種類の行動が確認されました")
            
            # 行動の偏りをチェック
            action_balance = min(action_distribution.values()) / max(action_distribution.values())
            if action_balance < 0.3:
                print(f"⚠️  警告: 行動の偏りが大きすぎます (バランス: {action_balance:.3f})")
            else:
                print(f"✓ 行動のバランス: {action_balance:.3f}")
        
        print("=" * 50)
    
    # 教師あり学習用の最適化器を一時的に調整
    
    if DEBUG:
        print(f"教師あり学習パラメータ:")
        print(f"  学習率: {SUPERVISED_LEARNING_RATE}")
        print(f"  バッチサイズ: {SUPERVISED_BATCH_SIZE}")
        print(f"  エポック数: {SUPERVISED_LEARNING_EPOCHS}")
        print(f"  エポックあたりの更新回数: {SUPERVISED_UPDATES_PER_EPOCH}")
    
            # 学習データの質を根本的に改善するための分析
        if DEBUG:
            print(f"\n=== 学習データの根本的分析 ===")
            experience_replay = ray.get(learner.get_experience_replay.remote())
            
            # エピソードの質を評価
            high_quality_episodes = 0
            low_quality_episodes = 0
            episode_quality_scores = []
            
            for priority, unique_step, transitions in experience_replay:
                if len(transitions) > 0:
                    # エピソードの質を評価（行動の多様性、報酬の多様性など）
                    actions = [t.action for t in transitions]
                    unique_actions = len(set(actions))
                    action_balance = min(actions.count(0), actions.count(1)) / max(actions.count(0), actions.count(1)) if len(set(actions)) > 1 else 0
                    
                    # 報酬の多様性
                    rewards = [t.reward for t in transitions]
                    reward_variance = np.var(rewards, axis=0)
                
                # 質のスコアを計算
                quality_score = 0
                if unique_actions >= 2:
                    quality_score += 0.3
                if action_balance > 0.3:
                    quality_score += 0.3
                if np.any(reward_variance > 0.01):
                    quality_score += 0.4
                
                episode_quality_scores.append(quality_score)
                
                # 質の判定
                if quality_score >= 0.7:  # 70%以上のスコアを高品質とする
                    high_quality_episodes += 1
                else:
                    low_quality_episodes += 1
            
            total_episodes = high_quality_episodes + low_quality_episodes
            if total_episodes > 0:
                quality_ratio = high_quality_episodes / total_episodes
                avg_quality_score = np.mean(episode_quality_scores)
                print(f"高品質エピソード: {high_quality_episodes}/{total_episodes} ({quality_ratio:.1%})")
                print(f"平均品質スコア: {avg_quality_score:.3f}")
                
                if quality_ratio < 0.5:
                    print("⚠️  警告: 高品質なエピソードが不足しています。")
                    print("    → より多様な初期エピソードの生成が必要です。")
                elif avg_quality_score < 0.6:
                    print("⚠️  警告: エピソードの平均品質が低すぎます。")
                    print("    → より多様な初期エピソードの生成が必要です。")
                else:
                    print("✓ エピソードの質は良好です。")
            
            print("=" * 50)
    
    # 学習履歴を記録
    supervised_training_history = {
        'epochs': [],
        'losses': [],
        'best_loss': float('inf'),
        'improvement_count': 0
    }
    
    phase2_train_result = ray.get(
        learner.supervised_train_epochs.remote(
            n_epochs=SUPERVISED_LEARNING_EPOCHS,
            updates_per_epoch=SUPERVISED_UPDATES_PER_EPOCH,
            learning_rate=SUPERVISED_LEARNING_RATE,
        )
    )
    _p2tag = f"Phase2 完了後 mode={'event_obs' if _USE_EVENT_OBS else 'bitmap_c'}"
    ray.get(learner.log_gpu_memory_snapshot.remote(_p2tag))
    _p2_frozen_retries = int(phase2_train_result.get("frozen_retries", 0) or 0)
    if _p2_frozen_retries > 0:
        print(f"⚠️ [FROZEN_RETRY] 凍結初期値を{_p2_frozen_retries}回検知→再initで回復済み (Phase2)")
    for epoch, epoch_result in enumerate(phase2_train_result.get("epochs", [])):
        if DEBUG:
            print(f"\n--- 教師あり学習エポック {epoch + 1}/{SUPERVISED_LEARNING_EPOCHS} ---")
        avg_epoch_loss = float(epoch_result.get("avg_loss", 0.0))
        cached_steps = int(epoch_result.get("cached_steps", 0) or 0)
        if cached_steps > 0:
            cache_place = "GPU" if epoch_result.get("cache_on_device", False) else "CPU"
            cache_mb = float(epoch_result.get("cache_mb", 0.0))
            print(
                f"Phase2教師データcacheを構築: {cached_steps} transitions "
                f"({cache_place}, {cache_mb:.1f} MB, obs_dim={epoch_result.get('obs_dim', 0)}, "
                f"PF重み対象={epoch_result.get('pf_episode_count', 0)}, "
                f"端点重み対象={epoch_result.get('endpoint_episode_count', 0)}, "
                f"Cost端重み対象={epoch_result.get('cost_endpoint_episode_count', 0)}, "
                f"MidPF重み対象={epoch_result.get('mid_pf_episode_count', 0)}, "
                f"LowWaitPF重み対象={epoch_result.get('low_wait_pf_episode_count', 0)}, "
                f"Cost端action0率={epoch_result.get('cost_endpoint_action0_rate', float('nan')):.3f}, "
                f"直近Achieved重み対象={epoch_result.get('recent_episode_count', 0)})"
            )
        if DEBUG:
            for update_log in epoch_result.get("updates", []):
                update = update_log["update"]
                if (update - 1) % 2 != 0:
                    continue
                metrics = update_log.get("metrics", {})
                loss_value = update_log["loss"]
                if isinstance(metrics, dict) and "policy_acc" in metrics and "true_prob_mean" in metrics:
                    print(
                        f"  更新 {update}/{SUPERVISED_UPDATES_PER_EPOCH}: 損失={loss_value:.4f}, "
                        f"acc={metrics['policy_acc']:.4f}, p_true={metrics['true_prob_mean']:.4f}"
                    )
                else:
                    print(f"  更新 {update}/{SUPERVISED_UPDATES_PER_EPOCH}: 損失 = {loss_value:.4f}")
        
        # 学習履歴を記録
        supervised_training_history['epochs'].append(epoch + 1)
        supervised_training_history['losses'].append(avg_epoch_loss)
        if avg_epoch_loss < supervised_training_history['best_loss']:
            supervised_training_history['best_loss'] = avg_epoch_loss
            supervised_training_history['improvement_count'] += 1
        
        print(f"エポック {epoch + 1} 完了: 平均損失 = {avg_epoch_loss:.4f}")
        
        
    if DEBUG:
        print("フェーズ2完了: 教師あり学習が完了しました")
        
        # 教師あり学習の結果を要約
        print(f"\n=== 教師あり学習結果要約 ===")
        print(f"実行エポック数: {len(supervised_training_history['epochs'])}")
        print(f"最良損失: {supervised_training_history['best_loss']:.4f}")
        print(f"改善回数: {supervised_training_history['improvement_count']}")
        
        if len(supervised_training_history['losses']) > 1:
            initial_loss = supervised_training_history['losses'][0]
            final_loss = supervised_training_history['losses'][-1]
            improvement = initial_loss - final_loss
            print(f"初期損失: {initial_loss:.4f}")
            print(f"最終損失: {final_loss:.4f}")
            print(f"改善量: {improvement:.4f}")
            
            if improvement > 0.01:
                print("✓ 教師あり学習で有意な改善が確認されました")
            elif improvement > 0:
                print("△ 教師あり学習でわずかな改善が確認されました")
            else:
                print("⚠️  教師あり学習で改善が見られませんでした")
        
        print("=" * 50)
    
    # フェーズ2の完了時間を記録
    if TIME_DEBUG:
        phase2_end_time = time.time()
        phase2_duration = phase2_end_time - phase2_start_time
        print(f"\n{'='*40}")
        print(f"フェーズ2完了: 教師あり学習")
        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {phase2_duration:.2f}秒 ({phase2_duration/60:.2f}分)")
        print(f"学習エポック数: {SUPERVISED_LEARNING_EPOCHS}")
        # if initial_e_returns is not None:
        #     print(f"n_points_first: {len(initial_e_returns)}", f"n_points: {len(initial_e_values)}")
        print(f"{'='*40}")

    if _PHASE2_IMPORTANCE:
        importance_dir = f"{execution_dir}/phase2_feature_importance"
        try:
            importance_result = ray.get(
                learner.export_phase2_feature_importance.remote(
                    importance_dir,
                    max_samples=_PHASE2_IMPORTANCE_SAMPLES,
                )
            )
            if importance_result.get("enabled", False):
                print(
                    "Phase2入力重要度出力完了: "
                    f"{importance_result.get('png_path')} / {importance_result.get('csv_path')}"
                )
            else:
                print(f"Phase2入力重要度出力をスキップ: {importance_result.get('reason')}")
        except Exception as e:
            print(f"Phase2入力重要度出力でエラー: {e}")
            import traceback
            traceback.print_exc()

    # Phase2 終了時点の cache を活かし、Phase3 では新規エピソード分のみ追記する
    if _PHASE3_GPU_CACHE:
        try:
            ray.get(learner.warmup_training_batch_cache.remote())
        except Exception as e:
            print(f"Phase3 cacheウォームアップでエラー（続行）: {e}")

    _phase3_init_ckpt = os.environ.get("DISTRIBUTED_PCN_INIT_CHECKPOINT_PHASE3", "").strip()
    if _phase3_init_ckpt and os.path.isfile(_phase3_init_ckpt):
        print(f"[INIT] Phase3 直前にチェックポイントをロード: {_phase3_init_ckpt}")
        ray.get(learner.load_checkpoint.remote(_phase3_init_ckpt))

    # Phase3突入直前: EMA shadow を現在(Phase2学習済み or ロード)重みで初期化(EMA無効時 no-op)
    ray.get(learner.reinit_ema.remote())
    ray.get(learner.start_lr_schedule.remote(N_ITERATIONS))  # lr減衰カウンタ初期化(LR_DECAY OFF時 no-op)

    # =========================
    # フェーズ3: 改良された経験の実現
    # =========================
    if DEBUG or TIME_DEBUG:
        print("\n" + "="*60)
        print("フェーズ3: 改良された経験の実現")
        print("="*60)
    
    # フェーズ3の開始時間を記録
    if TIME_DEBUG or _PROFILE_MODE:
        phase3_start_time = time.time()
        if TIME_DEBUG:
            print(f"フェーズ3開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 学習履歴を記録
    training_history = {
        'iterations': [],
        'losses': [],
        'pareto_front_sizes': [],
        'distances': [],  # Distanceを記録
        'initial_axis_ranges': initial_axis_ranges,  # 初期パレートフロントの軸範囲を保存
        'last_eval': None,  # 直近 evaluate 結果（最終可視化の再実行回避用）
    }

    # early-stop: 続学習が効率方策を壊す(検証済: 達成HVは中盤ピーク→劣化, docs/repro_512.html §8)ため、
    # 最終iterでなく「達成front HV(固定参照点)が最良」の学習中ckptを best_model.pth に保存する。
    # eval側で CKPT=<execution_dir>/best_model.pth を指定すれば early-stop されたモデルを使える。
    _EARLYSTOP = os.environ.get("DISTRIBUTED_PCN_EARLYSTOP", "0") == "1"
    _es_best_iter = -1
    _es_ref = None          # 環境変数で絶対nadirを与えた場合のみ固定。未指定なら全候補から動的に拡大。
    _es_candidates = []     # [(iter, nd_points)]: 全評価候補を保持し毎回「全候補nadir」で再評価する
    _es_best_path = os.path.join(execution_dir, "best_model.pth")
    if _EARLYSTOP:
        _ref_env = os.environ.get("DISTRIBUTED_PCN_EARLYSTOP_REF", "").strip()
        if _ref_env:
            _es_ref = np.array([float(x) for x in _ref_env.split(",")], dtype=np.float64)
        print(f"[EARLYSTOP] 有効: 達成HV最良ckptを {_es_best_path} に保存 (固定ref={_es_ref})")


    
    # 学習ループ（改良された経験の生成）
    # 非同期オーバーラップ: Learner(i)とActor(i+1)を並列実行して待ち時間を隠蔽
    if _ASYNC_OVERLAP and _PROFILE_MODE:
        print("[PROFILE] Actor-Learner非同期オーバーラップ有効")
    
    learner_future = None
    next_actor_futures = None
    n_commands_per_iter = N_ACTORS * EPISODES_PER_ITERATION
    n_jobs_for_value_scale = max(1, int(config['param_env'].get('n_jobs', N_JOBS)))
    def _normalize_commands_for_actor_and_log(commands_batch, iter_index):
        """commands_batch のメタ情報をログ出力し、Actor渡し用 (desired_return, horizon) に正規化する。"""
        actor_cmds = []
        deltas = []
        for c in commands_batch:
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                desired_return, desired_horizon, base_return = c[0], c[1], c[2]
                # values 軸: [cost, avg_waiting_time]
                base_values = np.array([-base_return[1], -base_return[0] / n_jobs_for_value_scale], dtype=np.float32)
                target_values = np.array([-desired_return[1], -desired_return[0] / n_jobs_for_value_scale], dtype=np.float32)
                delta = target_values - base_values
                deltas.append(delta)
                # Actor側の outcome 記録で base_return を使うため3要素のまま渡す
                actor_cmds.append((desired_return, desired_horizon, base_return))
            else:
                actor_cmds.append(c)
        if deltas:
            delta_arr = np.array(deltas, dtype=np.float32)
            mean_dx, mean_dy = float(np.mean(delta_arr[:, 0])), float(np.mean(delta_arr[:, 1]))
            min_dx, max_dx = float(np.min(delta_arr[:, 0])), float(np.max(delta_arr[:, 0]))
            min_dy, max_dy = float(np.min(delta_arr[:, 1])), float(np.max(delta_arr[:, 1]))
            mean_norm = float(np.mean(np.linalg.norm(delta_arr, axis=1)))
            print(
                f"[CMD Iter {iter_index}] target shift in VALUE space "
                f"(x=Cost, y=AvgWait): mean=({mean_dx:.3f}, {mean_dy:.3f}), "
                f"range_x=[{min_dx:.3f}, {max_dx:.3f}], range_y=[{min_dy:.3f}, {max_dy:.3f}], "
                f"mean_norm={mean_norm:.3f}, n={len(deltas)}"
            )
        return actor_cmds
    def _episodes_generated_sum(actor_results):
        total = 0
        for r in actor_results:
            if isinstance(r, dict):
                total += int(r.get("episodes_generated", 0))
            else:
                total += int(r)
        return total
    def _collect_command_outcomes(actor_results):
        outcomes = []
        for r in actor_results:
            if isinstance(r, dict):
                outcomes.extend(r.get("command_outcomes", []))
        return outcomes
    def _log_command_follow_stats(iter_index, outcomes):
        """command追従の方向性を reward/value 空間で診断表示する。"""
        if not outcomes:
            print(f"[CMD Iter {iter_index}] follow stats: no outcomes")
            return
        try:
            base_r = np.array([o.get("base_return", o["command_return"]) for o in outcomes], dtype=np.float32)
            target_r = np.array([o["command_return"] for o in outcomes], dtype=np.float32)
            achieved_r = np.array([o["achieved_return"] for o in outcomes], dtype=np.float32)
        except Exception as e:
            print(f"[CMD Iter {iter_index}] follow stats parse error: {e}")
            return

        tgt_base_r = target_r - base_r
        ach_base_r = achieved_r - base_r
        sign_match_r = np.sign(tgt_base_r) == np.sign(ach_base_r)
        # 0 を不一致扱いしない（目標差分0は一致扱い）
        zero_mask_r = np.isclose(tgt_base_r, 0.0)
        sign_match_r = np.logical_or(sign_match_r, zero_mask_r)
        sign_match_rate_r = np.mean(sign_match_r, axis=0)
        # 目標差分が非ゼロの軸のみで一致率を計算（実質的な追従度）
        nz_mask_r = ~zero_mask_r
        def _nz_match(sign_match, nz_mask, axis):
            denom = np.sum(nz_mask[:, axis])
            if denom == 0:
                return np.nan
            return float(np.sum(sign_match[:, axis] & nz_mask[:, axis]) / denom)
        nz_match_r1 = _nz_match(sign_match_r, nz_mask_r, 0)
        nz_match_r2 = _nz_match(sign_match_r, nz_mask_r, 1)

        # value空間へ変換（x=cost, y=avg_wait）
        base_v = np.stack([-base_r[:, 1], -base_r[:, 0] / n_jobs_for_value_scale], axis=1)
        target_v = np.stack([-target_r[:, 1], -target_r[:, 0] / n_jobs_for_value_scale], axis=1)
        achieved_v = np.array([o["achieved_values"] for o in outcomes], dtype=np.float32)
        tgt_base_v = target_v - base_v
        ach_base_v = achieved_v - base_v
        sign_match_v = np.sign(tgt_base_v) == np.sign(ach_base_v)
        zero_mask_v = np.isclose(tgt_base_v, 0.0)
        sign_match_v = np.logical_or(sign_match_v, zero_mask_v)
        sign_match_rate_v = np.mean(sign_match_v, axis=0)
        nz_mask_v = ~zero_mask_v
        nz_match_x = _nz_match(sign_match_v, nz_mask_v, 0)
        nz_match_y = _nz_match(sign_match_v, nz_mask_v, 1)

        print(
            f"[CMD Iter {iter_index}] follow(REWARD): "
            f"match_r1={sign_match_rate_r[0]:.3f}, match_r2={sign_match_rate_r[1]:.3f}, "
            f"nz_match_r1={nz_match_r1:.3f}, nz_match_r2={nz_match_r2:.3f}, "
            f"mean_target_norm={np.mean(np.linalg.norm(tgt_base_r, axis=1)):.2f}, "
            f"mean_achieved_norm={np.mean(np.linalg.norm(ach_base_r, axis=1)):.2f}, n={len(outcomes)}"
        )
        print(
            f"[CMD Iter {iter_index}] follow(VALUE x=Cost,y=AvgWait): "
            f"match_x={sign_match_rate_v[0]:.3f}, match_y={sign_match_rate_v[1]:.3f}, "
            f"nz_match_x={nz_match_x:.3f}, nz_match_y={nz_match_y:.3f}, "
            f"mean_target_norm={np.mean(np.linalg.norm(tgt_base_v, axis=1)):.2f}, "
            f"mean_achieved_norm={np.mean(np.linalg.norm(ach_base_v, axis=1)):.2f}"
        )
        # [CMD-TRACK] 指令↔達成の距離を「固定スケール正規化MSE」で計装する（学習は一切不変・診断のみ）。
        # 目的: 「指令追従が学習で良くなっているか」を iteration 横断で比較する。固定スケール(value-head
        # 既定の cost=1e5/wait=500)で割るのは、iteration ごとにスケールが揺れると距離を比較できないため。
        # これはユーザー要望「指定点をどれだけ再現できているかの距離をlossにして学習で小さくする」の
        # 第一歩=可視化基盤。実際にこの距離を下げる補助lossは別途 gate 付きで追加する。
        _cs = max(float(os.environ.get("PCN_VALUE_COST_SCALE", "100000.0")), 1.0)
        _ws = max(float(os.environ.get("PCN_VALUE_WAIT_SCALE", "500.0")), 1.0)
        _dc = (achieved_v[:, 0] - target_v[:, 0]) / _cs
        _dw = (achieved_v[:, 1] - target_v[:, 1]) / _ws
        _mse_cost = float(np.mean(_dc ** 2))
        _mse_wait = float(np.mean(_dw ** 2))
        # 片側（達成が指令に「届かない」側のみ）も併記。reward空間で大きいほど良い＝value空間では
        # cost/wait が指令を上回る(achieved>target)と「届いていない」。案1の片側ヒンジに対応する診断量。
        _miss_cost = float(np.mean(np.maximum(achieved_v[:, 0] - target_v[:, 0], 0.0) ** 2 / _cs ** 2))
        _miss_wait = float(np.mean(np.maximum(achieved_v[:, 1] - target_v[:, 1], 0.0) ** 2 / _ws ** 2))
        print(
            f"[CMD-TRACK Iter {iter_index}] dist(normalized MSE): "
            f"cost={_mse_cost:.4f} wait={_mse_wait:.4f} total={_mse_cost + _mse_wait:.4f} | "
            f"miss(片側) cost={_miss_cost:.4f} wait={_miss_wait:.4f}"
        )
        return {
            "iter": int(iter_index), "mse_cost": _mse_cost, "mse_wait": _mse_wait,
            "mse_total": _mse_cost + _mse_wait, "miss_cost": _miss_cost, "miss_wait": _miss_wait,
            "n": int(len(outcomes)),
        }

    _cmd_track_hist = []  # [CMD-TRACK] 指令追従距離の iteration 履歴（診断・プロット用）
    _prev_skip_stats = (0, 0)  # [STEP_SKIP] 差分計算用 (skip累計, step成功累計)
    for iteration in range(N_ITERATIONS):
        if _ASYNC_OVERLAP and N_ITERATIONS > 1:
            # 非同期オーバーラップモード
            if iteration == 0:
                if DEBUG:
                    print("Actorが改良されたエピソードを生成中...")
                    print("※ PCNエージェントの_choose_commandsと_nlargestメソッドにより改善された目標値を使用")
                # 一括で探索方向を取得（12回のリモート呼び出し→1回に削減）
                commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                commands_batch = _normalize_commands_for_actor_and_log(commands_batch, iteration + 1)
                actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                t_actor_start = time.time()
                actor_results = ray.get(actor_futures)
                t_actor = time.time() - t_actor_start
                if _PROFILE_MODE:
                    print(f"[PROFILE Iter {iteration+1}] Actor実行: {t_actor:.3f}s (合計{_episodes_generated_sum(actor_results)}エピソード)")
                
                if DEBUG:
                    print("Learnerが改良された経験で学習を実行中")
                t_learner_start = time.time()
                commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                commands_batch = _normalize_commands_for_actor_and_log(commands_batch, iteration + 1)
                learner_future = learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES, use_training_cache=True)
                next_actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                t_wait_start = time.time()
                loss = ray.get(learner_future)
                ray.get(next_actor_futures)  # Actor(1)完了待機（actor_resultsはActor(0)のまま）
                t_wait = time.time() - t_wait_start
                t_learner = time.time() - t_learner_start
                if _PROFILE_MODE:
                    print(f"[PROFILE Iter {iteration+1}] Learner+Actor(次)並列待機: {t_wait:.3f}s (Learner: {t_learner:.3f}s)")
                if N_ITERATIONS > 1:
                    learner_future = learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES, use_training_cache=True)
                    commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                    next_actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                else:
                    learner_future = None
                    next_actor_futures = None
            else:
                t_wait_start = time.time()
                # 前イテレーションで起動した Learner の結果を取得（戻り値を捨てると表示損失が固定される）
                loss_pending = ray.get(learner_future)
                actor_results = ray.get(next_actor_futures)  # Actor(i)完了
                t_wait = time.time() - t_wait_start
                if _PROFILE_MODE:
                    print(f"[PROFILE Iter {iteration+1}] Learner+Actor並列待機: {t_wait:.3f}s (合計{_episodes_generated_sum(actor_results)}エピソード)")
                _ct_stat = _log_command_follow_stats(iteration + 1, _collect_command_outcomes(actor_results))
                if _ct_stat is not None:
                    _cmd_track_hist.append(_ct_stat)
                
                if iteration < N_ITERATIONS - 1:
                    commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                    commands_batch = _normalize_commands_for_actor_and_log(commands_batch, iteration + 1)
                    learner_future = learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES, use_training_cache=True)
                    next_actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                    loss = loss_pending
                else:
                    # 最終イテレーション: 上で完了した並行 learn のあと、直前 Actor 波が ReplayBuffer に積んだ分をもう一度学習する
                    # （並行 learn の get_all が Actor 完了前に走ると最終バッチが取りこぼされるため）
                    t_learner_start = time.time()
                    loss = ray.get(learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES, use_training_cache=True))
                    t_learner = time.time() - t_learner_start
                    if DEBUG:
                        print(
                            f"[Learner] 最終イテレーション: pending並行学習損失={loss_pending:.4f}, "
                            f"最終バッファ学習損失={loss:.4f}"
                        )
                    if _PROFILE_MODE:
                        print(f"[PROFILE Iter {iteration+1}] Learner実行（最終・2段）: {t_learner:.3f}s")
                    learner_future = None
                    next_actor_futures = None
        else:
            # 従来の逐次モード（Actor完了→Learner実行）
            if DEBUG:
                print("Actorが改良されたエピソードを生成中...")
                print("※ PCNエージェントの_choose_commandsと_nlargestメソッドにより改善された目標値を使用")
            commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
            commands_batch = _normalize_commands_for_actor_and_log(commands_batch, iteration + 1)
            actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
            t_actor_start = time.time()
            actor_results = ray.get(actor_futures)
            t_actor = time.time() - t_actor_start
            if _PROFILE_MODE:
                print(f"[PROFILE Iter {iteration+1}] Actor実行: {t_actor:.3f}s (合計{_episodes_generated_sum(actor_results)}エピソード)")
            _ct_stat = _log_command_follow_stats(iteration + 1, _collect_command_outcomes(actor_results))
            if _ct_stat is not None:
                _cmd_track_hist.append(_ct_stat)            
            if DEBUG:
                print("Learnerが改良された経験で学習を実行中")
            t_learner_start = time.time()
            loss = ray.get(learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES, use_training_cache=True))
            t_learner = time.time() - t_learner_start
            if _PROFILE_MODE:
                print(f"[PROFILE Iter {iteration+1}] Learner実行: {t_learner:.3f}s")

        print(f"イテレーション {iteration + 1} 学習完了：平均損失: {loss:.4f}")
        # Phase3凍結検知: skip率(非有限勾配/スパイクでstep不成立の割合)を毎iteration記録。
        # 警告printは上限20で沈黙するため、ここが唯一の定量ログ。
        try:
            _sk, _st = ray.get(learner.get_step_skip_stats.remote())
            _d_sk, _d_st = _sk - _prev_skip_stats[0], _st - _prev_skip_stats[1]
            _prev_skip_stats = (_sk, _st)
            _tot = _d_sk + _d_st
            if _tot > 0:
                _rate = _d_sk / _tot
                _mark = " ⚠️FROZEN?" if _rate > 0.5 else ""
                print(f"[STEP_SKIP] iter={iteration + 1} skip率={_rate:.1%} (skip={_d_sk}/step={_d_st} 累計skip={_sk}){_mark}")
        except Exception:
            pass
        
        # メモリ解放: 学習完了後にReplayBufferをクリア（メモリスピルを防ぐため）
        # get_all_episodes()は既にバッファをクリアするため、明示的なクリアは不要
        # ただし、メモリ使用量を確認
        if DEBUG and iteration % 2 == 0:  # 2イテレーションごとに確認
            buffer_stats = ray.get(buffer.get_stats.remote())
            print(f"[メモリ管理] ReplayBuffer統計: サイズ={buffer_stats['buffer_size']}, 利用率={buffer_stats['utilization']:.1%}")
        
        if _GC_EACH_ITER:
            gc.collect()
        
        # 学習履歴を記録
        training_history['iterations'].append(iteration + 1)
        training_history['losses'].append(loss)
        training_history.setdefault('command_outcomes', []).append(_collect_command_outcomes(actor_results if actor_results is not None else []))
        

        
        # 定期的に評価を実行（最終 iter は EVAL_INTERVAL の倍数でなくても保存・EvalギャップFB）
        _is_eval_iter = (iteration + 1) % EVAL_INTERVAL == 0 or (iteration + 1) == N_ITERATIONS
        if _is_eval_iter:
            if DEBUG:
                print(f"\n=== イテレーション {iteration + 1} の評価 ===")
                print("※ 改良された経験によるパレートフロントの改善を確認")
            
            if USE_DISTRIBUTED_EVAL:
                _eval_n = EVAL_SAMPLES if EVAL_SAMPLES_DISTRIBUTED <= 0 else EVAL_SAMPLES_DISTRIBUTED
                e_returns, e_values, distances, map_fin = _distributed_evaluate_episodes(
                    learner, actors, _eval_n
                )
                training_history['last_eval'] = {
                    'iteration': iteration + 1,
                    'e_returns': e_returns,
                    'e_values': e_values,
                }
                if DEBUG:
                    print("分散評価を使用しました")
            else:
                # 通常の評価を使用
                e_returns, e_values, distances, map_fin = ray.get(
                    learner.evaluate.remote(
                        n=EVAL_SAMPLES,
                        training_iteration=iteration + 1,
                        eval_diag_path=eval_diag_path,
                    )
                )
                training_history['last_eval'] = {
                    'iteration': iteration + 1,
                    'e_returns': e_returns,
                    'e_values': e_values,
                }
                if DEBUG:
                    print("通常評価を使用しました")
            
            # PCNエージェントのevaluate()で既に詳細な出力が行われているため、
            # ここでは簡潔な確認のみ行う
            if DEBUG:
                print(f"評価完了 - 非支配解の数: {len(e_returns)}")
                if len(distances) > 0:
                    avg_distance = np.mean(distances)
                    min_distance = np.min(distances)
                    max_distance = np.max(distances)
        # print(f"Distance統計 - 平均: {avg_distance:.4f}, 最小: {min_distance:.4f}, 最大: {max_distance:.4f}")
            
            # 評価時にモデルを保存
            model_save_dir = f"{execution_dir}/iteration_{iteration + 1:03d}"
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = f"{model_save_dir}/model_iter_{iteration + 1:03d}.pth"
            saved_model_path = ray.get(learner.save_model.remote(model_save_path))
            if saved_model_path and DEBUG:
                print(f"モデルを保存しました: {saved_model_path}")
            
            # パレートフロントサイズ = 実数値空間の非支配解の数（e_returns の総数ではない）
            non_dom_values = len(get_non_dominated_inds_minimize(np.array(e_values, dtype=np.float64)))
            training_history['pareto_front_sizes'].append(non_dom_values)
            training_history['distances'].append(distances if len(distances) > 0 else [])

            # early-stop: 全評価候補を保持し、毎回「全候補をカバーするnadir」で再評価して最良iterを選ぶ。
            # 初回evalのnadirを固定すると初回front自身のHVが過小評価され(端点=ref→体積ゼロ)、最初が最良の
            # runで選び損ねる(検証済: es5 it10=85%を取り逃す)。拡大refで全候補を同一基準にすると真ベストと
            # 4/5一致・実用HV 68→77%・効率run1/5→3/5。詳細 docs/repro_512.html §8末。
            if _EARLYSTOP and len(e_values) > 0:
                try:
                    _vals = np.array(e_values, dtype=np.float64)
                    _nd = _vals[get_non_dominated_inds_minimize(_vals)]
                    _es_candidates.append((iteration + 1, _nd))
                    if _es_ref is not None:
                        _ref_use = _es_ref  # 環境変数で絶対nadirを与えた場合は固定
                    else:
                        _ref_use = np.vstack([c[1] for c in _es_candidates]).max(axis=0) * 1.1
                    _best_it, _best_hv = max(
                        ((it, _hypervolume_2d_min(nd, _ref_use)) for it, nd in _es_candidates),
                        key=lambda t: t[1])
                    if _best_it != _es_best_iter:
                        _es_best_iter = _best_it
                        _src_ck = f"{execution_dir}/iteration_{_best_it:03d}/model_iter_{_best_it:03d}.pth"
                        if os.path.isfile(_src_ck):
                            shutil.copyfile(_src_ck, _es_best_path)
                        else:
                            ray.get(learner.save_model.remote(_es_best_path))  # 現iterが最良時のフォールバック
                        print(f"[EARLYSTOP] iter {iteration + 1}: best=iter{_best_it} "
                              f"(達成HV={_best_hv:.4e}, 全候補nadir再評価) → best_model更新")
                    else:
                        print(f"[EARLYSTOP] iter {iteration + 1}: best=iter{_es_best_iter} 維持")
                except Exception as _exc:
                    print(f"[EARLYSTOP] HV計算失敗(続行): {_exc}")

            _plot_dir = execution_dir
            if os.environ.get("PCN_EVAL_GAP_FEEDBACK", "0") == "1":
                try:
                    gap_report = _driver_eval_gap_feedback(
                        learner,
                        actors,
                        iteration + 1,
                        _plot_dir,
                        int(config["param_env"].get("n_jobs", N_JOBS)),
                    )
                    weak = {
                        k: round(v["mean_gap"], 0)
                        for k, v in (gap_report or {}).items()
                        if v.get("n", 0) and np.isfinite(v.get("mean_gap", np.nan))
                    }
                    if weak:
                        print(f"[EVAL_GAP] iter {iteration + 1} mean_gaps={weak}")
                except Exception as exc:
                    print(f"[EVAL_GAP] 弱点帯域フィードバック失敗: {exc}")
            elif os.environ.get("DISTRIBUTED_PCN_LIVE_UNIFORM_PF", "0") == "1":
                try:
                    _driver_live_uniform_pf_plot(
                        learner,
                        actors,
                        iteration + 1,
                        _plot_dir,
                        int(config["param_env"].get("n_jobs", N_JOBS)),
                    )
                except Exception as exc:
                    print(f"[LIVE_PF] 均等格子 PF 図の保存失敗: {exc}")
        else:
            training_history['pareto_front_sizes'].append(None)
            training_history['distances'].append(None)
        

        if ENABLE_VISUALIZATION and (iteration + 1) % VISUALIZATION_INTERVAL == 0:

            print(f"\n=== イテレーション {iteration + 1} での可視化 ===")
            
            # イテレーション用のディレクトリを作成
            save_dir = f"{execution_dir}/iteration_{iteration + 1:03d}"
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                import matplotlib.pyplot as plt
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                
                # 評価と同時の場合はevaluate結果を再利用（二重実行を回避）
                if (iteration + 1) % EVAL_INTERVAL == 0:
                    current_e_returns, current_e_values = e_returns, e_values
                else:
                    current_e_returns, current_e_values, _, _ = ray.get(learner.evaluate.remote(n=EVAL_SAMPLES_VISUALIZATION))
                archive_snapshot = ray.get(learner.get_archive_pareto_snapshot.remote())
                
                # 可視化時にモデルを保存（EVAL_INTERVALと重なる場合は既に保存済みなのでスキップ）
                if (iteration + 1) % EVAL_INTERVAL != 0:
                    model_save_path = f"{save_dir}/model_visualization_{iteration + 1:03d}.pth"
                    ray.get(learner.save_model.remote(model_save_path))
                
                if len(current_e_returns) > 0 and len(current_e_values) > 0:
                #     # タイムスタンプを取得
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                #     # 1. 軽量化された報酬空間でのパレートフロント（最大化目的）
                #     plt.figure(figsize=(8, 6))  # サイズを小さく
                    
                #     # 初期ランダム行動で収集した全ての点をプロット
                #     # initial_all_returns = np.array(initial_pareto_front["returns"])
                #     # plt.scatter(initial_all_returns[:, 0], initial_all_returns[:, 1], c='lightblue', alpha=0.6, label='Initial Random Solutions', s=30)  # サイズを小さく
                    
                #     # # 初期パレートフロントを表示（比較用）
                #     # if training_history['initial_pareto_front'] is not None:
                #     #     initial_pf = training_history['initial_pareto_front']['pareto_front_reward']
                #     #     plt.scatter(initial_pf[:, 0], initial_pf[:, 1], c='orange', s=50, label='Initial Pareto Front', zorder=4, marker='s')
                    
                #     # 現在の学習結果の非支配解を強調表示
                    
                #     current_all_returns = np.array(current_e_returns)
                #     non_dominated_inds = get_non_dominated_inds(current_all_returns)
                #     pareto_front_returns = current_all_returns[non_dominated_inds]
                    
                #     # デバッグ：現在の値の範囲を確認
                #     print(f"=== Current Values Debug (Iter {iteration + 1}) ===")
                #     print(f"Current returns range: X[{current_all_returns[:, 0].min():.1f}, {current_all_returns[:, 0].max():.1f}], Y[{current_all_returns[:, 1].min():.1f}, {current_all_returns[:, 1].max():.1f}]")
                #     if initial_axis_ranges and 'rewards' in initial_axis_ranges:
                #         print(f"Initial axis range: X[{initial_axis_ranges['rewards']['x_min']:.1f}, {initial_axis_ranges['rewards']['x_max']:.1f}], Y[{initial_axis_ranges['rewards']['y_min']:.1f}, {initial_axis_ranges['rewards']['y_max']:.1f}]")
                    
                #     plt.scatter(current_all_returns[:, 0], current_all_returns[:, 1], c='blue', s=60, label='Current All Returns', zorder=5)  # サイズを小さく
                #     plt.scatter(pareto_front_returns[:, 0], pareto_front_returns[:, 1], c='red', s=60, label='Current Pareto Front', zorder=5)  # サイズを小さく
                #     print(f"n_points_first: {len(pareto_front_returns)}", f"n_points: {len(current_all_returns)}")
                    
                #     # パレートフロントの線を描画
                #     if len(pareto_front_returns) > 1:
                #         # パレートフロントをソート
                #         sorted_indices = np.lexsort((pareto_front_returns[:, 1], pareto_front_returns[:, 0]))
                #         sorted_pareto = pareto_front_returns[sorted_indices]
                #         plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=1.5, alpha=0.8)  # 線を細く
                    
                #     # 軸範囲を現在の値に焦点を当てて設定
                #     current_x_min, current_x_max = current_all_returns[:, 0].min(), current_all_returns[:, 0].max()
                #     current_y_min, current_y_max = current_all_returns[:, 1].min(), current_all_returns[:, 1].max()
                    
                #     # 現在の値範囲を基準に適度なマージンを追加（15%）
                #     x_range = current_x_max - current_x_min
                #     y_range = current_y_max - current_y_min
                #     x_margin = max(x_range * 0.15, abs(current_x_min) * 0.05)  # 最小マージンを確保
                #     y_margin = max(y_range * 0.15, abs(current_y_min) * 0.05)
                    
                #     plt.xlim(current_x_min - x_margin, current_x_max + x_margin)
                #     plt.ylim(current_y_min - y_margin, current_y_max + y_margin)
                    
                #     print(f"Focused axis range: X[{current_x_min - x_margin:.1f}, {current_x_max + x_margin:.1f}], Y[{current_y_min - y_margin:.1f}, {current_y_max + y_margin:.1f}]")
                    
                #     plt.title(f'Pareto Front (Reward) - Iter {iteration + 1}\nNon-dominated: {len(non_dominated_inds)}', fontsize=10)  # タイトルを短く
                #     plt.xlabel('Reward 1', fontsize=9)
                #     plt.ylabel('Reward 2', fontsize=9)
                #     plt.legend(fontsize=8)
                #     plt.grid(True, alpha=0.3)
                    
                #     plt.tight_layout()
                    
                #     # タイムスタンプ付きファイル名で保存（新規作成）
                #     reward_plot_path = f"{save_dir}/pareto_front_rewards_current_{timestamp}.png"
                #     plt.savefig(reward_plot_path, dpi=150, bbox_inches='tight')  # dpiを下げる
                #     plt.close()
                #     if DEBUG:
                #         print(f"軽量化報酬空間パレートフロント更新: {reward_plot_path}")
                    
                    # 2. 軽量化された実数値空間でのパレートフロント（最小化目的）
                    plt.figure(figsize=(8, 6))  # サイズを小さく
                    
                    # 初期ランダム行動で収集した全ての点をプロット
                    # initial_all_values = np.array(initial_pareto_front["values"])
                    # plt.scatter(initial_all_values[:, 0], initial_all_values[:, 1], c='lightgreen', alpha=0.6, label='Initial Random Solutions', s=30)  # サイズを小さく
                    
                    # 初期パレートフロントを表示（比較用）
                    # if training_history['initial_pareto_front'] is not None:
                    #     initial_pf_values = training_history['initial_pareto_front']['pareto_front_values']
                    #     plt.scatter(initial_pf_values[:, 0], initial_pf_values[:, 1], c='orange', s=50, label='Initial Pareto Front', zorder=4, marker='s')
                    
                    current_all_values = np.array(current_e_values)
                    current_all_values_vis = _dedupe_points_for_plot(current_all_values)
                    non_dominated_inds_values = get_non_dominated_inds_minimize(current_all_values_vis)
                    pareto_front_values = current_all_values_vis[non_dominated_inds_values]
                    archive_pf_values = _dedupe_points_for_plot(
                        np.array(archive_snapshot.get("pareto_front_values", []), dtype=np.float64)
                    )
                    
                    # デバッグ：現在の値の範囲を確認
                    # print(f"Current values range: X[{current_all_values[:, 0].min():.1f}, {current_all_values[:, 0].max():.1f}], Y[{current_all_values[:, 1].min():.1f}, {current_all_values[:, 1].max():.1f}]")
                    # if initial_axis_ranges and 'values' in initial_axis_ranges:
                    #     print(f"Initial values axis range: X[{initial_axis_ranges['values']['x_min']:.1f}, {initial_axis_ranges['values']['x_max']:.1f}], Y[{initial_axis_ranges['values']['y_min']:.1f}, {initial_axis_ranges['values']['y_max']:.1f}]")
                    
                    plt.scatter(
                        current_all_values_vis[:, 0], current_all_values_vis[:, 1],
                        c='blue', s=60, label='Eval All Values', zorder=5,
                    )
                    plt.scatter(
                        pareto_front_values[:, 0], pareto_front_values[:, 1],
                        c='red', s=60, label='Eval Pareto Front', zorder=5,
                    )
                    if archive_pf_values.size > 0:
                        plt.scatter(archive_pf_values[:, 0], archive_pf_values[:, 1], c='cyan', s=42, alpha=0.85, marker='D', label='Archive Pareto Front', zorder=6)
                        if len(archive_pf_values) > 1:
                            archive_sorted_i = np.lexsort((archive_pf_values[:, 1], archive_pf_values[:, 0]))
                            archive_sorted = archive_pf_values[archive_sorted_i]
                            plt.plot(archive_sorted[:, 0], archive_sorted[:, 1], color='cyan', linewidth=1.2, alpha=0.65, linestyle=':')
                    # 可視化: command（狙い）→ achieved（実際）の対応
                    iter_outcomes = []
                    if 'command_outcomes' in training_history and len(training_history['command_outcomes']) >= (iteration + 1):
                        iter_outcomes = training_history['command_outcomes'][iteration]
                    base_vals = None
                    cmd_vals = None
                    ach_vals = None
                    if _VIS_COMMAND_ARROWS and iter_outcomes:
                        max_arrows = 40
                        if len(iter_outcomes) > max_arrows:
                            sampled_idx = np.random.choice(len(iter_outcomes), size=max_arrows, replace=False)
                            iter_outcomes = [iter_outcomes[i] for i in sampled_idx]
                        base_ret_vals = np.array(
                            [o["base_return"] if "base_return" in o else o["command_return"] for o in iter_outcomes],
                            dtype=np.float32
                        )
                        base_vals = np.stack(
                            [-base_ret_vals[:, 1], -base_ret_vals[:, 0] / n_jobs_for_value_scale],
                            axis=1,
                        )
                        cmd_vals = np.array([o["command_values"] for o in iter_outcomes], dtype=np.float32)
                        ach_vals = np.array([o["achieved_values"] for o in iter_outcomes], dtype=np.float32)
                        ach_vals, cmd_vals, base_vals = _dedupe_aligned_points_for_plot(
                            ach_vals, cmd_vals, base_vals,
                        )
                        print(f"[VIS Iter {iteration + 1}] command-achieved pairs plotted: {len(ach_vals)}")
                        base_nd_i_vals = get_non_dominated_inds_minimize(base_vals)
                        base_pf_vals = base_vals[base_nd_i_vals]
                        plt.scatter(base_vals[:, 0], base_vals[:, 1], c='orange', s=22, alpha=0.55, label='Command Base Values', zorder=4)
                        plt.scatter(base_pf_vals[:, 0], base_pf_vals[:, 1], c='black', s=35, alpha=0.8, marker='x', label='Command Base PF', zorder=6)
                        if len(base_pf_vals) > 1:
                            base_sorted_i_vals = np.lexsort((base_pf_vals[:, 1], base_pf_vals[:, 0]))
                            base_sorted_vals = base_pf_vals[base_sorted_i_vals]
                            plt.plot(base_sorted_vals[:, 0], base_sorted_vals[:, 1], color='black', linewidth=1.0, alpha=0.55, linestyle='--')
                        plt.scatter(cmd_vals[:, 0], cmd_vals[:, 1], c='purple', s=20, alpha=0.5, label='Command Targets', zorder=4)
                        plt.quiver(
                            base_vals[:, 0], base_vals[:, 1],
                            cmd_vals[:, 0] - base_vals[:, 0], cmd_vals[:, 1] - base_vals[:, 1],
                            angles='xy', scale_units='xy', scale=1,
                            color='purple', alpha=0.30, width=0.0025, zorder=3
                        )
                        plt.scatter(ach_vals[:, 0], ach_vals[:, 1], c='green', s=20, alpha=0.5, label='Achieved Points', zorder=4)
                        plt.quiver(
                            cmd_vals[:, 0], cmd_vals[:, 1],
                            ach_vals[:, 0] - cmd_vals[:, 0], ach_vals[:, 1] - cmd_vals[:, 1],
                            angles='xy', scale_units='xy', scale=1,
                            color='gray', alpha=0.35, width=0.0025, zorder=3
                        )
                    
                    # パレートフロントの線を描画
                    if len(pareto_front_values) > 1:
                        # パレートフロントをソート
                        sorted_indices = np.lexsort((pareto_front_values[:, 1], pareto_front_values[:, 0]))
                        sorted_pareto = pareto_front_values[sorted_indices]
                        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=1.5, alpha=0.8)  # 線を細く
                    
                    axis_points = [current_all_values_vis]
                    if archive_pf_values.size > 0:
                        axis_points.append(archive_pf_values)
                    if base_vals is not None and len(base_vals) > 0:
                        axis_points.append(base_vals)
                    if cmd_vals is not None and len(cmd_vals) > 0:
                        axis_points.append(cmd_vals)
                    if ach_vals is not None and len(ach_vals) > 0:
                        axis_points.append(ach_vals)
                    axis_data = np.vstack(axis_points)
                    current_x_min, current_x_max = axis_data[:, 0].min(), axis_data[:, 0].max()
                    current_y_min, current_y_max = axis_data[:, 1].min(), axis_data[:, 1].max()
                    
                    # 現在の値範囲を基準に適度なマージンを追加（15%）
                    x_range = current_x_max - current_x_min
                    y_range = current_y_max - current_y_min
                    x_margin = max(x_range * 0.15, current_x_min * 0.05)  # 最小マージンを確保
                    y_margin = max(y_range * 0.15, 1.0)  # Y軸は最低でも1の幅を確保
                    
                    # Y軸の下限は0以下にならないように調整
                    y_min_adjusted = max(0, current_y_min - y_margin)
                    
                    plt.xlim(current_x_min - x_margin, current_x_max + x_margin)
                    plt.ylim(y_min_adjusted, current_y_max + y_margin)
                    
                    print(f"Focused values axis range: X[{current_x_min - x_margin:.1f}, {current_x_max + x_margin:.1f}], Y[{y_min_adjusted:.1f}, {current_y_max + y_margin:.1f}]")
                    
                    plt.title(
                        f'Evaluation Pareto Front (Value) - Iter {iteration + 1}\n'
                        f'Non-dominated: {len(non_dominated_inds_values)} (unique points)',
                        fontsize=10,
                    )
                    plt.xlabel('Cost', fontsize=9)
                    plt.ylabel('Average Waiting Time', fontsize=9)
                    plt.legend(fontsize=8)
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # タイムスタンプ付きファイル名で保存（新規作成）
                    values_plot_path = f"{save_dir}/pareto_front_values_current_{timestamp}.png"
                    plt.savefig(values_plot_path, dpi=_VIS_PLOT_DPI, bbox_inches='tight')
                    plt.close()
                    if DEBUG:
                        print(f"軽量化実数値空間パレートフロント更新: {values_plot_path}")

                    # 2.5 Reward空間（既定OFF: DISTRIBUTED_PCN_VIS_REWARD_PLOT=1 で有効）
                    if _VIS_REWARD_PLOT:
                        plt.figure(figsize=(8, 6))
                        current_all_returns = np.array(current_e_returns)
                        current_all_returns_vis = _dedupe_points_for_plot(current_all_returns)
                        non_dominated_inds_reward = get_non_dominated_inds(current_all_returns_vis)
                        pareto_front_returns = current_all_returns_vis[non_dominated_inds_reward]
                        archive_pf_returns = _dedupe_points_for_plot(
                            np.array(archive_snapshot.get("pareto_front_reward", []), dtype=np.float64)
                        )
                        plt.scatter(
                            current_all_returns_vis[:, 0], current_all_returns_vis[:, 1],
                            c='blue', s=60, label='Eval All Returns', zorder=5,
                        )
                        plt.scatter(
                            pareto_front_returns[:, 0], pareto_front_returns[:, 1],
                            c='red', s=60, label='Eval Pareto Front', zorder=5,
                        )
                        if archive_pf_returns.size > 0:
                            plt.scatter(
                                archive_pf_returns[:, 0], archive_pf_returns[:, 1],
                                c='cyan', s=42, alpha=0.85, marker='D', label='Archive Pareto Front', zorder=6,
                            )
                            if len(archive_pf_returns) > 1:
                                archive_sorted_i_r = np.lexsort(
                                    (archive_pf_returns[:, 1], archive_pf_returns[:, 0])
                                )
                                archive_sorted_r = archive_pf_returns[archive_sorted_i_r]
                                plt.plot(
                                    archive_sorted_r[:, 0], archive_sorted_r[:, 1],
                                    color='cyan', linewidth=1.2, alpha=0.65, linestyle=':',
                                )
                        if len(pareto_front_returns) > 1:
                            sorted_indices_r = np.lexsort(
                                (pareto_front_returns[:, 1], pareto_front_returns[:, 0])
                            )
                            sorted_pareto_r = pareto_front_returns[sorted_indices_r]
                            plt.plot(sorted_pareto_r[:, 0], sorted_pareto_r[:, 1], 'r-', linewidth=1.5, alpha=0.8)
                        axis_points_r = [current_all_returns_vis]
                        if archive_pf_returns.size > 0:
                            axis_points_r.append(archive_pf_returns)
                        axis_data_r = np.vstack(axis_points_r)
                        x_min_r, x_max_r = axis_data_r[:, 0].min(), axis_data_r[:, 0].max()
                        y_min_r, y_max_r = axis_data_r[:, 1].min(), axis_data_r[:, 1].max()
                        x_range_r = x_max_r - x_min_r
                        y_range_r = y_max_r - y_min_r
                        x_margin_r = max(x_range_r * 0.15, abs(x_min_r) * 0.05)
                        y_margin_r = max(y_range_r * 0.15, abs(y_min_r) * 0.05)
                        plt.xlim(x_min_r - x_margin_r, x_max_r + x_margin_r)
                        plt.ylim(y_min_r - y_margin_r, y_max_r + y_margin_r)
                        plt.title(
                            f'Pareto Front (Reward) - Iter {iteration + 1}\n'
                            f'Non-dominated: {len(non_dominated_inds_reward)} (unique points)',
                            fontsize=10,
                        )
                        plt.xlabel('Reward 1', fontsize=9)
                        plt.ylabel('Reward 2', fontsize=9)
                        plt.legend(fontsize=8)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        rewards_plot_path = f"{save_dir}/pareto_front_rewards_current_{timestamp}.png"
                        plt.savefig(rewards_plot_path, dpi=_VIS_PLOT_DPI, bbox_inches='tight')
                        plt.close()
                        if DEBUG:
                            print(f"軽量化報酬空間パレートフロント更新: {rewards_plot_path}")
                    
                    # # 3. 軽量化された学習履歴の可視化（シンプル版）
                    # plt.figure(figsize=(15, 5))  # サイズを拡大
                    
                    # # サブプロット1: 損失の推移
                    # plt.subplot(1, 3, 1)
                    # plt.plot(training_history['iterations'], training_history['losses'], 'b-', linewidth=1.5)
                    # plt.title('Training Loss', fontsize=9)
                    # plt.xlabel('Iteration', fontsize=8)
                    # plt.ylabel('Loss', fontsize=8)
                    # plt.grid(True, alpha=0.3)
                    
                    # # サブプロット2: パレートフロントサイズの推移
                    # plt.subplot(1, 3, 2)
                    # valid_pf_sizes = [size for size in training_history['pareto_front_sizes'] if size is not None]
                    # valid_iterations = [(i+1)*EVAL_INTERVAL for i, size in enumerate(valid_pf_sizes)]
                    # if valid_iterations:
                    #     plt.plot(valid_iterations, valid_pf_sizes, 'r-', linewidth=1.5, marker='o', markersize=4)
                    # plt.title('Pareto Front Size', fontsize=9)
                    # plt.xlabel('Iteration', fontsize=8)
                    # plt.ylabel('Non-dominated Solutions', fontsize=8)
                    # plt.grid(True, alpha=0.3)
                    
                    # # サブプロット3: Distance統計の推移
                    # plt.subplot(1, 3, 3)
                    # valid_distances = [dist for dist in training_history['distances'] if dist is not None and len(dist) > 0]
                    # valid_distance_iterations = [(i+1)*EVAL_INTERVAL for i, dist in enumerate(valid_distances)]
                    # if valid_distance_iterations:
                    #     avg_distances = [np.mean(dist) for dist in valid_distances]
                    #     min_distances = [np.min(dist) for dist in valid_distances]
                    #     max_distances = [np.max(dist) for dist in valid_distances]
                        
                    #     plt.plot(valid_distance_iterations, avg_distances, 'g-', linewidth=1.5, marker='o', markersize=4, label='Average')
                    #     plt.fill_between(valid_distance_iterations, min_distances, max_distances, alpha=0.3, color='green', label='Min-Max Range')
                    #     plt.legend(fontsize=7)
                    
                    # plt.title('Distance Statistics', fontsize=9)
                    # plt.xlabel('Iteration', fontsize=8)
                    # plt.ylabel('Distance', fontsize=8)
                    # plt.grid(True, alpha=0.3)
                    
                    # plt.tight_layout()
                    
                    # # 軽量化された保存（解像度を下げる）
                    # history_plot_path = f"{save_dir}/learning_history_current.png"
                    # plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')  # dpiを下げる
                    # plt.close()
                    # if DEBUG:
                    #     print(f"軽量化学習履歴更新: {history_plot_path}")
                    
                    # 4. 軽量化された詳細データの保存（簡潔版）
                    details_path = f"{save_dir}/pareto_front_details_current_{timestamp}.txt"
                    with open(details_path, 'w', encoding='utf-8') as f:
                        f.write(f"=== 軽量化パレートフロント詳細 (Iter {iteration + 1}) ===\n")
                        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"損失: {training_history['losses'][-1]:.4f}\n")
                        # f.write(f"報酬空間非支配解数: {len(non_dominated_inds)}\n")
                        f.write(f"実数値空間非支配解数: {len(non_dominated_inds_values)}\n")
                        f.write(f"Archive件数: {archive_snapshot.get('n_archive', 0)}\n")
                        f.write(f"Archiveユニーク実数値数: {archive_snapshot.get('n_unique_values', 0)}\n")
                        f.write(f"Archive実数値PF数: {len(archive_pf_values) if archive_pf_values.size > 0 else 0}\n")
                        
                        # 報酬空間の非支配解を詳細に記録
                        # f.write(f"\n=== 報酬空間の非支配解 (Iter {iteration + 1}) ===\n")
                        # for i, idx in enumerate(non_dominated_inds):
                        #     f.write(f"解{i+1}: {current_e_returns[idx]}\n")
                        
                        # 実数値空間の非支配解を詳細に記録
                        f.write(f"\n=== 実数値空間の非支配解 (Iter {iteration + 1}) ===\n")
                        for i, idx in enumerate(non_dominated_inds_values):
                            f.write(f"解{i+1}: {current_e_values[idx]}\n")
                    
                    if DEBUG:
                        print(f"軽量化詳細データ更新: {details_path}")
                        print(f"=== イテレーション {iteration + 1} の軽量化可視化完了 ===")
                        print(f"軽量化ファイル更新完了: '{save_dir}' ディレクトリ")
                
                else:
                    if DEBUG:
                        print("警告: パレートフロントのデータが取得できませんでした。")
                        
            except Exception as e:
                print(f"可視化中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
        
        # 学習後の重みを取得して確認
        # 2回のray.get()を避けるため、get_weights()を直接呼び出す
        weights = ray.get(learner.get_weights.remote())
        if DEBUG:
            print("学習が完了し、新しい重みが生成されました")

        # 学習の進捗を表示
        if DEBUG and iteration > 0:
            loss_improvement = training_history['losses'][-2] - training_history['losses'][-1]
            print(f"損失の改善: {loss_improvement:.4f}")
            
            if len(training_history['pareto_front_sizes']) > 1 and training_history['pareto_front_sizes'][-1] is not None:
                pf_improvement = training_history['pareto_front_sizes'][-1] - training_history['pareto_front_sizes'][-2]
                print(f"パレートフロントサイズの変化: {pf_improvement:+d}")
            
            # Distanceの改善を表示
            if len(training_history['distances']) > 1 and training_history['distances'][-1] is not None and len(training_history['distances'][-1]) > 0:
                if training_history['distances'][-2] is not None and len(training_history['distances'][-2]) > 0:
                    prev_avg_distance = np.mean(training_history['distances'][-2])
                    curr_avg_distance = np.mean(training_history['distances'][-1])
                    distance_improvement = prev_avg_distance - curr_avg_distance
                    print(f"Distanceの改善: {distance_improvement:.4f} (平均: {curr_avg_distance:.4f})")
            
            # バッファの統計情報を表示（10イテレーションごと）
            if iteration % 10 == 0:
                buffer_stats = ray.get(buffer.get_stats.remote())
                print(f"\n=== イテレーション {iteration} のバッファ統計 ===")
                print(f"バッファサイズ: {buffer_stats['buffer_size']}")
                print(f"ユニークエピソード数: {buffer_stats['unique_episodes']}")
                print(f"利用率: {buffer_stats['utilization']:.2%}")
                print("=" * 40)
        


    # フェーズ3の完了時間を記録
    if TIME_DEBUG:
        phase3_end_time = time.time()
        phase3_duration = phase3_end_time - phase3_start_time
        print(f"\n{'='*40}")
        print(f"フェーズ3完了: 改良された経験の実現")
        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {phase3_duration:.2f}秒 ({phase3_duration/60:.2f}分)")
        print(f"実行イテレーション数: {len(training_history['iterations'])}")
        print(f"{'='*40}")

    if _PROFILE_MODE:
        try:
            learn_stats = ray.get(learner.get_learn_profile_summary.remote())
            n_it = max(len(training_history["iterations"]), 1)
            phase3_wall = time.time() - phase3_start_time
            print(
                f"[PROFILE Phase3] per-iter mean: get={learn_stats.get('get_episodes_mean', 0):.3f}s "
                f"add={learn_stats.get('add_episodes_mean', 0):.3f}s "
                f"update={learn_stats.get('update_mean', 0):.3f}s "
                f"(phase3 wall={phase3_wall:.1f}s / {n_it} iter)"
            )
            profile_path = os.path.join(execution_dir, "phase3_learn_profile.json")
            with open(profile_path, "w", encoding="utf-8") as pf:
                json.dump(learn_stats, pf, indent=2)
            print(f"[PROFILE] Learner内訳 JSON: {profile_path}")
        except Exception as e:
            print(f"[PROFILE] サマリー取得失敗: {e}")

    # イテレーションごとの学習サマリーを JSON で保存（実験比較用）
    try:
        import json as _json
        _rows = []
        for _i in range(len(training_history['losses'])):
            _it = training_history['iterations'][_i] if _i < len(training_history['iterations']) else _i + 1
            _loss = float(training_history['losses'][_i])
            _pf = training_history['pareto_front_sizes'][_i] if _i < len(training_history['pareto_front_sizes']) else None
            _dlist = training_history['distances'][_i] if _i < len(training_history['distances']) else None
            _avg_d = None
            _min_d = None
            _max_d = None
            if _dlist is not None and len(_dlist) > 0:
                _avg_d = float(np.mean(_dlist))
                _min_d = float(np.min(_dlist))
                _max_d = float(np.max(_dlist))
            _rows.append({
                "iteration": int(_it),
                "loss": _loss,
                "pareto_front_size": _pf,
                "distance_avg": _avg_d,
                "distance_min": _min_d,
                "distance_max": _max_d,
            })
        _summary = {
            "n_jobs": int(config['param_env'].get('n_jobs', N_JOBS)),
            "n_iterations_config": int(N_ITERATIONS),
            "eval_interval": int(EVAL_INTERVAL),
            "use_event_obs": bool(_USE_EVENT_OBS),
            "event_to_bitmap": bool(_USE_EVENT_OBS and learner_bitmap_enabled()),
            "rows": _rows,
        }
        _summary_path = os.path.join(execution_dir, "training_iteration_summary.json")
        with open(_summary_path, "w", encoding="utf-8") as _sf:
            _json.dump(_summary, _sf, indent=2, allow_nan=False)
        print(f"[SUMMARY] イテレーション別サマリーを保存: {_summary_path}")
    except Exception as _e:
        print(f"[SUMMARY] JSON 保存に失敗: {_e}")

    # [CMD-TRACK] 指令↔達成の追従距離(固定スケール正規化MSE)の iteration 履歴を保存＋プロット。
    # 学習は一切不変（診断のみ）。ユーザー要望「指定点をどれだけ再現できているかの距離を見る」可視化基盤。
    # 有効化には DISTRIBUTED_PCN_CMD_OUTCOMES=1（command_outcomes 収集）が必要。
    try:
        if _cmd_track_hist:
            _track_path = os.path.join(execution_dir, "cmd_track_history.json")
            with open(_track_path, "w", encoding="utf-8") as _tf:
                _json.dump(_cmd_track_hist, _tf, indent=2, allow_nan=False)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            _its = [d["iter"] for d in _cmd_track_hist]
            _fig, _ax = _plt.subplots(figsize=(8.4, 5.0))
            _ax.plot(_its, [d["mse_total"] for d in _cmd_track_hist], "-o", color="#1a73e8", label="total")
            _ax.plot(_its, [d["mse_cost"] for d in _cmd_track_hist], "-s", color="#d93025", label="cost")
            _ax.plot(_its, [d["mse_wait"] for d in _cmd_track_hist], "-^", color="#188038", label="wait")
            _ax.plot(_its, [d["miss_cost"] for d in _cmd_track_hist], "--", color="#e37400", alpha=.7, label="miss(one-sided) cost")
            _ax.set_xlabel("iteration")
            _ax.set_ylabel("cmd-achieve normalized MSE (lower=better following)")
            _ax.set_title("Command-following distance over training (CMD-TRACK)")
            _ax.legend(fontsize=8.5)
            _ax.grid(alpha=.3)
            _fig.tight_layout()
            _fig.savefig(os.path.join(execution_dir, "cmd_track_distance.png"), dpi=120)
            _plt.close(_fig)
            print(f"[CMD-TRACK] 距離履歴を保存: {_track_path} (+ cmd_track_distance.png)")
    except Exception as _e:
        print(f"[CMD-TRACK] 保存/プロット失敗: {_e}")

    # 学習完了後の総括
    if DEBUG:
        print("\n" + "="*60)
        print("学習完了 - 総括")
        print("="*60)
        
        actual_iterations = len(training_history['iterations'])
        print(f"設定イテレーション数: {N_ITERATIONS}")
        print(f"実際の実行イテレーション数: {actual_iterations}")
        print(f"最終損失: {training_history['losses'][-1]:.4f}")
        
        # 早期終了機能は現在実装されていないため、常にFalse
        early_stop_triggered = False
        best_loss = None
        
        if early_stop_triggered:
            print(f"✓ 早期終了により学習時間を短縮しました")
            print(f"  最良損失: {best_loss:.4f}")
            print(f"  節約されたイテレーション数: {N_ITERATIONS - actual_iterations}")
        else:
            print(f"○ 全イテレーションを実行しました")
        
        # パレートフロントの進化を表示
        valid_pf_sizes = [size for size in training_history['pareto_front_sizes'] if size is not None]
        if valid_pf_sizes:
            print(f"\nパレートフロントサイズの進化:")
            for i, size in enumerate(valid_pf_sizes):
                iteration_num = (i + 1) * EVAL_INTERVAL
                if iteration_num <= actual_iterations:
                    print(f"  イテレーション {iteration_num}: {size}個の非支配解")
            
                    # 改善の統計
        if len(valid_pf_sizes) > 1:
            initial_pf_size = valid_pf_sizes[0]
            final_pf_size = valid_pf_sizes[-1]
            total_improvement = final_pf_size - initial_pf_size
            max_pf_size = max(valid_pf_sizes)
            
            print(f"\n改善効果の統計:")
            print(f"  初期パレートフロントサイズ: {initial_pf_size}")
            print(f"  最終パレートフロントサイズ: {final_pf_size}")
            print(f"  最大パレートフロントサイズ: {max_pf_size}")
            print(f"  総改善数: {total_improvement:+d}")
            
            if total_improvement > 0:
                print(f"✓ PCNエージェントの改善メカニズムにより、パレートフロントが{total_improvement}個改善されました")
            elif total_improvement == 0:
                print("○ パレートフロントサイズは維持されました")
            else:
                print(f"△ パレートフロントサイズが{abs(total_improvement)}個減少しました")
        
        # 初期パレートフロントとの比較
        if training_history['initial_pareto_front'] is not None:
            initial_pf = training_history['initial_pareto_front']
            initial_reward_count = len(initial_pf['non_dominated_inds_reward'])
            initial_values_count = len(initial_pf['non_dominated_inds_values'])
            final_reward_count = len(non_dominated_inds) if 'non_dominated_inds' in locals() else 0
            final_values_count = len(non_dominated_inds_values) if 'non_dominated_inds_values' in locals() else 0
            
            print(f"\n=== ランダム行動からの改善効果 ===")
            print(f"初期（ランダム行動後）:")
            print(f"  報酬空間非支配解数: {initial_reward_count}")
            print(f"  実数値空間非支配解数: {initial_values_count}")
            print(f"最終（学習完了後）:")
            print(f"  報酬空間非支配解数: {final_reward_count}")
            print(f"  実数値空間非支配解数: {final_values_count}")
            
            reward_improvement = final_reward_count - initial_reward_count
            values_improvement = final_values_count - initial_values_count
            
            print(f"\n改善効果:")
            print(f"  報酬空間: {reward_improvement:+d} ({initial_reward_count} → {final_reward_count})")
            print(f"  実数値空間: {values_improvement:+d} ({initial_values_count} → {final_values_count})")
            
            if reward_improvement > 0 or values_improvement > 0:
                print("✓ ランダム行動から学習により改善されました")
            elif reward_improvement == 0 and values_improvement == 0:
                print("○ ランダム行動と同等の性能を維持しました")
            else:
                print("△ ランダム行動から性能が低下しました")
        
        # 損失の改善統計
        if len(training_history['losses']) > 1:
            initial_loss = training_history['losses'][0]
            final_loss = training_history['losses'][-1]
            loss_improvement = initial_loss - final_loss
            
            print(f"\n損失の改善統計:")
            print(f"  初期損失: {initial_loss:.4f}")
            print(f"  最終損失: {final_loss:.4f}")
            print(f"  損失改善: {loss_improvement:.4f}")
            
            if loss_improvement > 0:
                print("✓ 損失が改善され、モデルの学習が進みました")
            else:
                print("△ 損失の改善が見られませんでした")
        
        # Distanceの改善統計
        valid_distances = [dist for dist in training_history['distances'] if dist is not None and len(dist) > 0]
        if len(valid_distances) > 1:
            initial_avg_distance = np.mean(valid_distances[0])
            final_avg_distance = np.mean(valid_distances[-1])
            distance_improvement = initial_avg_distance - final_avg_distance
            
            print(f"\nDistanceの改善統計:")
            print(f"  初期平均Distance: {initial_avg_distance:.4f}")
            print(f"  最終平均Distance: {final_avg_distance:.4f}")
            print(f"  Distance改善: {distance_improvement:.4f}")
            
            if distance_improvement > 0:
                print("✓ Distanceが改善され、目標達成精度が向上しました")
            else:
                print("△ Distanceの改善が見られませんでした")
        
        print(f"\n=== 改良された経験の実現結果 ===")
        print("PCNエージェントの_choose_commandsと_nlargestメソッドにより:")
        print("- 非支配解を優先的に選択")
        print("- パレートフロントの多様性を維持")
        print("- 既存解を少しずつ改善する方向を探索")
        print("- 継続的なパレートフロントの改善を実現")
    
    # パレートフロントの可視化と保存（可視化が有効な場合のみ）
    if ENABLE_VISUALIZATION:
        if DEBUG:
            print("\n=== パレートフロントの可視化 ===")
        try:
            import matplotlib.pyplot as plt
            import os
            
            # フォントエラーの対処（グローバルインポートを使用）
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
            # 最終評価（直近の evaluate を再利用可能）
            last_eval = training_history.get('last_eval')
            if _SKIP_FINAL_EVAL and last_eval and last_eval.get('e_returns') and last_eval.get('e_values'):
                e_returns = last_eval['e_returns']
                e_values = last_eval['e_values']
                distances = []
                map_fin = None
                print(
                    f"[FINAL] 最終評価をスキップし iter {last_eval.get('iteration')} の結果を再利用 "
                    f"(DISTRIBUTED_PCN_SKIP_FINAL_EVAL=1)"
                )
            else:
                if DEBUG:
                    print("最終パレートフロントを取得中...")
                e_returns, e_values, distances, map_fin = ray.get(
                    learner.evaluate.remote(n=EVAL_SAMPLES_FINAL)
                )
            
            # 最終モデルを保存
            save_dir = f"{execution_dir}/final"
            os.makedirs(save_dir, exist_ok=True)
            final_model_path = f"{save_dir}/final_model.pth"
            saved_final_model_path = ray.get(learner.save_model.remote(final_model_path))
            if saved_final_model_path and DEBUG:
                print(f"最終モデルを保存しました: {saved_final_model_path}")
            
            if len(e_returns) > 0 and len(e_values) > 0:
                # 保存ディレクトリの作成
                save_dir = f"{execution_dir}/final"
                os.makedirs(save_dir, exist_ok=True)
                
                # タイムスタンプを取得
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 1. 報酬空間でのパレートフロント（最大化目的）
                plt.figure(figsize=(12, 8))
                
                all_returns = np.array(e_returns)
                all_returns_vis = _dedupe_points_for_plot(all_returns)
                plt.scatter(
                    all_returns_vis[:, 0], all_returns_vis[:, 1],
                    c='lightblue', alpha=0.6, label='All Solutions', s=50,
                )
                
                if training_history['initial_axis_ranges'] and 'rewards' in training_history['initial_axis_ranges']:
                    plt.xlim(training_history['initial_axis_ranges']['rewards']['x_min'], training_history['initial_axis_ranges']['rewards']['x_max'])
                    plt.ylim(training_history['initial_axis_ranges']['rewards']['y_min'], training_history['initial_axis_ranges']['rewards']['y_max'])
                
                non_dominated_inds = get_non_dominated_inds(all_returns_vis)
                pareto_front_returns = all_returns_vis[non_dominated_inds]
                plt.scatter(
                    pareto_front_returns[:, 0], pareto_front_returns[:, 1],
                    c='red', s=100, label='Current Pareto Front', zorder=5,
                )
                
                # パレートフロントの線を描画
                if len(pareto_front_returns) > 1:
                    # パレートフロントをソート
                    sorted_indices = np.lexsort((pareto_front_returns[:, 1], pareto_front_returns[:, 0]))
                    sorted_pareto = pareto_front_returns[sorted_indices]
                    plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
                
                plt.title(
                    f'Pareto Front (Reward Space) - End of Training\n'
                    f'Non-dominated Solutions: {len(non_dominated_inds)} (unique points)',
                    fontsize=14,
                )
                plt.xlabel('Reward 1 (Maximize)', fontsize=12)
                plt.ylabel('Reward 2 (Maximize)', fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                
                # 最適方向の矢印
                plt.annotate('Optimal Direction', xy=(plt.xlim()[1]*0.8, plt.ylim()[1]*0.8), 
                            xytext=(plt.xlim()[1]*0.6, plt.ylim()[1]*0.6),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                            fontsize=12)
                
                plt.tight_layout()
                
                # 保存
                reward_plot_path = f"{save_dir}/pareto_front_rewards_{timestamp}.png"
                plt.savefig(reward_plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                if DEBUG:
                    print(f"報酬空間のパレートフロントを保存: {reward_plot_path}")
                
                # 2. 実数値空間でのパレートフロント（最小化目的）
                plt.figure(figsize=(12, 8))
                
                all_values = np.array(e_values)
                all_values_vis = _dedupe_points_for_plot(all_values)
                plt.scatter(
                    all_values_vis[:, 0], all_values_vis[:, 1],
                    c='lightgreen', alpha=0.6, label='All Solutions', s=50,
                )
                
                if training_history['initial_axis_ranges'] and 'values' in training_history['initial_axis_ranges']:
                    plt.xlim(training_history['initial_axis_ranges']['values']['x_min'], training_history['initial_axis_ranges']['values']['x_max'])
                    plt.ylim(training_history['initial_axis_ranges']['values']['y_min'], training_history['initial_axis_ranges']['values']['y_max'])
                
                non_dominated_inds_values = get_non_dominated_inds_minimize(all_values_vis)
                pareto_front_values = all_values_vis[non_dominated_inds_values]
                plt.scatter(
                    pareto_front_values[:, 0], pareto_front_values[:, 1],
                    c='red', s=100, label='Current Pareto Front', zorder=5,
                )
                
                # パレートフロントの線を描画
                if len(pareto_front_values) > 1:
                    # パレートフロントをソート
                    sorted_indices = np.lexsort((pareto_front_values[:, 1], pareto_front_values[:, 0]))
                    sorted_pareto = pareto_front_values[sorted_indices]
                    plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
                
                plt.title(
                    f'Pareto Front (Value Space) - End of Training\n'
                    f'Non-dominated Solutions: {len(non_dominated_inds_values)} (unique points)',
                    fontsize=14,
                )
                plt.xlabel('Cost (Minimize)', fontsize=12)
                plt.ylabel('Execution Time (Minimize)', fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                
                # 最適方向の矢印
                plt.annotate('Optimal Direction', xy=(plt.xlim()[0]*0.8, plt.ylim()[0]*0.8), 
                            xytext=(plt.xlim()[0]*0.6, plt.ylim()[0]*0.6),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                            fontsize=12)
                
                plt.tight_layout()
                
                # 保存
                values_plot_path = f"{save_dir}/pareto_front_values_{timestamp}.png"
                plt.savefig(values_plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                if DEBUG:
                    print(f"実数値空間のパレートフロントを保存: {values_plot_path}")
                
                # 3. 学習履歴の可視化
                plt.figure(figsize=(15, 12))
                
                # サブプロット1: 損失の推移
                plt.subplot(3, 2, 1)
                plt.plot(training_history['iterations'], training_history['losses'], 'b-', linewidth=2)
                plt.title('Training Loss Progression', fontsize=12)
                plt.xlabel('Iteration', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # サブプロット2: パレートフロントサイズの推移
                plt.subplot(3, 2, 2)
                valid_pf_sizes = [size for size in training_history['pareto_front_sizes'] if size is not None]
                valid_iterations = [(i+1)*EVAL_INTERVAL for i, size in enumerate(valid_pf_sizes)]
                plt.plot(valid_iterations, valid_pf_sizes, 'r-', linewidth=2, marker='o')
                plt.title('Pareto Front Size Progression', fontsize=12)
                plt.xlabel('Iteration', fontsize=10)
                plt.ylabel('Number of Non-dominated Solutions', fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # サブプロット3: Distance統計の推移
                plt.subplot(3, 2, 3)
                valid_distances = [dist for dist in training_history['distances'] if dist is not None and len(dist) > 0]
                valid_distance_iterations = [(i+1)*EVAL_INTERVAL for i, dist in enumerate(valid_distances)]
                if valid_distance_iterations:
                    avg_distances = [np.mean(dist) for dist in valid_distances]
                    min_distances = [np.min(dist) for dist in valid_distances]
                    max_distances = [np.max(dist) for dist in valid_distances]
                    
                    plt.plot(valid_distance_iterations, avg_distances, 'g-', linewidth=2, marker='o', label='Average Distance')
                    plt.fill_between(valid_distance_iterations, min_distances, max_distances, alpha=0.3, color='green', label='Min-Max Range')
                    plt.legend(fontsize=10)
                
                plt.title('Distance Statistics Progression', fontsize=12)
                plt.xlabel('Iteration', fontsize=10)
                plt.ylabel('Distance', fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # サブプロット4: Distance分布のヒストグラム
                plt.subplot(3, 2, 4)
                all_distances = []
                for dist_list in valid_distances:
                    all_distances.extend(dist_list)
                
                if all_distances:
                    plt.hist(all_distances, bins=20, alpha=0.7, color='green', edgecolor='black')
                    plt.axvline(np.mean(all_distances), color='red', linestyle='--', label=f'Mean: {np.mean(all_distances):.3f}')
                    plt.legend(fontsize=9)
                
                plt.title('Distance Distribution', fontsize=12)
                plt.xlabel('Distance', fontsize=10)
                plt.ylabel('Frequency', fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # サブプロット5: 統計情報
                plt.subplot(3, 2, 5)
                plt.axis('off')
                
                # loss_improvementとtotal_improvementの計算
                loss_improvement = 0.0
                total_improvement = 0
                distance_improvement = 0.0
                
                if len(training_history['losses']) > 1:
                    initial_loss = training_history['losses'][0]
                    final_loss = training_history['losses'][-1]
                    loss_improvement = initial_loss - final_loss
                
                if len(valid_pf_sizes) > 1:
                    initial_pf_size = valid_pf_sizes[0]
                    final_pf_size = valid_pf_sizes[-1]
                    total_improvement = final_pf_size - initial_pf_size
                
                if len(valid_distances) > 1:
                    initial_avg_distance = np.mean(valid_distances[0])
                    final_avg_distance = np.mean(valid_distances[-1])
                    distance_improvement = initial_avg_distance - final_avg_distance
                
                stats_text = f"""
Training Statistics:
• Total Iterations: {N_ITERATIONS}
• Final Loss: {training_history['losses'][-1]:.4f}
• Final Pareto Front Size: {valid_pf_sizes[-1] if valid_pf_sizes else 0}
• Final Avg Distance: {np.mean(valid_distances[-1]) if valid_distances else 0:.4f}
• Loss Improvement: {loss_improvement:.4f}
• Pareto Front Improvement: {total_improvement}
• Distance Improvement: {distance_improvement:.4f}
                """
                plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', transform=plt.gca().transAxes)
                
                plt.tight_layout()
                
                # 保存
                history_plot_path = f"{save_dir}/learning_history_{timestamp}.png"
                plt.savefig(history_plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                if DEBUG:
                    print(f"学習履歴を保存: {history_plot_path}")
                
                # 4. パレートフロントの詳細データをテキストファイルに保存
                details_path = f"{save_dir}/pareto_front_details_{timestamp}.txt"
                with open(details_path, 'w', encoding='utf-8') as f:
                    f.write("=== 学習完了時のパレートフロント詳細 ===\n")
                    f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"総イテレーション数: {N_ITERATIONS}\n")
                    f.write(f"最終損失: {training_history['losses'][-1]:.4f}\n")
                    
                    # 初期パレートフロントの軸範囲情報を追加
                    if training_history['initial_axis_ranges'] is not None:
                        f.write("\n=== 初期パレートフロント（ランダム行動後）の軸範囲 ===\n")
                        initial_axis = training_history['initial_axis_ranges']
                        if 'rewards' in initial_axis:
                            f.write(f"報酬空間軸範囲: X[{initial_axis['rewards']['x_min']:.4f}, {initial_axis['rewards']['x_max']:.4f}], Y[{initial_axis['rewards']['y_min']:.4f}, {initial_axis['rewards']['y_max']:.4f}]\n")
                        if 'values' in initial_axis:
                            f.write(f"実数値空間軸範囲: X[{initial_axis['values']['x_min']:.4f}, {initial_axis['values']['x_max']:.4f}], Y[{initial_axis['values']['y_min']:.4f}, {initial_axis['values']['y_max']:.4f}]\n")
                        f.write("\n")
                    
                    f.write("=== 最終報酬空間のパレートフロント ===\n")
                    f.write(f"非支配解数: {len(non_dominated_inds)}\n")
                    for i, idx in enumerate(non_dominated_inds):
                        f.write(f"解{i+1}: {e_returns[idx]}\n")
                    f.write("\n")
                    
                    f.write("=== 最終実数値空間のパレートフロント ===\n")
                    f.write(f"非支配解数: {len(non_dominated_inds_values)}\n")
                    for i, idx in enumerate(non_dominated_inds_values):
                        f.write(f"解{i+1}: {e_values[idx]}\n")
                    f.write("\n")
                    
                    # 改善効果の統計を追加（軸範囲の比較）
                    if training_history['initial_axis_ranges'] is not None:
                        f.write("=== 軸範囲の比較 ===\n")
                        initial_axis = training_history['initial_axis_ranges']
                        
                        f.write("\n")
                    
                    f.write("=== 学習履歴 ===\n")
                    f.write("イテレーション, 損失, パレートフロントサイズ, 平均Distance, 最小Distance, 最大Distance\n")
                    for i, pf_size in enumerate(valid_pf_sizes):
                        if i < len(valid_distances) and len(valid_distances[i]) > 0:
                            avg_dist = np.mean(valid_distances[i])
                            min_dist = np.min(valid_distances[i])
                            max_dist = np.max(valid_distances[i])
                            f.write(f"{i+1}, {training_history['losses'][i]:.4f}, {pf_size}, {avg_dist:.4f}, {min_dist:.4f}, {max_dist:.4f}\n")
                        else:
                            f.write(f"{i+1}, {training_history['losses'][i]:.4f}, {pf_size}, N/A, N/A, N/A\n")
                    
                if DEBUG:
                    print(f"パレートフロント詳細を保存: {details_path}")
                    
                    print(f"\n=== 可視化完了 ===")
                    print(f"全てのファイルは実行ディレクトリ '{execution_dir}' に保存されました")
                    print(f"• 初期パレートフロント: {execution_dir}/")
                    print(f"• 反復可視化とモデル: {execution_dir}/iteration_XXX/")
                    print(f"• 最終可視化とモデル: {execution_dir}/final/")
                    print(f"• 報酬空間パレートフロント: pareto_front_rewards_{timestamp}.png")
                    print(f"• 実数値空間パレートフロント: pareto_front_values_{timestamp}.png")
                    print(f"• 学習履歴: learning_history_{timestamp}.png")
                    print(f"• 詳細データ: pareto_front_details_{timestamp}.txt")
                    print(f"• 最終モデル: final_model.pth")
                
            else:
                if DEBUG:
                    print("警告: パレートフロントのデータが取得できませんでした。")
                
        except Exception as e:
            print(f"可視化中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        _snap_path = os.path.join(execution_dir, "learner_replay_snapshot.pkl.gz")
        ray.get(learner.save_replay_snapshot.remote(_snap_path))
    except Exception as _e:
        print(f"[Learner] replay スナップショット保存失敗: {_e}")

    _mo_hv_path = os.environ.get("DISTRIBUTED_PCN_MO_HV_EXPORT")
    if not _mo_hv_path:
        _mo_hv_path = os.path.join(execution_dir, "pcn_mo_hv.json")
    if _mo_hv_path:
        try:
            import json as _json
            _mo_hv_path = os.path.abspath(_mo_hv_path)
            os.makedirs(os.path.dirname(_mo_hv_path) or ".", exist_ok=True)
            _data = ray.get(learner.export_mo_hv_data.remote())
            _data["wall_total_s"] = float(time.perf_counter() - _main_wall_t0)
            with open(_mo_hv_path, "w", encoding="utf-8") as _mf:
                _json.dump(_data, _mf, indent=2, allow_nan=False)
            print(f"[MO_HV] 評価トレースを書き出しました: {_mo_hv_path}")
        except Exception as _e:
            print(f"[MO_HV] 書き出し失敗: {_e}")

    # 全体の完了時間を記録
    if TIME_DEBUG:
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        print(f"\n{'='*60}")
        print("分散PCN学習完了")
        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総経過時間: {overall_duration:.2f}秒 ({overall_duration/60:.2f}分)")
        print(f"実行ディレクトリ: {execution_dir}")
        print(f"{'='*60}")
    
    # 実行完了メッセージ
    print(f"\n{'='*60}")
    print("学習完了！")
    print(f"全ての結果は実行ディレクトリ '{execution_dir}' に保存されました")
    print(f"• 初期パレートフロント: {execution_dir}/")
    print(f"• 反復可視化とモデル: {execution_dir}/iteration_XXX/")
    print(f"• 最終可視化とモデル: {execution_dir}/final/")
    print(f"• モデルファイル: model_iter_XXX.pth, final_model.pth")
    print(f"{'='*60}")
    
    # 各フェーズの時間割合を表示
    if TIME_DEBUG:
        print(f"\n各フェーズの時間割合:")
        print(f"フェーズ1 (初期エピソード収集): {phase1_duration:.2f}秒 ({phase1_duration/overall_duration*100:.1f}%)")
        print(f"フェーズ2 (教師あり学習): {phase2_duration:.2f}秒 ({phase2_duration/overall_duration*100:.1f}%)")
        print(f"フェーズ3 (改良された経験の実現): {phase3_duration:.2f}秒 ({phase3_duration/overall_duration*100:.1f}%)")
        print(f"{'='*60}")
    
    if DEBUG:
        print("\n学習が完了しました")

if __name__ == "__main__":
    main()
