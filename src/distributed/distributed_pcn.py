import numpy as np
import torch as th
import yaml
from tqdm import tqdm
import time
import torch.nn.functional as F
import heapq
import os
from typing import List, Tuple, Optional, Union

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
N_JOBS = 256 # ジョブ数

EVAL_INTERVAL = 5  # 評価を実行する間隔（イテレーション数）
USE_DISTRIBUTED_EVAL = False  # 分散評価を使用するかどうか

BATCH_SIZE = 2048  # 実験規模を変えない（GPU 時も同じ）
N_UPDATES = 5  # 学習更新回数（3に減らすと高速化、精度はやや低下の可能性）
LEARNING_RATE = 1e-2

EARLY_STOPPING_PATIENCE = 5  # 改善が見られないイテレーション数
EARLY_STOPPING_THRESHOLD = 0.0001  # 改善とみなす最小変化量
MIN_ITERATIONS = 5  # 最低限実行するイテレーション数


INITIAL_EPISODES = 100  # フェーズ1: 各Actorあたりのランダム収集エピソード数（config.yml で上書き可）

USE_ENHANCED_MODEL = False  # True: EnhancedPCNModel, False: DiscreteActionsDefaultModel (3層NLPモデル)


SUPERVISED_LEARNING_EPOCHS = 30
SUPERVISED_BATCH_SIZE = 1024    
SUPERVISED_UPDATES_PER_EPOCH = 3 
SUPERVISED_LEARNING_RATE = 1e-2  

VISUALIZATION_INTERVAL = 5  # 可視化を実行する間隔（イテレーション数）

EPISODES_PER_ITERATION = 10  # 各イテレーションで各Actorが生成するエピソード数

EVAL_SAMPLES = 100  # 評価時に使用するサンプル数
EVAL_SAMPLES_DISTRIBUTED = 10  # 分散評価時に使用するサンプル数
EVAL_SAMPLES_FINAL = 100  # 最終評価時に使用するサンプル数
EVAL_SAMPLES_VISUALIZATION = 100  # 反復可視化用（少なめで高速化、パレートフロントは十分描ける）

# プロファイリング用: 環境変数で短時間実行モードを有効化
_PROFILE_MODE = os.environ.get('DISTRIBUTED_PCN_PROFILE', '0') == '1'
_QUICK_MODE = os.environ.get('DISTRIBUTED_PCN_QUICK', '0') == '1'
# Actor-Learner非同期オーバーラップ（Learner(i)とActor(i+1)を並列実行して待ち時間を隠蔽）
_ASYNC_OVERLAP = os.environ.get('DISTRIBUTED_PCN_ASYNC_OVERLAP', '1') == '1'
# 高速化モード: N_UPDATESを3に削減（本番でも有効、DISTRIBUTED_PCN_FAST=1）
_FAST_MODE = os.environ.get('DISTRIBUTED_PCN_FAST', '0') == '1'
_USE_JAX_LEARNER = os.environ.get('DISTRIBUTED_PCN_USE_JAX', '0') == '1'
# 既定: イベント観測（環境はイベント駆動）+ ラーナー側でビットマップ復元してNN入力
# 従来のCビットマップ観測のみに戻す: DISTRIBUTED_PCN_USE_EVENT_OBS=0
_USE_EVENT_OBS = os.environ.get('DISTRIBUTED_PCN_USE_EVENT_OBS', '1') == '1'
if _QUICK_MODE:
    # 短時間デバッグ用（本番実験規約の 100/100 ではない）
    N_ITERATIONS = 5
    N_ACTORS = 12
    INITIAL_EPISODES = 100
    EPISODES_PER_ITERATION = 1
    EVAL_INTERVAL = 5
    SUPERVISED_LEARNING_EPOCHS = 10
    # N_UPDATESは変更しない（ハイパラメータ変更は高速化ではない）
    ENABLE_VISUALIZATION = True
    print("[PROFILE] クイックモード（デバッグ）: N_ITERATIONS=5, N_ACTORS=12, INITIAL_EPISODES=100")
elif _FAST_MODE:
    # 削除: N_UPDATES変更はハイパラメータ変更のため高速化に含めない
    pass

from src.agents.pcn_agent import (
    PCN, 
    Transition, 
    get_non_dominated_inds, 
    get_non_dominated_inds_minimize,
    crowding_distance,
    hypervolume
)
from src.envs.scheduling_env import SchedulingEnv
from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
from src.envs.scheduling_env_event_obs import SchedulingEnvEventObs
from src.utils.event_obs_bitmap_adapter import (
    learner_bitmap_enabled,
    apply_learner_bitmap_to_event_env,
)
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.algorithm_compare_config import get_param_algorithm_compare

# 使用する環境クラス（イベント観測モード時はSchedulingEnvEventObs）
_EnvClass = SchedulingEnvEventObs if _USE_EVENT_OBS else SchedulingEnvCacheOptimized

# イベント観測環境使用時、NN入力をラーナー側でビットマップへ復元（既定ON、SCHEDULER_LEARNER_BITMAP / DISTRIBUTED_PCN_EVENT_TO_BITMAP）
if _USE_EVENT_OBS:
    if learner_bitmap_enabled():
        print("[ENV] イベント観測（環境）+ ラーナー側ビットマップ復元（NN入力）")
    else:
        print("[ENV] イベントベース観測（NNはイベントベクトルのみ、ビットマップ復元OFF）")


def _enable_event_bitmap_adapter(env):
    """イベント観測環境の get_observation をビットマップ復元版へ差し替える（共通ユーティリティ）。"""
    if not _USE_EVENT_OBS:
        return env
    return apply_learner_bitmap_to_event_env(env)

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
                
                optimized_episode.append(optimized_transition)
            result.append(optimized_episode)
        
        self.buffer.clear()
        self.episode_hashes.clear()  # ハッシュセットもクリア
        self._hash_cache.clear()  # ハッシュキャッシュもクリア
        if DEBUG:
            print(f"ReplayBuffer: retrieved all {len(result)} episodes and cleared buffer")
        return result

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
            self.env = _enable_event_bitmap_adapter(self.env)
            # C実装が正しく使用されているか確認
            # if hasattr(self.env, '_cache_onpre_c'):
            #     print(f"[Actor {self.actor_id}] ✓ C実装環境が正しく初期化されました")
            # else:
            #     print(f"[Actor {self.actor_id}] ⚠️ C実装環境の初期化に問題があります")
            
            # Actorは常にCPUで実行（Learnerとの互換性のため）
            actual_device = 'cpu'
            
            # PCNエージェントの初期化（CPUで実行）
            self.agent = PCN(
                self.env,
                device=actual_device,
                state_dim=self.env.observation_space.shape[0],
                scaling_factor=np.array([1, 1, 1]),
                learning_rate=LEARNING_RATE,
                batch_size=512,
                hidden_dim=512,
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
        return self.env

    def run(self, n_episodes=10, random_actions=False, pre_fetched_commands=None):
        """pre_fetched_commands: list of (desired_return, desired_horizon), length n_actors*n_episodes.
        指定時は_choose_commandsのリモート呼び出しをスキップ（Learner負荷削減）。"""
        if self.env is None:
            self._make_env()
        
        episodes_generated = 0
        collected_episodes = []  # 収集したエピソードを一時保存
        solution_execution_times = []  # 改良された解の実行時間を記録
        
        progress_interval = max(1, n_episodes // 10)
        
        if not random_actions:
            weights = ray.get(self.learner.get_weights.remote())
            self.agent.model.load_state_dict(weights)
        
        for ep in range(n_episodes):
            try:
                cmd = None
                if pre_fetched_commands is not None:
                    idx = self.actor_id * n_episodes + ep
                    if idx < len(pre_fetched_commands):
                        cmd = pre_fetched_commands[idx]
                episode = self._run_episode(random_actions, pre_fetched_command=cmd)
                # print("done")
                
                # エピソードを一時保存
                collected_episodes.append(episode)
                episodes_generated += 1
                
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
            ray.get(self.buffer.add_batch.remote(collected_episodes))
        
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
        return episodes_generated

    def _run_episode(self, random_actions=False, pre_fetched_command=None):
        """pre_fetched_command: (desired_return, desired_horizon) が指定されていれば_choose_commandsをスキップ"""
        obs = self.env.reset()
        done = False
        transitions = []
        
        solution_selection_start_time = None
        solution_execution_time = None
        
        if not random_actions:
            solution_selection_start_time = time.time()
            if pre_fetched_command is not None:
                desired_return, desired_horizon = pre_fetched_command
            else:
                t_choose_start = time.time()
                desired_return, desired_horizon = ray.get(self.learner._choose_commands.remote(50))
                if _PROFILE_MODE:
                    print(f"[PROFILE Actor {self.actor_id}] _choose_commands: {time.time()-t_choose_start:.3f}s")
            self.agent.set_desired_return_and_horizon(desired_return, desired_horizon)
            
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
                action = self.agent.eval(obs)
                
            n_obs, reward, scheduled, wt_step, done = self.env.step(action)
            step_count += 1
            
            # 観測データをfloat32に変換してメモリ使用量を削減（シリアライゼーション最適化）
            # 既にfloat32の場合は変換しない（メモリコピーを避ける）
            if hasattr(obs, 'dtype') and obs.dtype != np.float32:
                obs = np.array(obs, dtype=np.float32, copy=True)
            if hasattr(n_obs, 'dtype') and n_obs.dtype != np.float32:
                n_obs = np.array(n_obs, dtype=np.float32, copy=True)
            transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
            obs = n_obs
            
        # エピソード完了時に実数値を計算
        if done:
            t_steps = time.time() - t_steps_start
            if _PROFILE_MODE and step_count > 0:
                print(f"[PROFILE Actor {self.actor_id}] env.step loop: {t_steps:.3f}s ({step_count} steps, {t_steps/step_count*1000:.1f}ms/step)")
            self.env.finalize_window_history()
            cost,_,avg_waiting_time = self.env.calc_objective_values()
            
            solution_execution_time = time.time() - start_time
                
            # print(f"[Actor {self.actor_id}] 改良された解の実行完了")
            # print(f"  選択〜実行完了時間: {solution_execution_time:.4f}秒")
            # print(f"  最終コスト: {cost}")
            # print(f"  平均待機時間: {avg_waiting_time}")
            
            # 最初のTransitionに実数値を追加（後でアクセスできるように）
            if len(transitions) > 0:
                transitions[0].objective_values = [cost,_,avg_waiting_time]
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

    def evaluate_episode(self, desired_return, desired_horizon, max_return):
        """単一エピソードの評価を実行"""
        if self.env is None:
            self._make_env()
        
        # 重みを取得（共有された重みを使用）
        # 2回のray.get()を避けるため、get_weights()を直接呼び出す
        weights = ray.get(self.learner.get_weights.remote())
        self.agent.model.load_state_dict(weights)
        
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
        
        print(f"[Actor {self.actor_id}] 評価エピソード完了: 報酬={episode_return}, 実数値={value}")
        print(f"  評価実行時間: {evaluation_execution_time:.4f}秒")
        
        return episode_return, value, transitions, map_fin

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
        
        # PCNエージェントを正しいデバイスで初期化（GPU使用時はGPUで初期化、BATCH_SIZE使用）
        self.agent = PCN(
            self.env,
            device=self.actual_device,  # 検出したデバイスで初期化
            state_dim=self.env.observation_space.shape[0],
            scaling_factor=np.array([1, 1, 1]),
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            hidden_dim=512,
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
        self._weights_ref = None  # 重みのObjectRefを保持（重みの共有用）
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
        if DEBUG:
            print(f"Learner initialized with device: {self.actual_device}")
            print(f"Learner model: {'EnhancedPCNModel' if USE_ENHANCED_MODEL else 'DiscreteActionsDefaultModel'}")
            if self.actual_device == 'cuda':
                import torch
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

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

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        """エピソードを経験再生バッファに追加"""
        # 各Transitionのコピーを作成
        transitions_copy = []
        for t in transitions:
            # rewardのコピーを作成
            reward_copy = np.array(t.reward, copy=True)
            # 新しいTransitionオブジェクトを作成
            t_copy = Transition(
                observation=t.observation,
                action=t.action,
                reward=reward_copy,
                next_observation=t.next_observation,
                terminal=t.terminal
            )
            transitions_copy.append(t_copy)
        for i in reversed(range(len(transitions_copy) - 1)):
            transitions_copy[i].reward += self.gamma * transitions_copy[i + 1].reward

        # エピソードの内容に基づくハッシュ値を計算（重複検出用）
        episode_hash = self._compute_episode_hash(transitions_copy)
        
        # 既存のエピソードと重複していないかチェック
        if self._is_duplicate_episode(episode_hash):
            if DEBUG:
                print(f"[Learner] 重複エピソードをスキップしました。ハッシュ: {episode_hash}")
            return
        
        unique_step = (step, episode_hash)
        if len(self.agent.experience_replay) == max_size:
            heapq.heappushpop(self.agent.experience_replay, (1, unique_step, transitions_copy))
        else:
            heapq.heappush(self.agent.experience_replay, (1, unique_step, transitions_copy))
        
        # 重複検出用のハッシュセットに追加
        if not hasattr(self, '_episode_hashes'):
            self._episode_hashes = set()
        self._episode_hashes.add(episode_hash)
        
        if DEBUG:
            print(f"[Learner] エピソードを追加しました。現在のバッファサイズ: {len(self.agent.experience_replay)}")

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

    def _choose_commands(self, num_episodes: int) -> Tuple[np.ndarray, np.float32]:
        """次のエピソードの目標報酬とホライズンを選択"""
        return self.agent._choose_commands(num_episodes)

    def _choose_commands_batch(self, num_episodes: int, n_commands: int):
        """複数の異なる探索方向を一括で取得（Learner呼び出しを1回に削減、各Actorに異なる目標値）"""
        return self.agent._choose_commands_batch(num_episodes, n_commands)

    def learn(self, batch_size: int = 100, n_updates: int = 2) -> float:
        total_loss = []
        
        # ReplayBufferから全てのエピソードを取得（サンプリングせずに全部）
        # buffer.size()は不要（get_all_episodes()の戻り値が空かどうかで判定できる）
        t_get_episodes_start = time.time()
        all_episodes = ray.get(self.buffer.get_all_episodes.remote())
        t_get_episodes = time.time() - t_get_episodes_start
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
        t_add_start = time.time()
        for episode in all_episodes:
            # 追加前のバッファサイズを記録
            before_size = len(self.agent.experience_replay)
            
            # エピソードを追加
            self._add_episode(episode, max_size=10000, step=self.global_step)
            
            # 追加後のバッファサイズをチェック
            after_size = len(self.agent.experience_replay)
            if after_size > before_size:
                added_episodes += 1
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
        if DEBUG:
            print(f"[Learner] バッファ追加完了後のサイズ: {final_buffer_size}")
        
        if final_buffer_size == 0:
            print("エラー: 経験再生バッファにエピソードが追加されていません。")
            return 0.0
        elif final_buffer_size < len(all_episodes):
            print(f"警告: 取得したエピソード数 {len(all_episodes)} に対して、Learnerのバッファには {final_buffer_size} 個しか追加されていません。")
        
        # 学習更新を実行
        t_update_start = time.time()
        for i in range(n_updates):
            try:
                if getattr(self, '_use_jax', False):
                    loss_value = self._jax_update_step()
                else:
                    loss, _ = self.agent.update()
                    loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
                
                # NaNチェック
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"[Learner] 警告: 損失がNaN/Infになりました (update {i})")
                    print(f"  損失値: {loss_value}")
                    print(f"  バッファサイズ: {len(self.agent.experience_replay)}")
                    
                    # デバッグ情報を出力
                    if len(self.agent.experience_replay) > 0:
                        # サンプルエピソードを確認
                        sample_episode = self.agent.experience_replay[0][2]
                        if len(sample_episode) > 0:
                            sample_transition = sample_episode[0]
                            print(f"  サンプル観測: min={np.min(sample_transition.observation)}, max={np.max(sample_transition.observation)}")
                            print(f"  サンプル観測にNaN: {np.isnan(sample_transition.observation).any()}")
                            print(f"  サンプル観測にInf: {np.isinf(sample_transition.observation).any()}")
                            print(f"  サンプル報酬: {sample_transition.reward}")
                            print(f"  サンプル報酬にNaN: {np.isnan(sample_transition.reward).any() if hasattr(sample_transition.reward, '__iter__') else np.isnan(sample_transition.reward)}")
                    
                    # NaNの場合は0.0を記録（学習を続行）
                    loss_value = 0.0
                
                total_loss.append(loss_value)
            except Exception as e:
                print(f"[Learner] エラー: 学習更新中にエラーが発生しました (update {i}): {e}")
                import traceback
                traceback.print_exc()
                total_loss.append(0.0)  # エラーの場合は0.0を記録
            
            if DEBUG and i % 10 == 0:
                print(f"[Learner] {i} updates done. Buffer size: {ray.get(self.buffer.size.remote())}")
                if len(total_loss) > 0:
                    print(f"[Learner] Average loss: {np.mean(total_loss[-10:]):.4f}")
                if self.actual_device == 'cuda':
                    import torch
                    print(f"[Learner] GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            self.global_step += 1
        t_update = time.time() - t_update_start
        if _PROFILE_MODE and hasattr(self, '_learn_timings'):
            self._learn_timings['update'].append(t_update)
            print(f"[PROFILE Learner] get_episodes={t_get_episodes:.3f}s, add_episodes={t_add:.3f}s, update={t_update:.3f}s")
        
        # 学習後に重みのObjectRefを更新（全Actorで共有される）
        # 重みが更新された場合のみObjectRefを更新
        if total_loss and len(total_loss) > 0:
            self.update_weights_ref()
        
        return np.mean(total_loss) if total_loss else 0.0

    def evaluate(self, max_return=None, n=10, training_iteration=None, eval_diag_path=None):
        """エージェントの評価を実行"""
        if max_return is None:
            max_return = np.full(2, 100.0, dtype=np.float32)
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
            eval_diag = {"path": eval_diag_path, "training_iteration": training_iteration}
        e_returns, e_value, distances, map_fin = self.agent.evaluate(
            self.env, max_return, n=n, eval_diag=eval_diag
        )
        
        # PCNエージェントのevaluate()で既に出力されているため、
        # ここでは追加の出力処理を行わず、結果のみを返す
        return e_returns, e_value, distances, map_fin  # 実数値はPCNエージェント側で処理済み

    def evaluate_distributed(self, actors, max_return=None, n=10):
        """分散評価を実行"""
        if max_return is None:
            max_return = np.full(2, 100.0, dtype=np.float32)
        
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
        """モデルを指定パスに保存（リモートメソッド）"""
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

# =========================
# 4. ユーティリティ関数
# =========================

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
        
        # 全ての解をプロット
        plt.scatter(all_returns[:, 0], all_returns[:, 1], c='lightblue', alpha=0.6, label='All Solutions', s=50)
        
        # 非支配解を強調表示
        pareto_front_returns = all_returns[initial_non_dominated_inds]
        plt.scatter(pareto_front_returns[:, 0], pareto_front_returns[:, 1], c='red', s=100, label='Pareto Front', zorder=5)
        
        # パレートフロントの線を描画
        if len(pareto_front_returns) > 1:
            # パレートフロントをソート
            sorted_indices = np.lexsort((pareto_front_returns[:, 1], pareto_front_returns[:, 0]))
            sorted_pareto = pareto_front_returns[sorted_indices]
            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
        
        # 軸範囲を設定
        plt.xlim(axis_ranges['rewards']['x_min'], axis_ranges['rewards']['x_max'])
        plt.ylim(axis_ranges['rewards']['y_min'], axis_ranges['rewards']['y_max'])
        
        plt.title(f'Initial Random Experience - Pareto Front (Reward)\nNon-dominated: {len(initial_non_dominated_inds)}', fontsize=12)
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
        
        # 全ての解をプロット
        plt.scatter(all_values[:, 0], all_values[:, 1], c='lightgreen', alpha=0.6, label='All Solutions', s=50)
        
        # 非支配解を強調表示（最小化問題なので最小化用の関数を使用）
        pareto_front_values = all_values[initial_non_dominated_inds_values]
        plt.scatter(pareto_front_values[:, 0], pareto_front_values[:, 1], c='red', s=100, label='Pareto Front', zorder=5)
        
        # パレートフロントの線を描画
        if len(pareto_front_values) > 1:
            # パレートフロントをソート
            sorted_indices = np.lexsort((pareto_front_values[:, 1], pareto_front_values[:, 0]))
            sorted_pareto = pareto_front_values[sorted_indices]
            plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
        
        # 軸範囲を設定
        plt.xlim(axis_ranges['values']['x_min'], axis_ranges['values']['x_max'])
        plt.ylim(axis_ranges['values']['y_min'], axis_ranges['values']['y_max'])
        
        plt.title(f'Initial Random Experience - Pareto Front (Value)\nNon-dominated: {len(initial_non_dominated_inds_values)}', fontsize=12)
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
    
    # Singularity等: 作業ディレクトリを環境変数で指定可能（相対パス解決の基準）
    workdir = os.environ.get('DISTRIBUTED_PCN_WORKDIR')
    if workdir and os.path.isdir(workdir):
        os.chdir(workdir)
        if DEBUG:
            print(f"[DISTRIBUTED_PCN] 作業ディレクトリ: {os.getcwd()}")
    
    # 実行用のディレクトリを作成
    # 環境変数 DISTRIBUTED_PCN_OUTPUT_DIR で出力先を指定可能（Singularity等でマウント先を指定）
    output_base = os.environ.get('DISTRIBUTED_PCN_OUTPUT_DIR', '.')
    execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_dir = os.path.join(output_base, f"execution_{execution_timestamp}")
    os.makedirs(execution_dir, exist_ok=True)
    EVAL_DIAG = os.environ.get("DISTRIBUTED_PCN_EVAL_DIAG", "0") == "1"
    eval_diag_path = os.path.join(execution_dir, "pcn_eval_diag.jsonl") if EVAL_DIAG else None
    if EVAL_DIAG:
        print(f"[EVAL_DIAG] 各評価の統計を追記: {eval_diag_path}")
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
    global N_ITERATIONS, N_ACTORS, INITIAL_EPISODES, EPISODES_PER_ITERATION, EVAL_INTERVAL, SUPERVISED_LEARNING_EPOCHS
    _dpc = get_param_algorithm_compare(config).get("distributed_pcn") or {}
    N_ITERATIONS = int(_dpc.get("n_iterations", N_ITERATIONS))
    N_ACTORS = int(_dpc.get("n_actors", N_ACTORS))
    INITIAL_EPISODES = int(_dpc.get("initial_episodes", INITIAL_EPISODES))
    if _dpc.get("quick") is True:
        os.environ["DISTRIBUTED_PCN_QUICK"] = "1"
    elif _dpc.get("quick") is False:
        os.environ["DISTRIBUTED_PCN_QUICK"] = "0"
    if _dpc.get("profile") is True:
        os.environ["DISTRIBUTED_PCN_PROFILE"] = "1"
    elif _dpc.get("profile") is False:
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
    
    # 各Actorで初期エピソードを実行（ランダム行動）
    # 全ての初期化を並列で待つ（for文を避けて並列化を最大化）
    try:
        ray.get(init_futures)  # 全ての初期化を並列で待つ
    except Exception as e:
        print(f"一部のActorの初期化でエラーが発生: {e}")
    
    # 全てのエピソード生成を並列で開始（for文を避けて並列化を最大化）
    simulation_futures = [
        actor.run.remote(n_episodes=INITIAL_EPISODES, random_actions=True)
        for actor in actors
    ]
    
    # 全てのエピソード生成を並列で待つ
    total_episodes = 0
    completed_actors = 0
    try:
        results = ray.get(simulation_futures)  # 全ての結果を並列で取得
        for i, episodes_generated in enumerate(results):
            total_episodes += episodes_generated
            completed_actors += 1
            # 進捗を表示
            progress_percentage = (completed_actors / N_ACTORS) * 100
            if DEBUG:
                print(f"Actor {i} の初期エピソード生成完了: {episodes_generated} エピソード (進捗: {progress_percentage:.1f}%)")
    except Exception as e:
        print(f"一部のActorのエピソード生成でエラーが発生: {e}")

    if DEBUG:
        print(f"合計生成エピソード数: {total_episodes}")
        print("=== 初期経験収集完了 ===")
    
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
    
    for epoch in range(SUPERVISED_LEARNING_EPOCHS):
        if DEBUG:
            print(f"\n--- 教師あり学習エポック {epoch + 1}/{SUPERVISED_LEARNING_EPOCHS} ---")
        
        # 初期エピソードを使用して学習
        epoch_losses = []
        
        for update in range(SUPERVISED_UPDATES_PER_EPOCH):
            # 初期エピソードは既にLearnerの経験再生バッファに追加されているため、
            # PCNエージェントのupdateメソッドを直接呼び出して学習（教師あり学習用の学習率を使用）
            loss, _ = ray.get(learner.update.remote(SUPERVISED_LEARNING_RATE))
            epoch_losses.append(loss.item())
            
            if DEBUG and update % 2 == 0:
                print(f"  更新 {update + 1}/{SUPERVISED_UPDATES_PER_EPOCH}: 損失 = {loss.item():.4f}")
        
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        
        # 学習履歴を記録
        supervised_training_history['epochs'].append(epoch + 1)
        supervised_training_history['losses'].append(avg_epoch_loss)
        
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

    # =========================
    # フェーズ3: 改良された経験の実現
    # =========================
    if DEBUG or TIME_DEBUG:
        print("\n" + "="*60)
        print("フェーズ3: 改良された経験の実現")
        print("="*60)
    
    # フェーズ3の開始時間を記録
    if TIME_DEBUG:
        phase3_start_time = time.time()
        print(f"フェーズ3開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 学習履歴を記録
    training_history = {
        'iterations': [],
        'losses': [],
        'pareto_front_sizes': [],
        'distances': [],  # Distanceを記録
        'initial_axis_ranges': initial_axis_ranges,  # 初期パレートフロントの軸範囲を保存
    }


    
    # 学習ループ（改良された経験の生成）
    # 非同期オーバーラップ: Learner(i)とActor(i+1)を並列実行して待ち時間を隠蔽
    if _ASYNC_OVERLAP and _PROFILE_MODE:
        print("[PROFILE] Actor-Learner非同期オーバーラップ有効")
    
    learner_future = None
    next_actor_futures = None
    n_commands_per_iter = N_ACTORS * EPISODES_PER_ITERATION
    
    for iteration in range(N_ITERATIONS):
        if _ASYNC_OVERLAP and N_ITERATIONS > 1:
            # 非同期オーバーラップモード
            if iteration == 0:
                if DEBUG:
                    print("Actorが改良されたエピソードを生成中...")
                    print("※ PCNエージェントの_choose_commandsと_nlargestメソッドにより改善された目標値を使用")
                # 一括で探索方向を取得（12回のリモート呼び出し→1回に削減）
                commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                t_actor_start = time.time()
                actor_results = ray.get(actor_futures)
                t_actor = time.time() - t_actor_start
                if _PROFILE_MODE:
                    print(f"[PROFILE Iter {iteration+1}] Actor実行: {t_actor:.3f}s (合計{sum(actor_results)}エピソード)")
                
                if DEBUG:
                    print("Learnerが改良された経験で学習を実行中")
                t_learner_start = time.time()
                commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                learner_future = learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES)
                next_actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                t_wait_start = time.time()
                loss = ray.get(learner_future)
                ray.get(next_actor_futures)  # Actor(1)完了待機（actor_resultsはActor(0)のまま）
                t_wait = time.time() - t_wait_start
                t_learner = time.time() - t_learner_start
                if _PROFILE_MODE:
                    print(f"[PROFILE Iter {iteration+1}] Learner+Actor(次)並列待機: {t_wait:.3f}s (Learner: {t_learner:.3f}s)")
                if N_ITERATIONS > 1:
                    learner_future = learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES)
                    commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                    next_actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                else:
                    learner_future = None
                    next_actor_futures = None
            else:
                t_wait_start = time.time()
                ray.get(learner_future)  # Learner(i-1)完了
                actor_results = ray.get(next_actor_futures)  # Actor(i)完了
                t_wait = time.time() - t_wait_start
                if _PROFILE_MODE:
                    print(f"[PROFILE Iter {iteration+1}] Learner+Actor並列待機: {t_wait:.3f}s (合計{sum(actor_results)}エピソード)")
                
                if iteration < N_ITERATIONS - 1:
                    commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
                    learner_future = learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES)
                    next_actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
                else:
                    # 最終イテレーション: Learner(N-1)を実行（Actor(N-1)のデータを使用）
                    t_learner_start = time.time()
                    loss = ray.get(learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES))
                    t_learner = time.time() - t_learner_start
                    if _PROFILE_MODE:
                        print(f"[PROFILE Iter {iteration+1}] Learner実行（最終）: {t_learner:.3f}s")
                    learner_future = None
                    next_actor_futures = None
        else:
            # 従来の逐次モード（Actor完了→Learner実行）
            if DEBUG:
                print("Actorが改良されたエピソードを生成中...")
                print("※ PCNエージェントの_choose_commandsと_nlargestメソッドにより改善された目標値を使用")
            commands_batch = ray.get(learner._choose_commands_batch.remote(50, n_commands_per_iter))
            actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False, pre_fetched_commands=commands_batch) for actor in actors]
            t_actor_start = time.time()
            actor_results = ray.get(actor_futures)
            t_actor = time.time() - t_actor_start
            if _PROFILE_MODE:
                print(f"[PROFILE Iter {iteration+1}] Actor実行: {t_actor:.3f}s (合計{sum(actor_results)}エピソード)")
            
            if DEBUG:
                print("Learnerが改良された経験で学習を実行中")
            t_learner_start = time.time()
            loss = ray.get(learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES))
            t_learner = time.time() - t_learner_start
            if _PROFILE_MODE:
                print(f"[PROFILE Iter {iteration+1}] Learner実行: {t_learner:.3f}s")

        print(f"イテレーション {iteration + 1} 学習完了：平均損失: {loss:.4f}")
        
        # メモリ解放: 学習完了後にReplayBufferをクリア（メモリスピルを防ぐため）
        # get_all_episodes()は既にバッファをクリアするため、明示的なクリアは不要
        # ただし、メモリ使用量を確認
        if DEBUG and iteration % 2 == 0:  # 2イテレーションごとに確認
            buffer_stats = ray.get(buffer.get_stats.remote())
            print(f"[メモリ管理] ReplayBuffer統計: サイズ={buffer_stats['buffer_size']}, 利用率={buffer_stats['utilization']:.1%}")
        
        # ガベージコレクションを実行してメモリを解放
        gc.collect()
        
        # 学習履歴を記録
        training_history['iterations'].append(iteration + 1)
        training_history['losses'].append(loss)
        

        
        # 定期的に評価を実行
        if (iteration + 1) % EVAL_INTERVAL == 0:
            if DEBUG:
                print(f"\n=== イテレーション {iteration + 1} の評価 ===")
                print("※ 改良された経験によるパレートフロントの改善を確認")
            
            if USE_DISTRIBUTED_EVAL:
                # 分散評価を使用
                e_returns, e_values, distances, map_fin = ray.get(learner.evaluate_distributed.remote(actors, n=EVAL_SAMPLES_DISTRIBUTED))
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
                print(len(np.unique(e_returns, axis=0)))
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
                    
                    # 現在の学習結果の非支配解を強調表示（最小化問題なので最小化用の関数を使用）
                    current_all_values = np.array(current_e_values)
                    non_dominated_inds_values = get_non_dominated_inds_minimize(current_all_values)
                    pareto_front_values = current_all_values[non_dominated_inds_values]
                    
                    # デバッグ：現在の値の範囲を確認
                    # print(f"Current values range: X[{current_all_values[:, 0].min():.1f}, {current_all_values[:, 0].max():.1f}], Y[{current_all_values[:, 1].min():.1f}, {current_all_values[:, 1].max():.1f}]")
                    # if initial_axis_ranges and 'values' in initial_axis_ranges:
                    #     print(f"Initial values axis range: X[{initial_axis_ranges['values']['x_min']:.1f}, {initial_axis_ranges['values']['x_max']:.1f}], Y[{initial_axis_ranges['values']['y_min']:.1f}, {initial_axis_ranges['values']['y_max']:.1f}]")
                    
                    plt.scatter(current_all_values[:, 0], current_all_values[:, 1], c='blue', s=60, label='Current All Values', zorder=5)  # サイズを小さく
                    plt.scatter(pareto_front_values[:, 0], pareto_front_values[:, 1], c='red', s=60, label='Current Pareto Front', zorder=5)  # サイズを小さく
                    
                    # パレートフロントの線を描画
                    if len(pareto_front_values) > 1:
                        # パレートフロントをソート
                        sorted_indices = np.lexsort((pareto_front_values[:, 1], pareto_front_values[:, 0]))
                        sorted_pareto = pareto_front_values[sorted_indices]
                        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=1.5, alpha=0.8)  # 線を細く
                    
                    # 軸範囲を現在の値に焦点を当てて設定
                    current_x_min, current_x_max = current_all_values[:, 0].min(), current_all_values[:, 0].max()
                    current_y_min, current_y_max = current_all_values[:, 1].min(), current_all_values[:, 1].max()
                    
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
                    
                    plt.title(f'Pareto Front (Value) - Iter {iteration + 1}\nNon-dominated: {len(non_dominated_inds_values)}', fontsize=10)  # タイトルを短く
                    plt.xlabel('Cost', fontsize=9)
                    plt.ylabel('Execution Time', fontsize=9)
                    plt.legend(fontsize=8)
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # タイムスタンプ付きファイル名で保存（新規作成）
                    values_plot_path = f"{save_dir}/pareto_front_values_current_{timestamp}.png"
                    plt.savefig(values_plot_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    if DEBUG:
                        print(f"軽量化実数値空間パレートフロント更新: {values_plot_path}")
                    
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
            "event_to_bitmap": bool(_EVENT_TO_BITMAP),
            "rows": _rows,
        }
        _summary_path = os.path.join(execution_dir, "training_iteration_summary.json")
        with open(_summary_path, "w", encoding="utf-8") as _sf:
            _json.dump(_summary, _sf, indent=2, allow_nan=False)
        print(f"[SUMMARY] イテレーション別サマリーを保存: {_summary_path}")
    except Exception as _e:
        print(f"[SUMMARY] JSON 保存に失敗: {_e}")
    
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
            
            # 最終評価を実行してパレートフロントを取得
            if DEBUG:
                print("最終パレートフロントを取得中...")
            e_returns, e_values, distances, map_fin = ray.get(learner.evaluate.remote(n=EVAL_SAMPLES_FINAL))
            
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
                
                # 全ての解をプロット
                all_returns = np.array(e_returns)
                plt.scatter(all_returns[:, 0], all_returns[:, 1], c='lightblue', alpha=0.6, label='All Solutions', s=50)
                
                # 初期パレートフロントの軸範囲を設定（軸を合わせる）
                if training_history['initial_axis_ranges'] and 'rewards' in training_history['initial_axis_ranges']:
                    plt.xlim(training_history['initial_axis_ranges']['rewards']['x_min'], training_history['initial_axis_ranges']['rewards']['x_max'])
                    plt.ylim(training_history['initial_axis_ranges']['rewards']['y_min'], training_history['initial_axis_ranges']['rewards']['y_max'])
                
                # 非支配解を強調表示
                non_dominated_inds = get_non_dominated_inds(all_returns)
                pareto_front_returns = all_returns[non_dominated_inds]
                plt.scatter(pareto_front_returns[:, 0], pareto_front_returns[:, 1], c='red', s=100, label='Current Pareto Front', zorder=5)
                
                # パレートフロントの線を描画
                if len(pareto_front_returns) > 1:
                    # パレートフロントをソート
                    sorted_indices = np.lexsort((pareto_front_returns[:, 1], pareto_front_returns[:, 0]))
                    sorted_pareto = pareto_front_returns[sorted_indices]
                    plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
                
                plt.title(f'Pareto Front (Reward Space) - End of Training\nNon-dominated Solutions: {len(non_dominated_inds)}', fontsize=14)
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
                
                # 全ての解をプロット
                all_values = np.array(e_values)
                plt.scatter(all_values[:, 0], all_values[:, 1], c='lightgreen', alpha=0.6, label='All Solutions', s=50)
                
                # 初期パレートフロントの軸範囲を設定（軸を合わせる）
                if training_history['initial_axis_ranges'] and 'values' in training_history['initial_axis_ranges']:
                    plt.xlim(training_history['initial_axis_ranges']['values']['x_min'], training_history['initial_axis_ranges']['values']['x_max'])
                    plt.ylim(training_history['initial_axis_ranges']['values']['y_min'], training_history['initial_axis_ranges']['values']['y_max'])
                
                # 非支配解を強調表示（最小化問題なので最小化用の関数を使用）
                non_dominated_inds_values = get_non_dominated_inds_minimize(all_values)
                pareto_front_values = all_values[non_dominated_inds_values]
                plt.scatter(pareto_front_values[:, 0], pareto_front_values[:, 1], c='red', s=100, label='Current Pareto Front', zorder=5)
                
                # パレートフロントの線を描画
                if len(pareto_front_values) > 1:
                    # パレートフロントをソート
                    sorted_indices = np.lexsort((pareto_front_values[:, 1], pareto_front_values[:, 0]))
                    sorted_pareto = pareto_front_values[sorted_indices]
                    plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'r-', linewidth=2, alpha=0.8)
                
                plt.title(f'Pareto Front (Value Space) - End of Training\nNon-dominated Solutions: {len(non_dominated_inds_values)}', fontsize=14)
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
    
    _mo_hv_path = os.environ.get("DISTRIBUTED_PCN_MO_HV_EXPORT")
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
