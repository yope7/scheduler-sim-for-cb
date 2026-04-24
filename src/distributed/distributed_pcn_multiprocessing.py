"""
multiprocessingを使った並列化版のdistributed_pcn
Rayのシリアライゼーションオーバーヘッドを避けるため、multiprocessingを使用
"""
import numpy as np
import torch as th
import yaml
from tqdm import tqdm
import time
import torch.nn.functional as F
import heapq
import os
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, shared_memory
import copy
import warnings
import pickle

# CUDAが利用できない場合の警告を抑制
warnings.filterwarnings('ignore', message="Can't initialize NVML")
warnings.filterwarnings('ignore', message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available")

# 既存のdistributed_pcn.pyから設定をインポート
from src.distributed.distributed_pcn import (
    DEBUG, TIME_DEBUG, ENABLE_VISUALIZATION,
    N_ITERATIONS, N_ACTORS, N_JOBS,
    EVAL_INTERVAL, USE_DISTRIBUTED_EVAL,
    BATCH_SIZE, N_UPDATES, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD, MIN_ITERATIONS,
    INITIAL_EPISODES, USE_ENHANCED_MODEL,
    SUPERVISED_LEARNING_EPOCHS, SUPERVISED_BATCH_SIZE,
    SUPERVISED_UPDATES_PER_EPOCH, SUPERVISED_LEARNING_RATE,
    VISUALIZATION_INTERVAL, EPISODES_PER_ITERATION,
    EVAL_SAMPLES, EVAL_SAMPLES_DISTRIBUTED, EVAL_SAMPLES_FINAL
)

from src.agents.pcn_agent import (
    PCN, 
    Transition, 
    get_non_dominated_inds, 
    get_non_dominated_inds_minimize,
    crowding_distance,
    hypervolume
)
from src.envs.c_scheduling_env.scheduling_env_cache_optimized import SchedulingEnvCacheOptimized
from src.utils.job_gen.job_generator import JobGenerator


# =========================
# 1. Replay Buffer (multiprocessing版)
# =========================
class ReplayBuffer:
    """multiprocessing版のReplayBuffer（共有メモリを使用）"""
    def __init__(self, max_size=10000):
        self.buffer = Manager().list()  # 共有リスト
        self.max_size = max_size
        self.episode_hashes = Manager().set()  # 共有セット
        self._hash_cache = Manager().dict()  # 共有辞書
        self.lock = Manager().Lock()  # ロック
        if DEBUG:
            print(f"ReplayBuffer initialized with max_size={max_size}")

    def add(self, episode):
        """エピソードを追加（ロック付き）"""
        with self.lock:
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
                if oldest_episode_id in self._hash_cache:
                    del self._hash_cache[oldest_episode_id]
            
            # 新しいエピソードを追加
            self.buffer.append(episode)
            self.episode_hashes.add(episode_hash)
            
            # ログ出力を簡潔にする（100エピソードごとに表示）
            if DEBUG and len(self.buffer) % 100 == 0:
                print(f"ReplayBuffer: episode added, current size={len(self.buffer)}")
    
    def add_batch(self, episodes):
        """複数のエピソードを一度に追加（ロック付き）"""
        with self.lock:
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
                    if oldest_episode_id in self._hash_cache:
                        del self._hash_cache[oldest_episode_id]
                
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
        """全てのエピソードを取得してバッファをクリア（ロック付き）"""
        with self.lock:
            # 深いコピーを作成して、元のオブジェクトとの参照を完全に分離
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
# 2. Actor (multiprocessing版)
# =========================
def actor_worker(config, actor_id, learner_queue, buffer_queue, episode_queue, weights_queue, commands_queue):
    """Actorワーカー関数（multiprocessing版）"""
    # 環境とエージェントの初期化
    job_generator = JobGenerator(
        0, 1,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, N_JOBS, 0.2, 0
    )
    jobs_set = job_generator.generate_jobs_set()
    env = SchedulingEnvCacheOptimized(
        np.inf,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config['param_env']['n_job_queue_obs'],
        config['param_env']['n_job_queue_bck'],
        config['param_agent']['weight_wt'],
        config['param_agent']['weight_cost'],
        config['param_env']['penalty_not_allocate'],
        config['param_env']['penalty_invalid_action'],
        jobs_set,
        None, flag=0
    )
    
    # PCNエージェントの初期化（CPUで実行）
    agent = PCN(
        env,
        device='cpu',
        state_dim=env.observation_space.shape[0],
        scaling_factor=np.array([1, 1, 1]),
        learning_rate=LEARNING_RATE,
        batch_size=512,
        hidden_dim=512,
        project_name="temp",
        experiment_name="PCN",
        log=False,
        debug_mode=DEBUG,
        use_enhanced_model=USE_ENHANCED_MODEL,
    )
    
    if DEBUG:
        print(f"[Actor {actor_id}] ✓ C実装環境が正しく初期化されました")
    
    # メインループ
    while True:
        # コマンドを待機
        command = episode_queue.get()
        if command is None:  # 終了シグナル
            break
        
        n_episodes, random_actions = command
        
        # 最新重みを取得（非ランダムアクションの場合）
        if not random_actions:
            # 重みをリクエスト
            weights_queue.put(('request', actor_id))
            # 重みを受信
            weights = weights_queue.get()
            if weights is not None:
                agent.model.load_state_dict(weights)
            
            # 目標値をリクエスト
            commands_queue.put(('request', actor_id))
            # 目標値を受信
            desired_return, desired_horizon = commands_queue.get()
            if desired_return is not None:
                agent.set_desired_return_and_horizon(desired_return, desired_horizon)
        
        # エピソードを収集
        collected_episodes = []
        for ep in range(n_episodes):
            episode = _run_episode(env, agent, random_actions, actor_id)
            collected_episodes.append(episode)
        
        # エピソードをバッファに送信
        buffer_queue.put(('add_batch', collected_episodes))
        
        # 完了を通知
        episode_queue.put(('done', actor_id, len(collected_episodes)))


def _run_episode(env, agent, random_actions, actor_id):
    """エピソードを実行（multiprocessing版）"""
    obs = env.reset()
    done = False
    transitions = []
    
    if random_actions:
        episode_seed = (int(time.time() * 1000000) + actor_id * 10000 + hash(str(obs))) % 10000
        np.random.seed(episode_seed)
    
    start_time = time.time()
    while not done:
        if random_actions:
            action = env.action_space.sample()
        else:
            action = agent.eval(obs)
        
        n_obs, reward, scheduled, wt_step, done = env.step(action)
        
        # 観測データをfloat32に変換（既にfloat32の場合は変換しない）
        if hasattr(obs, 'dtype') and obs.dtype != np.float32:
            obs = np.array(obs, dtype=np.float32, copy=True)
        if hasattr(n_obs, 'dtype') and n_obs.dtype != np.float32:
            n_obs = np.array(n_obs, dtype=np.float32, copy=True)
        
        transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
        obs = n_obs
    
    # エピソード完了時に実数値を計算
    if done:
        env.finalize_window_history()
        cost, _, avg_waiting_time = env.calc_objective_values()
        solution_execution_time = time.time() - start_time
        
        # 最初のTransitionに実数値を追加
        if len(transitions) > 0:
            transitions[0].objective_values = [cost, _, avg_waiting_time]
            if not random_actions and solution_execution_time is not None:
                transitions[0].solution_execution_time = solution_execution_time
    
    return transitions


# =========================
# 3. Learner (multiprocessing版)
# =========================
class Learner:
    """multiprocessing版のLearner"""
    def __init__(self, config, buffer, device='cuda'):
        self.config = config
        self.env = self._make_env()
        
        # より堅牢なデバイス検出
        self.actual_device = self._get_available_device(device)
        
        # PCNエージェントを正しいデバイスで初期化
        self.agent = PCN(
            self.env,
            device=self.actual_device,
            state_dim=self.env.observation_space.shape[0],
            scaling_factor=np.array([1, 1, 1]),
            learning_rate=LEARNING_RATE,
            batch_size=512,
            hidden_dim=512,
            project_name="temp",
            experiment_name="PCN",
            log=False,
            debug_mode=DEBUG,
            use_enhanced_model=USE_ENHANCED_MODEL,
        )
        self.buffer = buffer
        self.global_step = 0
        self.experience_replay = []
        self.gamma = 1.0
        self.last_eval_step = 0
        self._hash_cache = {}
        if DEBUG:
            print(f"Learner initialized with device: {self.actual_device}")
            print(f"Learner model: {'EnhancedPCNModel' if USE_ENHANCED_MODEL else 'DiscreteActionsDefaultModel'}")
            if self.actual_device == 'cuda':
                import torch
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    def _get_available_device(self, requested_device):
        """利用可能なデバイスを検出"""
        import torch
        
        if requested_device == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        else:
            return requested_device

    def _make_env(self):
        """環境を作成"""
        job_generator = JobGenerator(
            0, 1,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config, N_JOBS, 0.2, 0
        )
        jobs_set = job_generator.generate_jobs_set()
        env = SchedulingEnvCacheOptimized(
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
        if hasattr(env, '_cache_onpre_c'):
            print("[Learner] ✓ C実装環境が正しく初期化されました")
        return env

    def get_weights(self):
        """モデルの重みを取得"""
        if USE_ENHANCED_MODEL and hasattr(self.agent, 'network'):
            model_state = self.agent.network.state_dict()
        else:
            model_state = self.agent.model.state_dict()
        return {k: v.cpu() for k, v in model_state.items()}

    def _choose_commands(self, num_episodes: int):
        """次のエピソードの目標報酬とホライズンを選択"""
        return self.agent._choose_commands(num_episodes)

    def learn(self, batch_size: int = 100, n_updates: int = 2):
        """学習を実行"""
        total_loss = []
        
        # ReplayBufferから全てのエピソードを取得
        buffer_size = self.buffer.size()
        if buffer_size == 0:
            return 0.0
        
        # 全てのエピソードを取得
        all_episodes = self.buffer.get_all_episodes()
        if not all_episodes:
            return 0.0
        
        # 全てのエピソードを経験再生バッファに追加
        for episode in all_episodes:
            self._add_episode(episode, max_size=10000, step=self.global_step)
        
        # 学習更新を実行
        for i in range(n_updates):
            try:
                loss, _ = self.agent.update()
                loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
                
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"[Learner] 警告: 損失がNaN/Infになりました (update {i})")
                    loss_value = 0.0
                
                total_loss.append(loss_value)
            except Exception as e:
                print(f"[Learner] エラー: 学習更新中にエラーが発生しました (update {i}): {e}")
                import traceback
                traceback.print_exc()
                total_loss.append(0.0)
            
            self.global_step += 1
        
        return np.mean(total_loss) if total_loss else 0.0

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        """エピソードを経験再生バッファに追加"""
        # 各Transitionのコピーを作成
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
            transitions_copy.append(t_copy)
        for i in reversed(range(len(transitions_copy) - 1)):
            transitions_copy[i].reward += self.gamma * transitions_copy[i + 1].reward

        # エピソードの内容に基づくハッシュ値を計算
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
        """エピソードの内容に基づくハッシュ値を計算"""
        import hashlib
        
        if not transitions:
            return 0
        
        transitions_id = id(transitions)
        if transitions_id in self._hash_cache:
            return self._hash_cache[transitions_id]
        
        hasher = hashlib.md5()
        episode_len = len(transitions)
        hasher.update(episode_len.to_bytes(8, byteorder='big'))
        
        first_obs = transitions[0].observation
        if hasattr(first_obs, 'tobytes'):
            obs_summary = first_obs.flatten()[:min(100, first_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(first_obs).encode())
        
        actions = np.array([t.action for t in transitions], dtype=np.int32)
        hasher.update(actions.tobytes())
        
        rewards = np.array([t.reward for t in transitions])
        if rewards.size > 0:
            reward_summary = np.array([rewards.sum(), rewards.mean()], dtype=np.float32)
            hasher.update(reward_summary.tobytes())
        
        last_obs = transitions[-1].next_observation
        if hasattr(last_obs, 'tobytes'):
            obs_summary = last_obs.flatten()[:min(100, last_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(last_obs).encode())
        
        terminal_info = np.array([t.terminal for t in transitions], dtype=bool)
        hasher.update(terminal_info.tobytes())
        
        hash_value = int(hasher.hexdigest(), 16)
        self._hash_cache[transitions_id] = hash_value
        
        return hash_value
    
    def _is_duplicate_episode(self, episode_hash: int) -> bool:
        """エピソードが重複しているかチェック"""
        if not hasattr(self, '_episode_hashes'):
            self._episode_hashes = set()
        return episode_hash in self._episode_hashes

    def _get_buffer_size(self) -> int:
        return len(self.agent.experience_replay)

    def update(self, learning_rate=None):
        """PCNエージェントのupdateメソッドを呼び出す"""
        return self.agent.update(learning_rate=learning_rate)

    def get_global_step(self) -> int:
        """グローバルステップを取得"""
        return self.global_step

    def get_experience_replay(self):
        """experience replayの内容を取得（コピーを返す）"""
        replay_copy = []
        for priority, unique_step, transitions in self.agent.experience_replay:
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
                if hasattr(t, 'objective_values'):
                    t_copy.objective_values = t.objective_values
                transitions_copy.append(t_copy)
            replay_copy.append((priority, unique_step, transitions_copy))
        
        if DEBUG:
            print(f"[Learner] experience_replayの内容を取得: {len(replay_copy)} エピソード")
        
        return replay_copy

    def evaluate(self, max_return=None, n=10, training_iteration=None, eval_diag_path=None):
        """エージェントの評価を実行"""
        if max_return is None:
            max_return = np.full(2, 100.0, dtype=np.float32)
        
        if DEBUG:
            print("評価を実行中...")
        eval_diag = None
        if eval_diag_path:
            eval_diag = {"path": eval_diag_path, "training_iteration": training_iteration}
        e_returns, e_value, distances, map_fin = self.agent.evaluate(
            self.env, max_return, n=n, eval_diag=eval_diag
        )
        return e_returns, e_value, distances, map_fin


# =========================
# 4. メイン関数（multiprocessing版）
# =========================
def main():
    """multiprocessing版のメイン関数"""
    import matplotlib.pyplot as plt
    import os
    
    # 実行用のディレクトリを作成
    execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_dir = f"execution_{execution_timestamp}"
    os.makedirs(execution_dir, exist_ok=True)
    
    if TIME_DEBUG:
        overall_start_time = time.time()
        print(f"\n{'='*60}")
        print("分散PCN学習開始（multiprocessing版）")
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"実行ディレクトリ: {execution_dir}")
        print(f"{'='*60}")
    
    # 設定ファイルの読み込み
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    
    # multiprocessingの設定
    mp.set_start_method('spawn', force=True)  # Windows互換性のため
    
    # Managerを作成（共有メモリ）
    manager = Manager()
    
    # Replay Buffer
    buffer = ReplayBuffer(max_size=10000)
    
    # Learner
    learner = Learner(config, buffer, device='cuda')
    
    # Actor用のキューを作成
    episode_queues = [Queue() for _ in range(N_ACTORS)]
    buffer_queue = Queue()
    weights_queue = Queue()
    commands_queue = Queue()
    
    # Actorプロセスを起動
    actor_processes = []
    for i in range(N_ACTORS):
        p = Process(
            target=actor_worker,
            args=(config, i, None, buffer_queue, episode_queues[i], weights_queue, commands_queue)
        )
        p.start()
        actor_processes.append(p)
    
    # 重みとコマンドを配布するワーカー
    def weights_distributor():
        """重みを配布するワーカー"""
        while True:
            msg = weights_queue.get()
            if msg is None:
                break
            if msg[0] == 'request':
                actor_id = msg[1]
                weights = learner.get_weights()
                weights_queue.put(weights)
    
    def commands_distributor():
        """コマンドを配布するワーカー"""
        while True:
            msg = commands_queue.get()
            if msg is None:
                break
            if msg[0] == 'request':
                actor_id = msg[1]
                desired_return, desired_horizon = learner._choose_commands(50)
                commands_queue.put((desired_return, desired_horizon))
    
    # バッファ処理ワーカー
    def buffer_worker():
        """バッファを処理するワーカー"""
        while True:
            msg = buffer_queue.get()
            if msg is None:
                break
            if msg[0] == 'add_batch':
                episodes = msg[1]
                buffer.add_batch(episodes)
    
    # ワーカープロセスを起動
    weights_process = Process(target=weights_distributor)
    commands_process = Process(target=commands_distributor)
    buffer_process = Process(target=buffer_worker)
    
    weights_process.start()
    commands_process.start()
    buffer_process.start()
    
    try:
        # フェーズ1: 初期エピソードの収集
        if DEBUG or TIME_DEBUG:
            print("\n" + "="*60)
            print("フェーズ1: 初期エピソードの収集")
            print("="*60)
        
        if TIME_DEBUG:
            phase1_start_time = time.time()
            print(f"フェーズ1開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 各Actorで初期エピソードを実行
        for i, queue in enumerate(episode_queues):
            queue.put((INITIAL_EPISODES, True))  # random_actions=True
        
        # 完了を待機
        total_episodes = 0
        for i, queue in enumerate(episode_queues):
            msg = queue.get()
            if msg[0] == 'done':
                total_episodes += msg[2]
        
        if TIME_DEBUG:
            phase1_end_time = time.time()
            phase1_duration = phase1_end_time - phase1_start_time
            print(f"\n{'='*40}")
            print(f"フェーズ1完了: 初期エピソード収集")
            print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"経過時間: {phase1_duration:.2f}秒 ({phase1_duration/60:.2f}分)")
            print(f"生成エピソード数: {total_episodes}")
            print(f"{'='*40}")
        
        # 初期学習
        initial_loss = learner.learn(batch_size=BATCH_SIZE, n_updates=N_UPDATES)
        print(f"初期学習の損失: {initial_loss}")
        
        # フェーズ2: 教師あり学習
        if DEBUG or TIME_DEBUG:
            print("\n" + "="*60)
            print("フェーズ2: 教師あり学習")
            print("="*60)
        
        if TIME_DEBUG:
            phase2_start_time = time.time()
            print(f"フェーズ2開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for epoch in range(SUPERVISED_LEARNING_EPOCHS):
            if DEBUG:
                print(f"\n--- 教師あり学習エポック {epoch + 1}/{SUPERVISED_LEARNING_EPOCHS} ---")
            
            epoch_losses = []
            for update in range(SUPERVISED_UPDATES_PER_EPOCH):
                loss, _ = learner.update(SUPERVISED_LEARNING_RATE)
                epoch_losses.append(loss.item())
            
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            print(f"エポック {epoch + 1} 完了: 平均損失 = {avg_epoch_loss:.4f}")
        
        if TIME_DEBUG:
            phase2_end_time = time.time()
            phase2_duration = phase2_end_time - phase2_start_time
            print(f"\n{'='*40}")
            print(f"フェーズ2完了: 教師あり学習")
            print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"経過時間: {phase2_duration:.2f}秒 ({phase2_duration/60:.2f}分)")
            print(f"{'='*40}")
        
        # フェーズ3: 改良された経験の実現
        if DEBUG or TIME_DEBUG:
            print("\n" + "="*60)
            print("フェーズ3: 改良された経験の実現")
            print("="*60)
        
        if TIME_DEBUG:
            phase3_start_time = time.time()
            print(f"フェーズ3開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        training_history = {
            'iterations': [],
            'losses': [],
            'pareto_front_sizes': [],
            'distances': [],
        }
        
        # 学習ループ
        for iteration in range(N_ITERATIONS):
            # Actorでエピソード生成を並列実行
            if DEBUG:
                print("Actorが改良されたエピソードを生成中...")
            
            # 各Actorにエピソード生成を依頼
            for i, queue in enumerate(episode_queues):
                queue.put((EPISODES_PER_ITERATION, False))  # random_actions=False
            
            # 完了を待機
            for i, queue in enumerate(episode_queues):
                msg = queue.get()
                if msg[0] == 'done':
                    pass  # 完了を確認
            
            # Learnerで学習を実行
            if DEBUG:
                print("Learnerが改良された経験で学習を実行中")
            loss = learner.learn(batch_size=BATCH_SIZE, n_updates=N_UPDATES)
            
            print(f"イテレーション {iteration + 1} 学習完了：平均損失: {loss:.4f}")
            
            training_history['iterations'].append(iteration + 1)
            training_history['losses'].append(loss)
            
            # 定期的に評価を実行
            if (iteration + 1) % EVAL_INTERVAL == 0:
                if DEBUG:
                    print(f"\n=== イテレーション {iteration + 1} の評価 ===")
                e_returns, e_values, distances, map_fin = learner.evaluate(n=EVAL_SAMPLES)
                training_history['pareto_front_sizes'].append(len(e_returns))
                training_history['distances'].append(distances if len(distances) > 0 else [])
            else:
                training_history['pareto_front_sizes'].append(None)
                training_history['distances'].append(None)
        
        if TIME_DEBUG:
            phase3_end_time = time.time()
            phase3_duration = phase3_end_time - phase3_start_time
            print(f"\n{'='*40}")
            print(f"フェーズ3完了: 改良された経験の実現")
            print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"経過時間: {phase3_duration:.2f}秒 ({phase3_duration/60:.2f}分)")
            print(f"{'='*40}")
    
    finally:
        # プロセスを終了
        for queue in episode_queues:
            queue.put(None)  # 終了シグナル
        
        buffer_queue.put(None)
        weights_queue.put(None)
        commands_queue.put(None)
        
        for p in actor_processes:
            p.join()
        
        weights_process.join()
        commands_process.join()
        buffer_process.join()
        
        if TIME_DEBUG:
            overall_end_time = time.time()
            overall_duration = overall_end_time - overall_start_time
            print(f"\n{'='*60}")
            print("分散PCN学習完了（multiprocessing版）")
            print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"総経過時間: {overall_duration:.2f}秒 ({overall_duration/60:.2f}分)")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()

