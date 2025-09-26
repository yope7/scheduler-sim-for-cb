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
import ray
import copy

# =========================
# 0. ハイパーパラメータ設定
# =========================

DEBUG = False
TIME_DEBUG = True  # 各フェーズの経過時間を表示
ENABLE_VISUALIZATION = True

N_ITERATIONS = 300  # 全体の学習イテレーション数
N_ACTORS = 50      # 並列実行するActorの数
N_JOBS = 128 # ジョブ数

EVAL_INTERVAL = 5  # 評価を実行する間隔（イテレーション数）
USE_DISTRIBUTED_EVAL = False  # 分散評価を使用するかどうか

BATCH_SIZE = 2048
N_UPDATES = 3
LEARNING_RATE = 1e-2

EARLY_STOPPING_PATIENCE = 5  # 改善が見られないイテレーション数
EARLY_STOPPING_THRESHOLD = 0.0001  # 改善とみなす最小変化量
MIN_ITERATIONS = 5  # 最低限実行するイテレーション数


INITIAL_EPISODES =  40 #初期エピソード数

USE_ENHANCED_MODEL = False  # True: EnhancedPCNModel, False: DiscreteActionsDefaultModel (3層NLPモデル)


SUPERVISED_LEARNING_EPOCHS = 60
SUPERVISED_BATCH_SIZE = 1024    
SUPERVISED_UPDATES_PER_EPOCH = 3 
SUPERVISED_LEARNING_RATE = 1e-2  

VISUALIZATION_INTERVAL =5  # 可視化を実行する間隔（イテレーション数）

EPISODES_PER_ITERATION = 1  # 各イテレーションで各Actorが生成するエピソード数

EVAL_SAMPLES = 100  # 評価時に使用するサンプル数
EVAL_SAMPLES_DISTRIBUTED = 10  # 分散評価時に使用するサンプル数
EVAL_SAMPLES_FINAL = 50  # 最終評価時に使用するサンプル数

from src.agents.pcn_agent import (
    PCN, 
    Transition, 
    get_non_dominated_inds, 
    get_non_dominated_inds_minimize,
    crowding_distance,
    hypervolume
)
from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator

# =========================
# 1. Replay Buffer (Ray Actor)
# =========================
@ray.remote
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.episode_hashes = set()  # 重複検出用のハッシュセット
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
        
        # 新しいエピソードを追加
        self.buffer.append(episode)
        self.episode_hashes.add(episode_hash)
        
        # ログ出力を簡潔にする（100エピソードごとに表示）
        if DEBUG and len(self.buffer) % 100 == 0:
            print(f"ReplayBuffer: episode added, current size={len(self.buffer)}")

    def _compute_episode_hash(self, episode):
        """エピソードの内容に基づくハッシュ値を計算"""
        import hashlib
        # エピソードの特徴を文字列として結合
        episode_str = ""
        for t in episode:
            # 観察、行動、報酬の情報を文字列化
            obs_str = str(t.observation.tobytes()) if hasattr(t.observation, 'tobytes') else str(t.observation)
            action_str = str(t.action)
            reward_str = str(t.reward.tobytes()) if hasattr(t.reward, 'tobytes') else str(t.reward)
            next_obs_str = str(t.next_observation.tobytes()) if hasattr(t.next_observation, 'tobytes') else str(t.next_observation)
            terminal_str = str(t.terminal)
            
            episode_str += f"{obs_str}|{action_str}|{reward_str}|{next_obs_str}|{terminal_str}|"
        
        # ハッシュ値を計算
        return hash(episode_str)

    def get_all_episodes(self):
        """全てのエピソードを取得してバッファをクリア"""
        import copy
        # 深いコピーを作成して、元のオブジェクトとの参照を完全に分離
        result = copy.deepcopy(self.buffer)
        self.buffer.clear()
        self.episode_hashes.clear()  # ハッシュセットもクリア
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
        if DEBUG:
            print(f"Actor {actor_id} initialized")

    def _get_available_device(self, requested_device):
        """利用可能なデバイスを安全に検出"""
        import torch
        
        if requested_device == 'cuda':
            try:
                # CUDAが利用可能かチェック
                if torch.cuda.is_available():
                    # 実際にCUDAデバイスにアクセスできるかテスト
                    test_tensor = torch.tensor([1.0], device='cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                    if DEBUG:
                        print(f"Actor {self.actor_id}: CUDA is available and working.")
                    return 'cuda'
                else:
                    if DEBUG:
                        print(f"Actor {self.actor_id}: CUDA is not available. Using CPU.")
                    return 'cpu'
            except Exception as e:
                if DEBUG:
                    print(f"Actor {self.actor_id}: CUDA test failed: {e}. Using CPU.")
                return 'cpu'
        else:
            if DEBUG:
                print(f"Actor {self.actor_id}: Using requested device: {requested_device}")
            return requested_device

    def _make_env(self):
        if self.env is None:
            job_generator = JobGenerator(
                0, 1,
                self.config['param_env']['n_window'],
                self.config['param_env']['n_on_premise_node'],
                self.config['param_env']['n_cloud_node'],
                self.config, N_JOBS, 0.2, 0
            )
            jobs_set = job_generator.generate_jobs_set()
            self.env = SchedulingEnv(
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

    def run(self, n_episodes=10, random_actions=False):
        # print("init env start")
        if self.env is None:
            self._make_env()
            # print("act init env")
        
        episodes_generated = 0
        collected_episodes = []  # 収集したエピソードを一時保存
        solution_execution_times = []  # 改良された解の実行時間を記録
        
        # 進捗表示の間隔を計算（n_episodesの10分の1）
        progress_interval = max(1, n_episodes // 10)
        
        for ep in range(n_episodes):
            try:
                # 最新重みをLearnerからpull
                weights = ray.get(self.learner.get_weights.remote())
                self.agent.model.load_state_dict(weights)
                # print("1epi")
                # 1エピソード実行
                episode = self._run_episode(random_actions)
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
        
        # 全てのエピソードをReplayBufferに追加（完了を待機）
        for episode in collected_episodes:
            ray.get(self.buffer.add.remote(episode))  # 完了を待機
        
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

    def _run_episode(self, random_actions=False):
        obs = self.env.reset()
        done = False
        transitions = []
        
        # 改良された解の選択から実行完了までの時間計測を開始
        solution_selection_start_time = None
        solution_execution_time = None
        
        if not random_actions:
            # 改良された解の選択開始時刻を記録
            solution_selection_start_time = time.time()
            
            # Learnerから目標値を取得（改良された解の選択）
            desired_return, desired_horizon = ray.get(self.learner._choose_commands.remote(50))
            self.agent.set_desired_return_and_horizon(desired_return, desired_horizon)
            
            # print(f"[Actor {self.actor_id}] 改良された解の選択完了: 目標報酬={desired_return}, ホライズン={desired_horizon}")
        
        # ランダムアクションの場合、エピソードごとに異なるシードを設定
        if random_actions:
            # Actor ID、現在時刻、エピソードIDを組み合わせてユニークなシードを生成
            episode_seed = (int(time.time() * 1000000) + self.actor_id * 10000 + hash(str(obs))) % 10000
            np.random.seed(episode_seed)
            if DEBUG:
                print(f"[Actor {self.actor_id}] ランダムアクション用シード設定: {episode_seed}")
        start_time = time.time()
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
            
            transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
            obs = n_obs
            
        # エピソード完了時に実数値を計算
        if done:
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
        
        # 最新重みをLearnerからpull
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
@ray.remote
class Learner:
    def __init__(self, config, buffer, device='cuda'):
        self.config = config
        self.env = self._make_env()
        
        # より堅牢なデバイス検出
        self.actual_device = self._get_available_device(device)
        
        # PCNエージェントはCPUで初期化（GPU使用時のみGPUに転送）
        self.agent = PCN(
            self.env,
            device='cpu',  # 常にCPUで初期化
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
        self.buffer = buffer
        self.global_step = 0
        self.experience_replay = []  # PCNエージェントの経験再生バッファ
        self.gamma = 1.0  # 割引率
        self.last_eval_step = 0  # 最後に評価を行ったステップ
        if DEBUG:
            print(f"Learner initialized with device: {self.actual_device} (agent on CPU)")
            print(f"Learner model: {'EnhancedPCNModel' if USE_ENHANCED_MODEL else 'DiscreteActionsDefaultModel'}")

    def _get_available_device(self, requested_device):
        """利用可能なデバイスを安全に検出"""
        import torch
        
        if requested_device == 'cuda':
            try:
                # Ray環境でのGPUリソース確認
                if hasattr(ray, 'get_gpu_ids') and ray.get_gpu_ids():
                    if DEBUG:
                        print(f"Ray GPU detected: {ray.get_gpu_ids()}")
                
                # CUDAが利用可能かチェック
                if torch.cuda.is_available():
                    # 実際にCUDAデバイスにアクセスできるかテスト
                    test_tensor = torch.tensor([1.0], device='cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                    if DEBUG:
                        print(f"CUDA is available and working. Using CUDA device.")
                        print(f"CUDA device count: {torch.cuda.device_count()}")
                        print(f"Current CUDA device: {torch.cuda.current_device()}")
                    return 'cuda'
                else:
                    if DEBUG:
                        print(f"CUDA is not available. Falling back to CPU.")
                    return 'cpu'
            except Exception as e:
                if DEBUG:
                    print(f"CUDA test failed: {e}. Falling back to CPU.")
                return 'cpu'
        else:
            if DEBUG:
                print(f"Using requested device: {requested_device}")
            return requested_device

    def _make_env(self):
        job_generator = JobGenerator(
            0, 1,
            self.config['param_env']['n_window'],
            self.config['param_env']['n_on_premise_node'],
            self.config['param_env']['n_cloud_node'],
            self.config, N_JOBS, 0.2, 0
        )
        jobs_set = job_generator.generate_jobs_set()
        env = SchedulingEnv(
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
        return env

    def get_weights(self):
        # CPUデバイスでモデルの重みを返す（ActorがCPUで実行されるため）
        return {k: v.cpu() for k, v in self.agent.model.state_dict().items()}

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
        """エピソードの内容に基づくハッシュ値を計算"""
        import hashlib
        # エピソードの特徴を文字列として結合
        episode_str = ""
        for t in transitions:
            # 観察、行動、報酬の情報を文字列化
            obs_str = str(t.observation.tobytes()) if hasattr(t.observation, 'tobytes') else str(t.observation)
            action_str = str(t.action)
            reward_str = str(t.reward.tobytes()) if hasattr(t.reward, 'tobytes') else str(t.reward)
            next_obs_str = str(t.next_observation.tobytes()) if hasattr(t.next_observation, 'tobytes') else str(t.next_observation)
            terminal_str = str(t.terminal)
            
            episode_str += f"{obs_str}|{action_str}|{reward_str}|{next_obs_str}|{terminal_str}|"
        
        # ハッシュ値を計算
        return hash(episode_str)
    
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

    def learn(self, batch_size: int = 100, n_updates: int = 2) -> float:
        total_loss = []
        
        # ReplayBufferから全てのエピソードを取得（サンプリングせずに全部）
        buffer_size = ray.get(self.buffer.size.remote())
        if buffer_size == 0:
            return 0.0
        
        # 全てのエピソードを取得
        all_episodes = ray.get(self.buffer.get_all_episodes.remote())
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
        for i in range(n_updates):
            # GPU使用時のみGPUに転送（初回のみ）
            if self.actual_device == 'cuda' and i == 0:
                self.agent.model.to('cuda')
                if hasattr(self.agent, 'target_model'):
                    self.agent.target_model.to('cuda')
            
            # PCNエージェントのupdateメソッドを呼び出し
            loss, _ = self.agent.update()
            total_loss.append(loss.item())
            
            # 学習後はCPUに戻す（最後の更新のみ）
            if self.actual_device == 'cuda' and i == n_updates - 1:
                self.agent.model.to('cpu')
                if hasattr(self.agent, 'target_model'):
                    self.agent.target_model.to('cpu')
            
            if DEBUG and i % 10 == 0:
                print(f"[Learner] {i} updates done. Buffer size: {ray.get(self.buffer.size.remote())}")
                print(f"[Learner] Average loss: {np.mean(total_loss[-10:]):.4f}")
            
            self.global_step += 1
        
        return np.mean(total_loss) if total_loss else 0.0

    def evaluate(self, max_return=None, n=10):
        """エージェントの評価を実行"""
        if max_return is None:
            max_return = np.full(2, 100.0, dtype=np.float32)
        
        if DEBUG:
            print("評価を実行中...")
        # PCNエージェントのevaluate()を呼び出し（内部で出力処理も行われる）
        e_returns, e_value, distances, map_fin = self.agent.evaluate(self.env, max_return, n=n)
        
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
        
        return e_returns, e_values, [], map_fin  # distancesは計算しない（分散評価では不要）

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
            
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # モデルの状態辞書を保存
            model_state = {
                'model_state_dict': self.agent.model.state_dict(),
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
        plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
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
    
    # 実行用のディレクトリを作成
    execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_dir = f"execution_{execution_timestamp}"
    os.makedirs(execution_dir, exist_ok=True)
    
    if TIME_DEBUG:
        overall_start_time = time.time()
        print(f"\n{'='*60}")
        print("分散PCN学習開始")
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"実行ディレクトリ: {execution_dir}")
        print(f"{'='*60}")
    
    # 設定ファイルの読み込み
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    ray.init(ignore_reinit_error=True)

    # Replay Buffer
    buffer = ReplayBuffer.remote(max_size=10000)

    learner = Learner.remote(config, buffer, device='cuda')

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
    initial_futures = []
    total_episodes = 0
    completed_actors = 0

    for i, (actor, init_future) in enumerate(zip(actors, init_futures)):
        try:
            ray.get(init_future)

            simulation_future = actor.run.remote(n_episodes=INITIAL_EPISODES, random_actions=True)
            initial_futures.append((i, simulation_future))
        
        except Exception as e:
            print(f"Actor {i}の初期化でエラーが発生: {e}")

    for i, future in initial_futures:
        try:
            episodes_generated = ray.get(future)
            total_episodes += episodes_generated
            completed_actors += 1
            
            # 進捗を表示
            progress_percentage = (completed_actors / N_ACTORS) * 100
            if DEBUG:
                print(f"Actor {i} の初期エピソード生成完了: {episodes_generated} エピソード (進捗: {progress_percentage:.1f}%)")
                
        except Exception as e:
            print(f"Actor {i} でエラーが発生: {e}")

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

    initial_batch = ray.get(learner.get_experience_replay.remote())

    # =========================
    # 初期経験収集後のパレートフロント可視化
    # =========================
    # final_buffer_size = ray.get(learner._get_buffer_size.remote())
    # initial_axis_ranges = visualize_initial_pareto_front(initial_batch, save_dir=execution_dir)

    # =========================
    # フェーズ1終了時の学習データ分析と保存
    # =========================
    print("\n" + "="*60)
    print("フェーズ1終了: 学習データの分析と保存")
    print("="*60)
    
    try:
        # Learnerから学習データを取得
        experience_replay = ray.get(learner.get_experience_replay.remote())
        
        if len(experience_replay) > 0:
            print(f"✓ 学習データの分析を開始します...")
            print(f"  エピソード数: {len(experience_replay)}")
            
            # 詳細分析ファイルの生成
            analysis_file = ray.get(learner.save_learning_data_to_file.remote(
                filename=f"phase1_learning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                sample_size=500
            ))
            
            
            print(f"✓ フェーズ1学習データの分析完了!")
            print(f"  詳細分析ファイル: {analysis_file}")
            # print(f"  サンプルデータCSV: {csv_file}")
            
            # 簡単な統計情報の表示
            print("\n=== フェーズ1学習データ統計 ===")
            total_transitions = sum(len(episode[2]) for episode in experience_replay)
            print(f"総エピソード数: {len(experience_replay)}")
            print(f"総遷移数: {total_transitions}")
            print(f"平均エピソード長: {total_transitions / len(experience_replay):.1f}")
            
            # 行動分布の確認
            all_actions = []
            for episode in experience_replay:
                for transition in episode[2]:
                    all_actions.append(transition.action)
            
            unique_actions, action_counts = np.unique(all_actions, return_counts=True)
            print(f"行動分布:")
            for action, count in zip(unique_actions, action_counts):
                percentage = (count / len(all_actions)) * 100
                print(f"  行動{action}: {count}回 ({percentage:.1f}%)")
            
            # 不均衡チェック
            max_action_ratio = np.max(action_counts) / len(all_actions)
            if max_action_ratio > 0.8:
                print(f"⚠️  行動不均衡検出: {max_action_ratio:.1%}が同じ行動")
            else:
                print(f"✓ 行動分布はバランス良好")
                
        else:
            print("⚠️  学習データが空です。フェーズ1のデータ収集を確認してください。")
            
    except Exception as e:
        print(f"❌ 学習データ分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("フェーズ1完了: Enterキーを押してフェーズ2に進む")
    print("="*60)

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
        print(f"  学習率: {current_learning_rate}")
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
                print(f"  報酬の最小値: {np.min(rewards_array, axis=0)}")
                print(f"  報酬の最大値: {np.max(rewards_array, axis=0)}")
                
                # 行動の分布も確認
                all_actions = [t.action for t in episode[2]]
                actions_array = np.array(all_actions)
                unique_actions, counts = np.unique(actions_array, return_counts=True)
                print(f"  行動の分布: {dict(zip(unique_actions, counts))}")
                    
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
    for iteration in range(N_ITERATIONS):
        # Actorでエピソード生成を並列実行（教師あり学習済みモデルを使用）
        if DEBUG:
            print("Actorが改良されたエピソードを生成中...")
            print("※ PCNエージェントの_choose_commandsと_nlargestメソッドにより改善された目標値を使用")
        actor_futures = [actor.run.remote(n_episodes=EPISODES_PER_ITERATION, random_actions=False) for actor in actors]
        
        # Actorの並列実行完了を待機
        actor_results = ray.get(actor_futures)
        # print("Actorのエピソード生成完了")
        # print(f"  総エピソード数: {sum(actor_results)}")
        # print("  改良された解の選択〜実行完了時間の統計が各Actorで表示されました")
        
            # print(f"新規追加エピソード数: {buffer_size - pre_iteration_buffer_size}")
        
        # Learnerで学習を実行（改良された経験を使用）
        if DEBUG:
            print("Learnerが改良された経験で学習を実行中")
            print("※ _nlargestメソッドにより選択された上位エピソードを使用")
        loss = ray.get(learner.learn.remote(batch_size=BATCH_SIZE, n_updates=N_UPDATES))

        print(f"イテレーション {iteration + 1} 学習完了：平均損失: {loss:.4f}")
        
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
                e_returns, e_values, distances, map_fin = ray.get(learner.evaluate.remote(n=EVAL_SAMPLES))
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
            
            training_history['pareto_front_sizes'].append(len(e_returns))
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
                
                # フォントエラーの対処（グローバルインポートを使用）
                import matplotlib.pyplot as plt
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                
                # 現在の状態で評価を実行してパレートフロントを取得（サンプル数を削減）
                current_e_returns, current_e_values, current_distances, current_map_fin = ray.get(learner.evaluate.remote(n=EVAL_SAMPLES_FINAL))
                
                # 可視化時にもモデルを保存
                model_save_path = f"{save_dir}/model_visualization_{iteration + 1:03d}.pth"
                saved_model_path = ray.get(learner.save_model.remote(model_save_path))
                if saved_model_path and DEBUG:
                    print(f"可視化時のモデルを保存しました: {saved_model_path}")
                
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
                    plt.savefig(values_plot_path, dpi=150, bbox_inches='tight')  # dpiを下げる
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
    
    # 学習完了後の総括
    if DEBUG:
        print("\n" + "="*60)
        print("学習完了 - 総括")
        print("="*60)
        
        actual_iterations = len(training_history['iterations'])
        print(f"設定イテレーション数: {N_ITERATIONS}")
        print(f"実際の実行イテレーション数: {actual_iterations}")
        print(f"最終損失: {training_history['losses'][-1]:.4f}")
        
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
                plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
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
                plt.savefig(values_plot_path, dpi=300, bbox_inches='tight')
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
                plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
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
