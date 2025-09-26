import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Optional, Union, List
from dataclasses import dataclass

@dataclass
class DQNTransition:
    observation: np.ndarray
    action: int
    reward: float  # スカラー報酬
    next_observation: np.ndarray
    terminal: bool
    
    def __getitem__(self, index):
        """インデックスアクセスを属性アクセスにマッピング"""
        if index == 0:
            return self.observation
        elif index == 1:
            return self.action
        elif index == 2:
            return self.reward
        elif index == 3:
            return self.next_observation
        elif index == 4:
            return self.terminal
        else:
            raise IndexError("DQNTransition index out of range")
    
    def to_tensor_batch(self, batch):
        """バッチデータを効率的にtensorに変換"""
        # numpy配列を事前に結合してからtensorに変換
        observations = np.array([t.observation for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_observations = np.array([t.next_observation for t in batch])
        terminals = np.array([t.terminal for t in batch])
        
        return (
            torch.FloatTensor(observations),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_observations),
            torch.BoolTensor(terminals)
        )

class DQNNetwork(nn.Module):
    """シンプルなDQNネットワーク"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # シンプルな3層ネットワーク
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """ネットワークの重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前方伝播"""
        return self.network(x)

class DQNAgent:
    """シンプルなDQNエージェント"""
    
    def __init__(
        self,
        env,
        device: Union[torch.device, str] = "auto",
        state_dim: int = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 256,
        hidden_dim: int = 1024,
        target_update: int = 5,
        weight_id: Optional[str] = None,
        weight_cost: float = 0.0,
    ):
        """DQNエージェントの初期化"""
        # 基本設定
        self.env = env
        
        # デバイス設定
        self._setup_device(device)
        
        # 状態次元の設定
        if state_dim is None:
            try:
                initial_state = env.reset()
                if hasattr(initial_state, '__len__'):
                    self.state_dim = len(initial_state)
                elif hasattr(initial_state, 'shape'):
                    self.state_dim = initial_state.shape[0] if len(initial_state.shape) > 0 else 1
                else:
                    self.state_dim = 1
                env.reset()
            except Exception as e:
                self.state_dim = 100
        else:
            self.state_dim = state_dim
        
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim
        
        # ネットワークの初期化
        self._initialize_networks()
        
        # 最適化器
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # リプレイバッファ
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        
        # ハイパーパラメータ
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        #重み
        self.weight_id = weight_id
        self.weight_cost = weight_cost
        self.weight_wt = 1 - weight_cost
        
        # 重みの設定と確認
        self.weight_id = weight_id
        self.weight_cost = weight_cost
        self.weight_wt = 1 - weight_cost
        
        # 重み設定の確認ログ
        print(f"DQN Agent initialized with weights: WT={self.weight_wt:.3f}, Cost={self.weight_cost:.3f}")
        
        # 重みの妥当性チェック
        if not (0 <= self.weight_wt <= 1) or not (0 <= self.weight_cost <= 1):
            print(f"Warning: Invalid weights detected: WT={self.weight_wt}, Cost={self.weight_cost}")
        
        if abs(self.weight_wt + self.weight_cost - 1.0) > 1e-6:
            print(f"Warning: Weights don't sum to 1: {self.weight_wt + self.weight_cost}")
        
        # 学習統計
        self.global_step = 0
        self.episode_count = 0
    
    def _setup_device(self, device: Union[torch.device, str]):
        """デバイス設定"""
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
    
    def _initialize_networks(self):
        """ネットワーク初期化"""
        self.policy_net = DQNNetwork(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim
        ).to(self.device)
        
        # ターゲットネットワークの重みをコピー
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state: np.ndarray) -> int:
        """行動選択（ε-greedy戦略）"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state: np.ndarray, action: int, reward: Union[List[float],np.ndarray], 
                        next_state: np.ndarray, done: bool):
        """トランジション保存（重み付け報酬を正しく統合）"""
        # データ型を統一
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        if isinstance(reward, (list, np.ndarray)):
            # 環境から受け取った報酬を負の値に変換（最小化問題を最大化問題に変換）
            negative_reward = [-reward[0], -reward[1]]
            
            # 重み付けしてスカラー値に統合
            # これにより、costのみ重んじる場合はcostの影響が強く反映される
            scalar_reward = negative_reward[0] * self.weight_wt + negative_reward[1] * self.weight_cost
            
            # デバッグ用（重み付けの確認）
            # if self.global_step % 100 == 0:
            #     print(f"Step {self.global_step}: Original reward: {reward}, "
            #           f"Weighted reward: {scalar_reward:.4f} "
            #           f"(wt: {self.weight_wt:.3f}, cost: {self.weight_cost:.3f})")
        else:
            scalar_reward = reward
        
        # トランジション作成（スカラー報酬）
        transition = DQNTransition(
            observation=state, 
            action=action, 
            reward=scalar_reward,  # スカラー値
            next_observation=next_state, 
            terminal=done
        )
        
        self.memory.append(transition)

    def update(self) -> Optional[float]:
        """ネットワーク更新（シンプルな最適化版）"""
        if len(self.memory) < self.batch_size:
            return None
        
        try:
            # 1. バッチサンプリング
            batch = random.sample(self.memory, self.batch_size)

            # 2. バッチデータの準備（numpy配列を事前に結合）
            observations = np.array([t.observation for t in batch])
            actions = np.array([t.action for t in batch])
            rewards = np.array([t.reward for t in batch])
            next_observations = np.array([t.next_observation for t in batch])
            terminals = np.array([t.terminal for t in batch])
            
            # 3. tensorに変換してデバイスに移動
            states = torch.FloatTensor(observations).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_observations).to(self.device)
            dones = torch.BoolTensor(terminals).to(self.device)

            # 3. 現在のQ値の計算
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # 4. 次のQ値の計算
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

            # 5. 損失計算
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # 6. 最適化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 7. ε減衰
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # 8. ターゲットネットワーク更新
            if self.global_step % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.global_step += 1
            
            # # デバッグ情報（頻度を下げる）
            # if self.global_step % 500 == 0:
            #     print(f"Step {self.global_step}: Loss={loss.item():.6f}, "
            #           f"Current Q={current_q_values.mean().item():.6f}, "
            #           f"Target Q={target_q_values.mean().item():.6f}, "
            #           f"Rewards mean={rewards.mean().item():.6f}")
            
            return loss.item()
            
        except Exception as e:
            print(f"DQN update error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train(self, num_episodes: int) -> List[float]:
        import os
        import datetime
        
        log_dir = "dqn_logs4"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{log_dir}/dqn_training_{self.weight_id}_{timestamp}.log"

        print(f"Log file: {log_filename}")
        
        """学習実行"""
        losses = []
        
        for episode in range(num_episodes):
            episode_loss = self._run_episode()
            self.env.finalize_window_history()
            value_cost, _, value_wt = self.env.calc_objective_values()
            
            if episode_loss is not None:
                losses.append(episode_loss)
            
            # 進捗表示
            if episode % 5 == 0 or episode == num_episodes - 1:
                # print(f"Episode {episode}/{num_episodes}, value_cost: {value_cost:.2f}, value_wt: {value_wt:.2f}")
                with open(log_filename, 'a') as f:
                    f.write(f"Episode {episode}/{num_episodes}, value_cost: {value_cost:.2f}, value_wt: {value_wt:.2f}\n")
                    f.flush()
        
        return losses
    
    def _run_episode(self) -> Optional[float]:
        """1エピソードの実行（学習進捗の詳細確認）"""
        state = self.env.reset()
        episode_losses = []
        episode_rewards = []
        done = False
        step_count = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, scheduled, wt_step, done = self.env.step(action)
            
            # 重み付け前の報酬を記録
            episode_rewards.append(reward)
            
            # 重み付けしてスカラー値に統合
            if isinstance(reward, (list, np.ndarray)):
                scalar_reward = -reward[0] * self.weight_wt + -reward[1] * self.weight_cost
            else:
                scalar_reward = reward
            
            self.store_transition(state, action, reward, next_state, done)
            
            if len(self.memory) >= self.batch_size:
                loss = self.update()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            step_count += 1
        
        # # エピソード終了時の統計
        # if episode_rewards:
        #     avg_reward = np.mean([r[1] if isinstance(r, (list, np.ndarray)) else r for r in episode_rewards])
        #     print(f"Episode {self.episode_count}: Steps={step_count}, "
        #           f"Avg Cost Reward={avg_reward:.4f}, "
        #           f"Epsilon={self.epsilon:.4f}")
        
        self.episode_count += 1
        return np.mean(episode_losses) if episode_losses else None