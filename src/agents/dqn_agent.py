import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from morl_baselines.common.pareto import get_non_dominated_inds

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(
        self,
        env,
        device="auto",
        state_dim=1,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=128,
        batch_size=64,
        hidden_dim=256,
        target_update=10,
        weight_cost=0.0,
        weight_id=None  # 重みのID（可視化用）
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        self.state_dim = state_dim
        self.action_dim = env.action_space.n
        
        # ネットワークの初期化
        self.policy_net = DQNNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=buffer_size)
        
        # ハイパーパラメータ
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        self.weight_wt = 1-weight_cost
        self.weight_cost = weight_cost
        self.weight_id = weight_id if weight_id is not None else f"wt{self.weight_wt:.2f}_cost{self.weight_cost:.2f}"

        # 報酬の管理
        self.total_wt_reward = 0
        self.total_cost_reward = 0
        
        # 学習履歴の記録用（新規追加）
        self.training_history = {
            'episodes': [],     # エピソード番号
            'wt_values': [],    # 待ち時間の値
            'cost_values': [],  # コストの値
            'wt_rewards': [],   # 待ち時間の報酬
            'cost_rewards': [], # コストの報酬
            'losses': []        # 損失値
        }
        
        # 評価用の最良値を初期化
        self.best_wt = float('inf')
        self.best_cost = float('inf')

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        # 重み付けした報酬の計算
        weighted_reward = self.weight_wt * reward[0] + self.weight_cost * reward[1]
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, weighted_reward, next_state, done))

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # バッチのサンプリングとデータの準備
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # NumPyの配列に変換してからTensorに変換
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)

        # Q値の計算
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # 損失の計算と最適化
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 勾配クリッピングを追加
        self.optimizer.step()

        # εの減衰
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # ターゲットネットワークの更新
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps += 1
        return loss.item()

    def train(self, num_episodes, early_stop_threshold=0.01, patience=50, min_episodes=100, 
              record_interval=10):
        """
        DQNエージェントの学習
        
        Args:
            num_episodes: 学習エピソード数
            early_stop_threshold: 早期終了の閾値
            patience: 早期終了の我慢回数
            min_episodes: 最小エピソード数
            record_interval: 履歴を記録する間隔（エピソード数）
        """
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        avg_loss = float('inf')
        min_loss_improvement = 0.001  # 最小改善閾値
        window_size = 10  # 移動平均のウィンドウサイズ
        
        # 学習履歴の記録をリセット
        self.training_history = {
            'episodes': [],
            'wt_values': [],
            'cost_values': [],
            'wt_rewards': [],
            'cost_rewards': [],
            'losses': []
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            self.total_wt_reward = 0
            self.total_cost_reward = 0
            done = False
            episode_steps = 0
            episode_losses = []

            while not done:
                action = self.select_action(state)
                next_state, reward, scheduled, wt_step, done = self.env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    if loss is not None:
                        episode_losses.append(loss)
                
                state = next_state
                total_reward += reward[0] + reward[1]
                self.total_wt_reward += reward[0]
                self.total_cost_reward += reward[1]
                episode_steps += 1
                
            self.env.finalize_window_history()
            value_cost, value_wt = self.env.calc_objective_values()
            
            # 最良値の更新
            self.best_wt = min(self.best_wt, value_wt)
            self.best_cost = min(self.best_cost, value_cost)

            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                losses.append(avg_loss)
                
                # 移動平均の計算
                if len(losses) >= window_size:
                    current_avg = np.mean(losses[-window_size:])
                    if len(losses) >= window_size * 2:
                        prev_avg = np.mean(losses[-window_size*2:-window_size])
                        improvement = (prev_avg - current_avg) / prev_avg
                        
                        # 早期終了の条件チェック
                        if episode >= min_episodes:  # 最小エピソード数を超えている
                            if improvement < min_loss_improvement:  # 改善が閾値未満
                                patience_counter += 1
                            else:
                                patience_counter = 0
                                
                            if patience_counter >= patience:
                                print(f"\nEarly stopping at episode {episode}")
                                print(f"Loss improvement below threshold: {improvement:.6f}")
                                
                                # 履歴を記録
                                if episode % record_interval == 0 or episode == num_episodes - 1:
                                    self._record_history(episode, value_wt, value_cost, 
                                                        self.total_wt_reward, self.total_cost_reward, 
                                                        avg_loss if episode_losses else None)
                                break

            # 学習経過の記録（一定間隔）
            if episode % record_interval == 0 or episode == num_episodes - 1:
                self._record_history(episode, value_wt, value_cost, 
                                    self.total_wt_reward, self.total_cost_reward, 
                                    avg_loss if episode_losses else None)

            if episode % 10 == 0:
                print(f"\nEpisode {episode}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Waiting Time Reward: {self.total_wt_reward:.2f}")
                print(f"Cost Reward: {self.total_cost_reward:.2f}")
                print(f"Cost: {value_cost:.2f}")
                print(f"Waiting Time: {value_wt:.2f}")
                if episode_losses:
                    print(f"Average Loss: {avg_loss:.4f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                if len(losses) >= window_size:
                    print(f"Loss Moving Average: {current_avg:.4f}")

        return losses
    
    def _record_history(self, episode, wt, cost, wt_reward, cost_reward, loss):
        """学習経過を記録"""
        self.training_history['episodes'].append(episode)
        self.training_history['wt_values'].append(wt)
        self.training_history['cost_values'].append(cost)
        self.training_history['wt_rewards'].append(wt_reward)
        self.training_history['cost_rewards'].append(cost_reward)
        if loss is not None:
            self.training_history['losses'].append(loss)
    
    def get_rewards(self):
        return self.total_wt_reward, self.total_cost_reward
    
    def get_training_history(self):
        """学習履歴を取得"""
        return self.training_history
    
    def get_best_values(self):
        """最良の待ち時間とコストを取得"""
        return self.best_wt, self.best_cost
    
    def get_final_values(self):
        """最終的な待ち時間とコストを取得"""
        if len(self.training_history['wt_values']) > 0 and len(self.training_history['cost_values']) > 0:
            return (self.training_history['wt_values'][-1], 
                    self.training_history['cost_values'][-1])
        return None, None

    def train_with_history(self, num_episodes, early_stop_threshold=0.01, patience=50, min_episodes=100, record_interval=10):
        """学習履歴を記録するバージョンのtrain関数"""
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        avg_loss = float('inf')
        min_loss_improvement = 0.001
        window_size = 10
        
        # 学習履歴の初期化
        training_history = {
            'wt_values': [],    # 待ち時間の値
            'cost_values': [],  # コストの値
            'episodes': []      # 対応するエピソード番号
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            self.total_wt_reward = 0
            self.total_cost_reward = 0
            done = False
            episode_steps = 0
            episode_losses = []

            while not done:
                action = self.select_action(state)
                next_state, reward, scheduled, wt_step, done = self.env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    if loss is not None:
                        episode_losses.append(loss)
                
                state = next_state
                total_reward += reward[0] + reward[1]
                self.total_wt_reward += reward[0]
                self.total_cost_reward += reward[1]
                episode_steps += 1
            
            self.env.finalize_window_history()
            cost, makespan = self.env.calc_objective_values()
            
            # 学習経過の記録（一定間隔）
            if episode % record_interval == 0:
                training_history['episodes'].append(episode)
                training_history['wt_values'].append(makespan)
                training_history['cost_values'].append(cost)

            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                losses.append(avg_loss)
                
                # 移動平均の計算
                if len(losses) >= window_size:
                    current_avg = np.mean(losses[-window_size:])
                    if len(losses) >= window_size * 2:
                        prev_avg = np.mean(losses[-window_size*2:-window_size])
                        improvement = (prev_avg - current_avg) / prev_avg
                        
                        # 早期終了の条件チェック
                        if episode >= min_episodes:
                            if improvement < min_loss_improvement:
                                patience_counter += 1
                            else:
                                patience_counter = 0
                            
                            if patience_counter >= patience:
                                print(f"\nEarly stopping at episode {episode}")
                                print(f"Loss improvement below threshold: {improvement:.6f}")
                                break

            if episode % 10 == 0:
                print(f"\nEpisode {episode}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Waiting Time Reward: {self.total_wt_reward:.2f}")
                print(f"Cost Reward: {self.total_cost_reward:.2f}")
                print(f"Cost: {cost:.2f}")
                print(f"Waiting Time: {makespan:.2f}")
                if episode_losses:
                    print(f"Average Loss: {avg_loss:.4f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                if len(losses) >= window_size:
                    print(f"Loss Moving Average: {current_avg:.4f}")

        return losses, training_history