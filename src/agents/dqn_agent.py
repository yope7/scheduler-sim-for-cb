import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

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
        weight_wt=0.7,    # 待ち時間の重み
        weight_cost=0.3   # コストの重み
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
        self.weight_wt = weight_wt
        self.weight_cost = weight_cost

        # 報酬の管理
        self.total_wt_reward = 0
        self.total_cost_reward = 0

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

    def train(self, num_episodes, early_stop_threshold=0.01, patience=50, min_episodes=100):
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        avg_loss = float('inf')
        min_loss_improvement = 0.001  # 最小改善閾値
        window_size = 10  # 移動平均のウィンドウサイズ
        
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
                                break

            if episode % 10 == 0:
                print(f"\nEpisode {episode}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Waiting Time Reward: {self.total_wt_reward:.2f}")
                print(f"Cost Reward: {self.total_cost_reward:.2f}")
                if episode_losses:
                    print(f"Average Loss: {avg_loss:.4f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                if len(losses) >= window_size:
                    print(f"Loss Moving Average: {current_avg:.4f}")

        return losses
    
    def get_rewards(self):
        return self.total_wt_reward, self.total_cost_reward