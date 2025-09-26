#!/usr/bin/env python3
"""
シンプルなDQNの動作確認用テストスクリプト
"""

import numpy as np
import torch
from src.agents.dqn_agent import DQNAgent

class SimpleTestEnv:
    """シンプルなテスト環境"""
    
    def __init__(self, state_dim=10, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = type('ActionSpace', (), {'n': action_dim, 'sample': lambda: np.random.randint(0, action_dim)})()
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self):
        """環境をリセット"""
        self.current_step = 0
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def step(self, action):
        """環境を1ステップ進める"""
        self.current_step += 1
        
        # ランダムな報酬
        reward = np.random.randn(2)  # 2次元の報酬
        
        # ランダムな次状態
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        
        # 終了判定
        done = self.current_step >= self.max_steps
        
        # その他の情報
        scheduled = True
        wt_step = 0.0
        
        return next_state, reward, scheduled, wt_step, done

def test_simple_dqn():
    """シンプルなDQNのテスト"""
    print("シンプルなDQNのテスト開始")
    
    # 環境の作成
    env = SimpleTestEnv(state_dim=10, action_dim=4)
    print(f"環境作成完了: 状態次元={env.state_dim}, 行動次元={env.action_dim}")
    
    # DQNエージェントの作成
    agent = DQNAgent(
        env=env,
        device="auto",
        state_dim=None,  # 自動取得
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=1000,
        batch_size=32,
        hidden_dim=64,
        target_update=5
    )
    print(f"DQNエージェント作成完了: デバイス={agent.device}")
    
    # 学習の実行
    print("学習開始...")
    losses = agent.train(num_episodes=100)
    print(f"学習完了: {len(losses)}エピソード完了")
    
    # 結果の表示
    if losses:
        print(f"最終損失: {losses[-1]:.6f}")
        print(f"平均損失: {np.mean(losses):.6f}")
        print(f"最小損失: {np.min(losses):.6f}")
        print(f"最大損失: {np.max(losses):.6f}")
    
    # テスト実行
    print("\nテスト実行...")
    test_state = np.random.randn(10).astype(np.float32)
    action = agent.select_action(test_state)
    print(f"テスト状態での行動選択: {action}")
    
    print("テスト完了！")

if __name__ == "__main__":
    test_simple_dqn() 