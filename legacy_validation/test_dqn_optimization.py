#!/usr/bin/env python3
"""
DQNエージェントの最適化テスト
新しく追加した機能をテストするスクリプト
"""

import yaml
import numpy as np
import torch
from src.envs.scheduling_env import SchedulingEnv
from src.agents.dqn_agent import DQNAgent

def test_optimized_dqn():
    """最適化されたDQNエージェントのテスト"""
    print("最適化されたDQNエージェントのテスト開始")
    
    # 設定ファイル読み込み
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 環境の初期化
    env = SchedulingEnv(config)
    
    # 最適化されたDQNエージェントの初期化
    agent = DQNAgent(
        env=env,
        device="auto",
        state_dim=env.observation_space.shape[0],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,  # より緩やかな減衰
        buffer_size=50000,     # 大きなバッファ
        batch_size=256,
        hidden_dim=512,        # 大きなネットワーク
        target_update=5,       # 頻繁な更新
        weight_cost=config['param_agent']['weight_cost'],
        debug_mode=True,       # デバッグ情報表示
        lr_scheduler=True,     # 学習率スケジューリング
        double_dqn=True,       # Double DQN
        dueling_dqn=True       # Dueling DQN
    )
    
    print(f"\n設定された最適化機能:")
    print(f"- エピソード数: {config['param_simulation']['nb_episodes']}")
    print(f"- バッファサイズ: 50,000")
    print(f"- ネットワークサイズ: 512")
    print(f"- Double DQN: 有効")
    print(f"- Dueling DQN: 有効") 
    print(f"- 学習率スケジューリング: 有効")
    print(f"- 頻繁な更新: 有効")
    print(f"- 正則化: Dropout追加")
    
    # 短いテスト実行（100エピソード）
    print(f"\n短いテスト実行（100エピソード）...")
    test_episodes = 100
    losses = agent.train(
        num_episodes=test_episodes,
        early_stop_threshold=0.005,
        patience=50,
        min_episodes=20,
        record_interval=10
    )
    
    # 結果表示
    final_wt, final_cost = agent.get_final_values()
    best_wt, best_cost = agent.get_best_values()
    
    print(f"\n{'='*60}")
    print(f"テスト結果:")
    print(f"最終値 - Cost: {final_cost:.4f}, WT: {final_wt:.4f}")
    print(f"ベスト値 - Cost: {best_cost:.4f}, WT: {best_wt:.4f}")
    print(f"平均損失: {np.mean(losses[-10:]):.6f}" if losses else "N/A")
    print(f"{'='*60}")
    
    return agent

if __name__ == "__main__":
    test_optimized_dqn() 