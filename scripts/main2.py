import time
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os
from src.envs.scheduling_env import SchedulingEnv
from src.agents.pcn_agent import PCN

def benchmark_with_pcn_agent():
    """実際のPCNエージェントを使ったベンチマーク"""
    print("Starting PCN agent benchmark...")
    
    # 環境作成
    env = SchedulingEnv(
        max_step=100, n_window=10, n_on_premise_node=6, n_cloud_node=6,
        n_job_queue_obs=10, n_job_queue_bck=10, weight_wt=0.5, weight_cost=0.5,
        penalty_not_allocate=0.1, penalty_invalid_action=0.5,
        jobs_set=[np.random.randint(1, 5, size=(100, 8)) for _ in range(10)]
    )
    
    # PCNエージェントを初期化（ハイパーパラメータは元のコードに合わせる）
    scaling_factor = np.array([1.0, 1.0, 1.0])
    agent = PCN(
        env=env,
        scaling_factor=scaling_factor,
        device="cpu",
        learning_rate=0.001,
        gamma=0.99,
        batch_size=64,
        hidden_dim=64,
        log=False,
    )
    
    # 保存済みモデルを読み込む
    agent.load("PCN_model", "weights")
    
    # パレートフロント生成のベンチマーク
    num_points = 20
    desired_returns = []
    
    # 異なる望ましい報酬を生成
    for i in range(num_points):
        r1 = 10.0 - i * 20.0 / (num_points - 1)
        r2 = -i * 20.0 / (num_points - 1)
        desired_returns.append([r1, r2])
    
    # 推論時間計測
    start_time = time.time()
    
    actions = []
    for desired_return in desired_returns:
        # エージェントの期待する形式で望ましい報酬を設定
        agent.set_desired_return_and_horizon(
            desired_return=np.array(desired_return, dtype=np.float32),
            desired_horizon=50.0
        )
        
        # 観測を取得して行動を決定
        obs = env.reset()
        action = agent.eval(obs)
        actions.append(action)
    
    inference_time = time.time() - start_time
    per_point_time = inference_time / num_points * 1000  # ms/point
    
    print(f"Inference completed in {inference_time:.6f} seconds ({per_point_time:.2f} ms/point)")
    
    # 結果をプロット（エラーなく実行できるように簡略化）
    plt.figure(figsize=(10, 8))
    plt.bar(['PCN Inference Time'], [inference_time])
    plt.ylabel('Time (seconds)')
    plt.title(f'PCN Inference Time for {num_points} Pareto Points')
    
    info_text = f"Average time per point: {per_point_time:.2f} ms"
    plt.annotate(info_text, xy=(0.5, inference_time/2), ha='center')
    
    plt.savefig('pcn_inference_benchmark.png')
    print(f"Figure saved as pcn_inference_benchmark.png")

if __name__ == "__main__":
    benchmark_with_pcn_agent()
