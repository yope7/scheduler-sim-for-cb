import numpy as np
import torch as th
import yaml
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random

from src.agents.pcn_agent import PCN, Transition, DiscreteActionsDefaultModel
from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator

# シードを設定
random.seed(0)
np.random.seed(0)
th.manual_seed(0)
if th.cuda.is_available():
    th.cuda.manual_seed_all(0)

# 設定ファイルの読み込み
with open('config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

# 環境パラメータの設定
max_step = np.inf
n_window = config['param_env']['n_window']
n_on_premise_node = config['param_env']['n_on_premise_node']
n_cloud_node = config['param_env']['n_cloud_node']
n_job_queue_obs = config['param_env']['n_job_queue_obs']
n_job_queue_bck = config['param_env']['n_job_queue_bck']
penalty_not_allocate = config['param_env']['penalty_not_allocate']
penalty_invalid_action = config['param_env']['penalty_invalid_action']
weight_wt = config['param_agent']['weight_wt']
weight_cost = config['param_agent']['weight_cost']

# ジョブ生成
job_generator = JobGenerator(0, 1, n_window, n_on_premise_node, n_cloud_node, config, 100, 0.23, 1)
jobs_set = job_generator.generate_jobs_set()

# 環境の初期化
env = SchedulingEnv(
    max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
    weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
    None, flag=0
)



# モデルファイルの選択部分を対話型に変更
print("\n利用可能なモデルファイル:")
model_files = [f for f in os.listdir('weights') if f.endswith('.pt')]
for i, file in enumerate(model_files, 1):
    print(f"{i}. {file}")

while True:
    try:
        choice = int(input("\n使用するモデルファイルの番号を入力してください: "))
        if 1 <= choice <= len(model_files):
            model_path = model_files[choice-1]
            model_file = f"weights/{model_path}"
            break
        else:
            print("無効な番号です。もう一度入力してください。")
    except ValueError:
        print("数値を入力してください。")

print(f"\n選択されたモデル: {model_path}")
file_size = os.path.getsize(model_file) / (1024 * 1024)
print(f"モデルファイルのサイズ: {file_size:.2f} MB")

# エージェントの初期化
agent = PCN(
    env,
    device="cpu",
    state_dim=1,
    scaling_factor=np.array([1, 1, 1]),
    learning_rate=1e-2,
    batch_size=1024,
    hidden_dim=256,
    project_name="temp",
    experiment_name="PCN",
    log=False,
)

# モデルを直接読み込む（ロード方法を変更）
try:
    agent.model = th.load(model_file, map_location=agent.device)
    print(f"モデルを読み込みました: {model_file}")
    
    # モデルの基本情報を確認
    if hasattr(agent.model, 'action_dim'):
        print(f"モデルのアクション次元: {agent.model.action_dim}")
    
    # モデルの型の確認
    print(f"モデルのタイプ: {type(agent.model).__name__}")
    
    # モデルが評価モードになっていることを確認
    agent.model.eval()
except Exception as e:
    print(f"モデル読み込み中にエラーが発生しました: {e}")
    sys.exit(1)

# 行動確率を直接計算する関数
def get_action_probs(obs, desired_return, desired_horizon):
    with th.no_grad():
        obs_tensor = th.tensor(np.array([obs]), device=agent.device).float()
        return_tensor = th.tensor(np.array([desired_return]), device=agent.device).float()
        horizon_tensor = th.tensor([[desired_horizon]], device=agent.device).float()
        
        prediction = agent.model(obs_tensor, return_tensor, horizon_tensor)
        log_probs = prediction.detach()[0]
        probs = th.exp(log_probs).cpu().numpy()
        return probs, np.argmax(probs)

# テスト目的のパラメータを定義
test_desired_returns = np.array([
    np.linspace(-1000, 1000, 10),  # 1番目の目標値を-1000から1000まで5段階で設定
    np.linspace(-1000, 1000, 10)   # 2番目の目標値を-1000から1000まで5段階で設定
]).T.tolist()  # 転置して各ペアを作成

test_horizons = [1, 5, 10, 20, 50, 100]  # 様々なホライズン値

# すべての組み合わせをテスト
print("\n異なるパラメータでの行動予測テスト:")
obs = env.reset()

# 行動予測のテスト関数
def test_action_prediction(obs, desired_return, desired_horizon):
    with th.no_grad():
        obs_tensor = th.tensor(np.array([obs]), device=agent.device).float()
        return_tensor = th.tensor(np.array([desired_return]), device=agent.device).float()
        horizon_tensor = th.tensor([[desired_horizon]], device=agent.device).float()
        
        prediction = agent.model(obs_tensor, return_tensor, horizon_tensor)
        log_probs = prediction.detach()[0]
        probs = th.exp(log_probs).cpu().numpy()
        
        return probs, np.argmax(probs)

# テスト実行
for dr in test_desired_returns:
    for h in test_horizons:
        probs, max_action = test_action_prediction(obs, dr, h)
        
        print(f"\n目標報酬: {dr}, ホライズン: {h}")
        print(f"  最大確率の行動: {max_action} ({probs[max_action]:.6f})")
        print(f"  上位3行動の確率: {probs[np.argsort(-probs)[:3]]}")
        
        # 確率分布が均一かどうかチェック
        is_uniform = np.allclose(probs, probs[0], atol=1e-4)
        if is_uniform:
            print("  警告: すべての行動の確率がほぼ同じです！")

# パレートフロント生成のメインコード
print("\nパレートフロント生成を開始...")

# 目標報酬とホライズンの範囲を設定
import numpy as np

# -1000から1000までの対数スケールで20個の値を生成
reward_ranges = [
    np.linspace(-1000, 1000, 250).tolist(),  # 目標報酬1の候補
    np.linspace(-1000, 1000, 250).tolist()   # 目標報酬2の候補
]

horizon_values = [180]  # ホライズンの候補

# 全組み合わせの生成
param_combinations = []
for r1 in reward_ranges[0]:
    for r2 in reward_ranges[1]:
        for h in horizon_values:
            param_combinations.append(([r1, r2], h))

print(f"推論する組み合わせの総数: {len(param_combinations)}")

# 結果を保存する配列
results = []
action_distribution = {}  # 各行動の出現回数を記録

# 各組み合わせについて推論を実行
for i, (desired_return, horizon) in enumerate(tqdm(param_combinations)):
    # 目標報酬と期間を設定
    agent.set_desired_return_and_horizon(
        desired_return=np.array(desired_return, dtype=np.float32),
        desired_horizon=horizon
    )
    
    # 環境をリセット
    obs = env.reset()
    done = False
    actions_taken = []
    
    # エピソードを実行
    step_count = 0
    while not done:
        # 行動の選択
        action = agent.eval(obs)
        actions_taken.append(action)
        
        # 行動の出現回数を記録
        if action not in action_distribution:
            action_distribution[action] = 0
        action_distribution[action] += 1
        
        # 行動の実行
        n_obs, reward, scheduled, wt_step, done = env.step(action)
        
        if done:
            env.finalize_window_history()
        
        obs = n_obs
        step_count += 1
        
    
    # 結果の計算
    cost, makespan = env.calc_objective_values()
    
    # 結果を保存
    results.append({
        'desired_return': desired_return,
        'horizon': horizon,
        'cost': cost,
        'makespan': makespan,
        'actions': actions_taken,
        'steps': step_count
    })

# 行動分布の分析
print("\n行動分布の統計:")
total_actions = sum(action_distribution.values())
sorted_actions = sorted(action_distribution.items(), key=lambda x: x[1], reverse=True)
for action, count in sorted_actions:
    print(f"行動 {action}: {count}回 ({count/total_actions*100:.2f}%)")

# 結果の多様性を確認
unique_costs = set([r['cost'] for r in results])
unique_makespans = set([r['makespan'] for r in results])
print(f"\n異なるコスト値の数: {len(unique_costs)}")
print(f"異なる待ち時間値の数: {len(unique_makespans)}")

# パレートフロントを計算
def is_pareto_efficient(costs):
    """
    パレート効率的な解のインデックスを見つける
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # 現在の点よりも両方の次元で良いか同等な点を見つける
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True  # 自分自身は維持
    return is_efficient

# コストと待ち時間の両方が小さいほど良いので、これを考慮したパレート計算
costs_makespans = np.array([(r['cost'], r['makespan']) for r in results])
pareto_mask = is_pareto_efficient(costs_makespans)

# パレートフロントの点を抽出
pareto_results = [results[i] for i, mask in enumerate(pareto_mask) if mask]
print(f"\nパレートフロント上の点の数: {len(pareto_results)}")

# パレートフロントの詳細表示
print("\nパレートフロント上の点の詳細:")
for i, res in enumerate(pareto_results):
    print(f"点 {i+1}:")
    print(f"  目標報酬: {res['desired_return']}")
    print(f"  ホライズン: {res['horizon']}")
    print(f"  コスト: {res['cost']:.2f}")
    print(f"  待ち時間: {res['makespan']:.2f}")
    print(f"  ステップ数: {res['steps']}")
    print(f"  最初の3行動: {res['actions'][:3]}")

# 可視化：パレートフロント
plt.figure(figsize=(12, 10))
all_costs = [r['cost'] for r in results]
all_makespans = [r['makespan'] for r in results]
pareto_costs = [r['cost'] for r in pareto_results]
pareto_makespans = [r['makespan'] for r in pareto_results]

plt.scatter(all_costs, all_makespans, c='lightgray', s=50, alpha=0.5, label='All')
plt.scatter(pareto_costs, pareto_makespans, c='red', s=100, alpha=0.8, label='PF')

# パレートフロントに線を引く
if len(pareto_costs) > 1:
    pareto_indices = np.argsort(pareto_costs)
    plt.plot(np.array(pareto_costs)[pareto_indices], np.array(pareto_makespans)[pareto_indices], 'r--', linewidth=2)

plt.xlabel('Cost', fontsize=14)
plt.ylabel('Makespan', fontsize=14)
plt.title('Pareto Front by PCN', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('pcn_pareto_front_with_horizon.png', dpi=300)

# ヒートマップ: 3次元のパラメータ空間をどう可視化するか
# ここでは、目標報酬1とホライズンの関係を示す例
plt.figure(figsize=(14, 8))
horizon_groups = {}
for res in results:
    h = res['horizon']
    if h not in horizon_groups:
        horizon_groups[h] = {'dr1': [], 'dr2': [], 'cost': [], 'makespan': []}
    
    horizon_groups[h]['dr1'].append(res['desired_return'][0])
    horizon_groups[h]['dr2'].append(res['desired_return'][1])
    horizon_groups[h]['cost'].append(res['cost'])
    horizon_groups[h]['makespan'].append(res['makespan'])

# 特定のホライズン値についてプロット
for i, h in enumerate([1, 10, 50]):
    if h in horizon_groups:
        plt.subplot(1, 3, i+1)
        plt.scatter(horizon_groups[h]['dr1'], horizon_groups[h]['makespan'], 
                   c=horizon_groups[h]['dr2'], cmap='viridis', alpha=0.7)
        plt.colorbar(label='Target Return 2')
        plt.xlabel('Target Return 1', fontsize=12)
        plt.ylabel('Makespan', fontsize=12)
        plt.title(f'Horizon = {h}', fontsize=14)
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pcn_horizon_analysis.png', dpi=300)

# パレート解をCSVに保存
pareto_data = pd.DataFrame([
    {
        'DesiredReturn1': res['desired_return'][0],
        'DesiredReturn2': res['desired_return'][1],
        'Horizon': res['horizon'],
        'Cost': res['cost'],
        'Makespan': res['makespan'],
        'Steps': res['steps']
    }
    for res in pareto_results
])
pareto_data.to_csv('pcn_pareto_solutions_with_horizon.csv', index=False)
print("パレート解のデータをCSVファイルに保存しました: pcn_pareto_solutions_with_horizon.csv")

print("\n実験完了！")
