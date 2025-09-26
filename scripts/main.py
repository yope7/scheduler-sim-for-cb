import numpy as np
import yaml
import sys
import argparse
import itertools
import matplotlib.pyplot as plt

from src.agents.pcn_agent import PCN  
from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.map_visualizer import visualize_map
from src.agents.dqn_agent import DQNAgent
from numba import jit
from src.agents.all_agent import ExhaustiveSearchAgent
from src.agents.all_agent_distributed import ExhaustiveSearchAgentDistributed
np.set_printoptions(linewidth=np.inf) 
import itertools
from morl_baselines.common.pareto import get_non_dominated_inds, get_non_dominated_inds_minimize
from src.agents.nsga2_agent import NSGA2Agent
from src.agents.nsga2_agent_distributed import DistributedNSGA2Agent
from multiprocessing import Pool
import os
# os.environ[“RAY_DEDUP_LOGS”] = “0”
import ray
import json
import datetime


with open('config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)
# 環境のパラメータをUUU
max_step = np.inf
n_window = config['param_env']['n_window']
n_on_premise_node = config['param_env']['n_on_premise_node']
n_cloud_node = config['param_env']['n_cloud_node']
n_job_queue_obs = config['param_env']['n_job_queue_obs']
n_job_queue_bck = config['param_env']['n_job_queue_bck']
penalty_not_allocate = config['param_env']['penalty_not_allocate']  # 割り当てない(一時キューに格納する)という行動を選択した際のペナルティー
penalty_invalid_action = config['param_env']['penalty_invalid_action']  # actionが無効だった場合のペナルティー
n_action = 2
weight_wt = config['param_agent']['weight_wt']
weight_cost = config['param_agent']['weight_cost']
nb_steps = config['param_simulation']['nb_steps']
nb_episodes = config['param_simulation']['nb_episodes']
nb_max_episode_steps = config['param_simulation']['nb_max_episode_steps'] # 1エピソードあたりの最大ステップ数(-1:最大ステップ無し)
if nb_max_episode_steps == -1:
    nb_max_episode_steps = np.inf
multi_algorithm = config['param_simulation']['multi_algorithm'] # 0:single algorithm(後でジョブを決める) 1:multi algorithm

# デバッグフラグ
DEBUG_MODE = False
# DEBUG_MODE = True

def run_debug_episode(env):
    """デバッグ用に1エピソードを実行し、スケジューリング結果を表示"""
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # ランダムな行動を選択
        obs, rewards, is_fifo, wt_step, std_mean_after, done, _ = env.step(action)

    env.render_map()
    # env.show_final_window_history()
    sys.exit()  # コメントアウト

def evaluate_and_render(agent, env, objective_index):
    """指定されたob番号で評価とレンダリングを行う"""
    print(f"training finished, start evaluation {objective_index}...")

    _, values = agent.evaluate_and_execute_selected_policy(
        env, 
        max_return=np.full(2, 100.0, dtype=np.float32), 
        objective_index=objective_index, 
        n=10
    )
    # print(next_init_windows)
    print(f"evaluation finished, start rendering {objective_index}...")

    agent.env.render_map(f"ob{objective_index}")

    return values


    # Define environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', 
                      choices=['single', 'pareto', 'pareto_distributed', 'pcn', 'all', 'all_distributed', 'nsga2', 'nsga2_distributed', 'heuristic'], 
                      help='実行モード（single: 単一重み, pareto: パレートフロント探索, pareto_distributed: 分散パレートフロント探索, pcn: PCN, all: 全探索, all_distributed: 分散処理による全探索（20ジョブ以上はサンプリング）, nsga2: NSGA-II, nsga2_distributed: 分散処理によるNSGA-II, heuristic: ヒューリスティック）')
    # 既存の引数
    parser.add_argument('--how_many_episodes', type=int, default=1000)
    parser.add_argument('--nb_jobs', type=int, default=11)
    # CNN関連の引数
    parser.add_argument('--use_cnn', action='store_true', 
                      help='PCNモードでCNNベースの拡張モデルを使用する')
    parser.add_argument('--use_wandb', action='store_true', 
                      help='Wandbを使用するかどうかを設定')
    # NSGA-II用の引数
    parser.add_argument('--pop_size', type=int, default=100, help='NSGA-IIの集団サイズ')
    parser.add_argument('--num_generations', type=int, default=300, help='NSGA-IIの世代数')
    parser.add_argument('--load_model', action='store_true', help='保存済みモデルを読み込む')
    parser.add_argument('--model_path', type=str, default='PCN_model_latest', 
                      help='読み込むモデルのパス（拡張子なし）')
    # 分散処理版NSGA-II用の引数
    parser.add_argument('--num_workers', type=int, default=32, help='分散処理のワーカー数')
    parser.add_argument('--migration_interval', type=int, default=10, help='解の移住間隔')
    
    # 分散パレート探索用の引数
    parser.add_argument('--weight_steps', type=int, default=10, help='パレート探索の重み分割数')
    
    # ヒューリスティック用の引数
    parser.add_argument('--base_threshold', type=int, default=5, help='基本待ち時間閾値')
    parser.add_argument('--width_factor', type=float, default=0.3, help='ジョブ幅の影響度（0.3 = 幅の30%）')
    
    return parser.parse_args()

def run_single_rl_mode(nb_steps: int, lams: list, loops: int, how_many_episodes: int, 
                       ob_number: int, nb_jobs: int) -> list:
    next_init_windows = None
    values_all = []
    np.random.seed(0)
    job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                               n_cloud_node, config, nb_jobs, 0.2, how_many_episodes)
    jobs_set = job_generator.generate_jobs_set()

    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows, flag=0
    )

    # 観測空間のサイズを取得
    obs_space_size = (12790)

    agent = DQNAgent(
        env,
        device="auto",
        state_dim=obs_space_size,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        hidden_dim=256,
        target_update=10
    )

    losses = agent.train(how_many_episodes)
    
    # 損失履歴をプロット
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss History')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    return 0

def run_PCN_mode(nb_steps: int, lams: list, loops: int, how_many_episodes: int, 
                ob_number: int, nb_jobs: int, use_cnn: bool = True, use_wandb: bool = False, load_model: bool = False, model_path: str = 'PCN_model_latest') -> list:
    """PCN強化学習モードの実行"""
    next_init_windows = None
    values_all = []
    np.random.seed(0)
    job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                               n_cloud_node, config, nb_jobs, 0.2, how_many_episodes)
    jobs_set = job_generator.generate_jobs_set()

    # print("jobs_set: ",jobs_set)

    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows, flag=0
    )

    # CNN拡張モデルを使用するかどうかを設定
    agent = PCN(
        env,
        device="auto",
        state_dim=1,
        scaling_factor=np.array([1, 1, 1]),
        learning_rate=1e-3,
        batch_size=512,
        hidden_dim=256,
        project_name="temp",
        experiment_name="PCN_Enhanced" if use_cnn else "PCN",
        log=True,
        use_wandb=use_wandb,  # CNNベースの拡張モデルを使用
        use_enhanced_model=use_cnn,  # CNNベースの拡張モデルを使用
        debug_mode=False,  # デバッグモードをオフに設定
    )

    # 保存済みモデルが指定されている場合は読み込む
    if load_model:
        print(f"保存済みモデル weights/{model_path} を読み込みます...")
        try:
            agent.load(filename=model_path, savedir="weights")
            print(f"モデルの読み込みが完了しました。学習を継続します。")
        except Exception as e:
            print(f"モデルの読み込み中にエラーが発生しました: {e}")
            print("新規モデルから学習を開始します。")

    agent.initialize_buffer_with_heuristics(env, num_episodes_per_pattern=50)
    
    agent.train(
        eval_env=env,
        total_timesteps=int(how_many_episodes),
        ref_point=np.array([-100000,-100000]),
        num_er_episodes=1000,
        num_step_episodes=20,  
        num_model_updates=10,
        num_eval_episodes=100,
        max_buffer_size=10000,
        known_pareto_front=[1, 1],
        max_return=np.array([1000, 1000]),
        log_episode_only=True,
        reset_buffer=False,  # ヒューリスティックで初期化されたバッファを保持する
    )

    on_premise_map, cloud_map = env.get_windows()
    # print(env.calc_objective_values())
    # print(agent.get_e_returns())

    agent.visualize_evaluation_history()

    return agent.get_mapmap()

def run_exhaustive_mode(nb_jobs: int):
    """全探索モードの実行"""
    # 環境パラメータの設定
    next_init_windows = None
    np.random.seed(0)
    job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                               n_cloud_node, config, nb_jobs, 0.2, 0)

    jobs_set = job_generator.generate_jobs_set()

    # 環境の初期化
    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows, flag=1
    )
    
    # エージェントの初期化と実行
    agent = ExhaustiveSearchAgent()
    search_results = agent.run_exhaustive_search(env, nb_jobs)
    
    # print("results: ", search_results['results'])
    print("pareto_front: ", search_results['pareto_front'])
    
    return search_results['results']

def run_exhaustive_mode_distributed(nb_jobs: int):
    """分散全探索モードの実行（20以上のジョブ数ではサンプリング）"""
    # 環境パラメータの設定

    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    
    print(f"=== ジョブ生成の詳細確認 ===")
    print(f"要求されたジョブ数: {nb_jobs}")
    print(f"nb_steps: {nb_steps}")
    print(f"n_window: {n_window}")
    print(f"n_on_premise_node: {n_on_premise_node}")
    print(f"n_cloud_node: {n_cloud_node}")
    
    job_generator = JobGenerator(
        0, 1,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, nb_jobs, 0.2, 0
    )
    
    jobs_set = job_generator.generate_jobs_set()
    
    print(f"生成されたjobs_setのキー: {list(jobs_set.keys())}")
    if jobs_set:
        first_episode = list(jobs_set.keys())[0]
        print(f"エピソード{first_episode}のジョブ数: {len(jobs_set[first_episode]) if jobs_set[first_episode] is not None else 'None'}")
        if jobs_set[first_episode] is not None and len(jobs_set[first_episode]) > 0:
            print(f"最初のジョブの構造: {jobs_set[first_episode]}")
            print(f"ジョブの型: {type(jobs_set[first_episode][0])}")
    
    # 環境の初期化
    env = SchedulingEnv(
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
    
    # エージェントの初期化と実行
    agent = ExhaustiveSearchAgentDistributed(num_workers=16)
    search_results = agent.run_exhaustive_search(env, nb_jobs)
    
    # 結果の表示
    print("pareto_front: ", search_results['pareto_front'])
    
    # サンプリング情報の表示
    if search_results['is_sampling']:
        print(f"\n=== サンプリング実行結果 ===")
        print(f"評価した組み合わせ数: {search_results['total_evaluated']:,}")
        print(f"総可能組み合わせ数: {search_results['total_possible']:,}")
        print(f"サンプリング率: {search_results['sampling_rate']*100:.4f}%")
    else:
        print(f"\n=== 全探索実行結果 ===")
        print(f"評価した組み合わせ数: {search_results['total_evaluated']:,}")
    
    return search_results['results']

def run_pareto_search(nb_steps: int, how_many_episodes: int, nb_jobs: int, weight_steps: int = 10):
    pareto_points_reward = []
    pareto_points_values = []
    training_histories = {}  # 各重みでの学習履歴を保存
    
    # 重みを生成
    weights = np.linspace(0, 1, weight_steps)
    
    for w_idx, w_wt in enumerate(weights):
        w_cost = 1 - w_wt
        weight_id = f"weight_{w_idx}"
        print(f"Training with weights - Waiting Time: {w_wt:.2f}, Cost: {w_cost:.2f}")
        
        # 環境の初期化
        np.random.seed(0)
        job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                                   n_cloud_node, config, nb_jobs, 0.2, how_many_episodes)
        jobs_set = job_generator.generate_jobs_set()
        
        env = SchedulingEnv(
            max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
            w_wt, w_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
            None, flag=0
        )
        
        # エージェントの初期化
        agent = DQNAgent(
            env,
            device="auto",
            state_dim=(12790),
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1024,
            batch_size=1024,
            hidden_dim=256,
            target_update=10,
            weight_wt=w_wt,
            weight_cost=w_cost,
            weight_id=weight_id
        )
        
        # 学習実行
        losses = agent.train(how_many_episodes)
        
        # 最終的な値を記録
        cost, _, wt = env.calc_objective_values()
        pareto_points_values.append((cost, wt))
        
        # このエージェントの学習履歴を保存
        training_histories[weight_id] = {
            'weight_wt': w_wt,
            'weight_cost': w_cost,
            'history': {
                'episodes': [how_many_episodes-1],
                'wt_values': [wt],
                'cost_values': [cost]
            }
        }
    
    # 学習過程を可視化
    visualize_training_progress(training_histories)
    
    return pareto_points_values

def visualize_training_progress(training_histories):
    """各重みでの学習進捗を可視化する関数"""
    plt.figure(figsize=(18, 15))
    
    # カラーマップの設定
    cmap = plt.cm.viridis
    num_weights = len(training_histories)
    
    # 重みパラメータでソート
    sorted_keys = sorted(training_histories.keys(), 
                         key=lambda k: training_histories[k]['weight_wt'])
    
    # サブプロット1: 各重みでの軌跡
    plt.subplot(2, 2, 1)
    for i, key in enumerate(sorted_keys):
        data = training_histories[key]
        w_wt = data['weight_wt']
        color = cmap(i / (num_weights - 1)) if num_weights > 1 else cmap(0.5)
        
        # 学習軌跡をプロット
        plt.plot(data['history']['cost_values'], data['history']['wt_values'], 
                 'o-', color=color, alpha=0.7, linewidth=1.5,
                 label=f'wt: {w_wt:.2f}, cost: {data["weight_cost"]:.2f}')
        
        # 最終点を強調
        plt.plot(data['history']['cost_values'][-1], data['history']['wt_values'][-1], 
                 'o', color=color, markersize=10)
    
    plt.xlabel('コスト', fontsize=14)
    plt.ylabel('待ち時間', fontsize=14)
    plt.title('各重みでの学習軌跡', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # サブプロット2: 待ち時間の推移
    plt.subplot(2, 2, 2)
    for i, key in enumerate(sorted_keys):
        data = training_histories[key]
        w_wt = data['weight_wt']
        color = cmap(i / (num_weights - 1)) if num_weights > 1 else cmap(0.5)
        
        plt.plot(data['history']['episodes'], data['history']['wt_values'], 
                 '-', color=color, linewidth=2, 
                 label=f'wt: {w_wt:.2f}')
    
    plt.xlabel('エピソード', fontsize=14)
    plt.ylabel('待ち時間', fontsize=14)
    plt.title('エピソードごとの待ち時間の変化', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # サブプロット3: コストの推移
    plt.subplot(2, 2, 3)
    for i, key in enumerate(sorted_keys):
        data = training_histories[key]
        w_wt = data['weight_wt']
        color = cmap(i / (num_weights - 1)) if num_weights > 1 else cmap(0.5)
        
        plt.plot(data['history']['episodes'], data['history']['cost_values'], 
                 '-', color=color, linewidth=2, 
                 label=f'wt: {w_wt:.2f}')
    
    plt.xlabel('エピソード', fontsize=14)
    plt.ylabel('コスト', fontsize=14)
    plt.title('エピソードごとのコストの変化', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # サブプロット4: パレートフロント
    plt.subplot(2, 2, 4)
    costs = []
    wts = []
    
    for key in sorted_keys:
        data = training_histories[key]
        costs.append(data['history']['cost_values'][-1])
        wts.append(data['history']['wt_values'][-1])
    
    # 非支配解の取得
    solutions = np.array(list(zip(costs, wts)))
    non_dominated_inds = get_non_dominated_inds(solutions)
    pareto_front = solutions[non_dominated_inds]
    
    # すべての最終解をプロット
    plt.scatter(costs, wts, c='lightgray', s=100, alpha=0.7, label='すべての解')
    
    # パレートフロントをプロット
    if len(pareto_front) > 0:
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', s=150, 
                    alpha=0.9, label='パレート最適解')
        # パレートフロントを線で結ぶ
        sorted_front = pareto_front[pareto_front[:, 0].argsort()]
        plt.plot(sorted_front[:, 0], sorted_front[:, 1], 'r--', linewidth=2)
    
    plt.xlabel('コスト', fontsize=14)
    plt.ylabel('待ち時間', fontsize=14)
    plt.title("パレートフロントの進化", fontname='IPAexGothic')
    plt.xlabel("時間報酬", fontname='IPAexGothic')
    plt.ylabel("コスト", fontname='IPAexGothic')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('training_progress_visualization.png', dpi=300)
    plt.show()

def save_pareto_solutions_to_json(self, mode_name: str = "default"):
    """最終的なパレート解集合をJSONファイルとして保存
    
    Args:
        mode_name: 実行モード名（ファイル名に含まれる）
    """
    import json
    import datetime
    import os
    
    # ディレクトリ作成
    save_dir = "result_jsons"
    os.makedirs(save_dir, exist_ok=True)
    
    # 最新の評価結果がない場合は評価を実行
    if not self.evaluation_history:
        print("評価履歴がありません。評価を実行します。")
        self.evaluate(self.env, np.full(self.reward_dim, 100.0, dtype=np.float32))
    
    # 最新の評価結果を取得
    latest_eval = self.evaluation_history[-1]
    
    # 非支配解のインデックスを取得
    non_dominated_inds = get_non_dominated_inds(np.array(latest_eval['all_returns']))
    
    # パレート解集合を構築
    pareto_solutions = []
    
    for i in non_dominated_inds:
        # 実際の値を取得（コストと実行時間）
        actual_values = latest_eval['values'][i]
        
        # 対応する報酬値も取得
        reward_values = latest_eval['all_returns'][i].tolist()
        
        # 解の情報を辞書として追加
        solution = {
            "cost": float(actual_values[0]),      # コスト
            "execution_time": float(actual_values[1]),  # 実行時間
            "reward_cost": reward_values[1],      # コスト報酬
            "reward_time": reward_values[0],      # 時間報酬
            "solution_id": int(i)                 # 解のID
        }
        pareto_solutions.append(solution)
    
    # JSONファイルに書き込むデータ
    data = {
        "mode": mode_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "global_step": int(self.global_step),
        "total_episodes": len(self.experience_replay),
        "pareto_solutions": pareto_solutions,
        "pareto_solution_count": len(pareto_solutions),
        # 追加のメタデータ
        "environment_info": {
            "reward_dim": self.reward_dim,
            "continuous_action": self.continuous_action
        },
        "model_info": {
            "hidden_dim": self.hidden_dim,
            "learning_rate": float(self.learning_rate),
            "gamma": float(self.gamma)
        }
    }
    
    # 一意のファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/pareto_solutions_{mode_name}_{timestamp}_step{self.global_step}.json"
    
    # JSONファイルに保存
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"パレート解集合をJSONファイルに保存しました: {filename}")
    print(f"パレート解の数: {len(pareto_solutions)}")
    
    return filename

def run_nsga2_mode(nb_jobs: int, pop_size: int = 100, num_generations: int = 100):
    """NSGA-IIモードの実行"""
    # 環境パラメータの設定
    next_init_windows = None
    job_generator = JobGenerator(
        0, 1,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, nb_jobs, 0.2, 0
    )
    jobs_set = job_generator.generate_jobs_set()

    print(f"生成されたjobs_set: {jobs_set}")
    env = SchedulingEnv(
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
    
    # エージェントの初期化と最適化実行
    agent = NSGA2Agent(
        pop_size=pop_size,
        num_generations=num_generations,
        crossover_prob=0.7,
        mutation_prob=0.1,
        eliminate_duplicates=True
    )
    
    # n_jobsパラメータを追加（使用するCPUコア数、-1は全コア使用）
    result = agent.run(env, nb_jobs, n_jobs=-1)
    
    # パレートフロントの可視化（独自のグラフ）
    visualize_nsga2_results(result)
    
    return result['objectives']

def run_nsga2_mode_distributed(nb_jobs: int, pop_size: int = 100, num_generations: int = 100, num_workers: int = 32, migration_interval: int = 10):
    """分散処理版NSGA-IIモードの実行"""
    # 環境パラメータの設定
    next_init_windows = None
    np.random.seed(0)
    job_generator = JobGenerator(
        0, 1,
        config['param_env']['n_window'],
        config['param_env']['n_on_premise_node'],
        config['param_env']['n_cloud_node'],
        config, nb_jobs, 0.2, 1
    )
    jobs_set = job_generator.generate_jobs_set()
    env = SchedulingEnv(
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
    
    # 分散処理版エージェントの初期化
    # ワーカーあたりの集団サイズを計算
    pop_size_per_worker = pop_size // num_workers
    
    agent = DistributedNSGA2Agent(
        num_workers=num_workers,
        pop_size_per_worker=pop_size_per_worker,
        num_generations=num_generations,
        migration_interval=migration_interval,
        crossover_prob=0.9,
        mutation_prob=0.3
    )
    
    # 最適化実行
    result = agent.run(env, nb_jobs)
    
    # パレートフロントの可視化
    from src.agents.nsga2_agent_distributed import visualize_nsga2_results
    visualize_nsga2_results(result)
    
    return result['objectives']

def visualize_nsga2_results(result):
    """NSGA-IIの結果を可視化する関数"""
    plt.figure(figsize=(12, 10))
    
    # パレートフロントのプロット
    objectives = result['objectives']
    plt.scatter(objectives[:, 1], objectives[:, 0], c='blue', s=100, label='NSGA-II Pareto Front', alpha=0.7)
    
    # 各解をラベル付け
    for i, obj in enumerate(objectives):
        plt.annotate(
            f"{i+1}", 
            (obj[1], obj[0]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.title('NSGA-IIによる最適化結果のパレートフロント', fontsize=16)
    plt.xlabel('待ち時間（Makespan）', fontsize=14)
    plt.ylabel('コスト', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    # 保存
    plt.tight_layout()
    plt.savefig('nsga2_pareto_front.png')
    plt.close()
    
    # 詳細結果の表示と保存
    print("\nNSGA-II Pareto Front:")
    print("----------------------------------")
    print("No. | コスト  | 待ち時間")
    print("----------------------------------")
    
    for i, obj in enumerate(objectives):
        print(f"{i+1:2d} | {obj[0]:7.2f} | {obj[1]:7.2f}")

def run_heuristic_mode(nb_jobs: int, base_threshold: int = 5, width_factor: float = 0.3):
    """ヒューリスティックモードの実行"""
    # 環境パラメータの設定
    next_init_windows = None
    np.random.seed(0)
    job_generator = JobGenerator(0, nb_steps, n_window, n_on_premise_node, 
                               n_cloud_node, config, nb_jobs, 0.2, 0)
    jobs_set = job_generator.generate_jobs_set()

    # 環境の初期化
    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows, flag=0
    )
    
    # 環境をリセット
    env.reset()
    
    # ヒューリスティックエージェントの初期化
    from src.agents.heuristic_agent import HeuristicAgent
    agent = HeuristicAgent(
        base_wait_time_threshold=base_threshold,
        width_factor=width_factor,
        use_cloud_fallback=True
    )
    
    # ジョブの順次処理
    total_wait_time = 0
    total_cost = 0
    completed_jobs = 0
    
    # エピソードの実行
    while not env.check_is_done():
        # ジョブキューが空でない場合
        if env.rear_job_queue > 0:
            # ヒューリスティックでジョブをスケジュール
            result = agent.schedule_job(env)
            
            if result is not None:
                wait_time, cost, position = result
                total_wait_time += wait_time
                total_cost += cost
                completed_jobs += 1
                
                # ジョブキューから削除
                env.job_queue = np.roll(env.job_queue, -1, axis=0)
                env.job_queue[-1] = 0
                env.rear_job_queue -= 1
        
        # 時間を進める
        env.time_transition()
        env.update_window_history()
        env.step_count += 1
        
        # 新しいジョブを追加
        env.append_new_job2job_queue()
    
    # 環境の最終化
    env.finalize_window_history()
    
    # 結果の表示
    print(f"\n=== ヒューリスティック実行結果 ===")
    print(f"完了ジョブ数: {completed_jobs}")
    print(f"総待ち時間: {total_wait_time}")
    print(f"総コスト: {total_cost}")
    if completed_jobs > 0:
        print(f"平均待ち時間: {total_wait_time / completed_jobs:.2f}")
        print(f"平均コスト: {total_cost / completed_jobs:.2f}")
    
    # 統計情報の表示
    agent.print_stats()
    
    # パレートフロントの可視化
    visualize_heuristic_results(env, agent)
    
    return {
        'wait_time': total_wait_time,
        'cost': total_cost,
        'completed_jobs': completed_jobs,
        'stats': agent.get_stats()
    }

def visualize_heuristic_results(env, agent):
    """Visualize heuristic results"""
    # Get statistics
    stats = agent.get_stats()
    
    plt.figure(figsize=(12, 8))
    
    # Job allocation ratio pie chart
    plt.subplot(2, 2, 1)
    labels = ['On-Premise', 'Cloud']
    sizes = [stats['on_premise_allocations'], stats['cloud_allocations']]
    colors = ['lightblue', 'lightcoral']
    
    if sum(sizes) > 0:
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Job Allocation Ratio')
    else:
        plt.text(0.5, 0.5, 'No jobs allocated', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Job Allocation Ratio')
    
    # Fallback situation
    plt.subplot(2, 2, 2)
    if stats['on_premise_allocations'] > 0:
        fallback_rate = stats['fallback_to_cloud'] / stats['on_premise_allocations']
        plt.bar(['Fallback Rate'], [fallback_rate], color='orange')
        plt.title('Cloud Fallback Rate')
        plt.ylim(0, 1)
    else:
        plt.text(0.5, 0.5, 'No on-premise allocations', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Cloud Fallback Rate')
    
    # Environment state information
    plt.subplot(2, 2, 3)
    env_info = [
        f"Completed Jobs: {env.completed_jobs if hasattr(env, 'completed_jobs') else 'N/A'}",
        f"Total Wait Time: {env.total_waiting_time if hasattr(env, 'total_waiting_time') else 'N/A'}",
        f"Total Cost: {env.total_cost if hasattr(env, 'total_cost') else 'N/A'}",
        f"Step Count: {env.step_count if hasattr(env, 'step_count') else 'N/A'}"
    ]
    plt.text(0.1, 0.8, '\n'.join(env_info), transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.title('Environment Info')
    plt.axis('off')
    
    # Statistics summary
    plt.subplot(2, 2, 4)
    if sum(sizes) > 0:
        summary_text = [
            f"Total Jobs: {sum(sizes)}",
            f"On-Premise: {stats['on_premise_allocations']}",
            f"Cloud: {stats['cloud_allocations']}",
            f"Fallback: {stats['fallback_to_cloud']}"
        ]
        plt.text(0.1, 0.8, '\n'.join(summary_text), transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.title('Statistics Summary')
    else:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Statistics Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('heuristic_results.png', dpi=300)
    plt.show()

@ray.remote
class DistributedDQNWorker:
    """Ray リモートワーカークラス"""
    
    def __init__(self, config):
        self.config = config
        
    def train_with_weight(self, nb_steps, how_many_episodes, nb_jobs, w_wt, w_cost, weight_id, 
                         max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, 
                         n_job_queue_bck, penalty_not_allocate, penalty_invalid_action):
        """指定された重みでDQNエージェントを学習（PCNスタイル）"""
        import numpy as np
        from src.agents.dqn_agent import DQNAgent
        from src.envs.scheduling_env import SchedulingEnv
        from src.utils.job_gen.job_generator import JobGenerator
        import time
        
        start_time = time.time()
        print(f"Worker {weight_id}: 学習開始 - 重み[WT: {w_wt:.3f}, Cost: {w_cost:.3f}]")
        
        try:
            # 環境の初期化（PCNスタイル）

            job_generator = JobGenerator(
                0, 1,
                self.config['param_env']['n_window'],
                self.config['param_env']['n_on_premise_node'],
                self.config['param_env']['n_cloud_node'],
                self.config, nb_jobs, 0.2, 0
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
            
            # エージェントの初期化
            agent = DQNAgent(
                self.env,
                device="auto",
                state_dim=38440,
                learning_rate=1e-2,
                gamma=0.95,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.99,
                buffer_size=2000,
                batch_size=512,
                hidden_dim=512,
                target_update=10,
                # weight_wt=w_wt,
                weight_cost=w_cost,
                weight_id=weight_id
            )
            
            # 学習実行
            losses = agent.train(how_many_episodes)
            
            # 最終的な値を計算
            cost, _, wt = self.env.calc_objective_values()
            
            elapsed_time = time.time() - start_time
            print(f"Worker {weight_id}: 学習完了 - 時間: {elapsed_time:.2f}s, Cost: {cost:.2f}, WT: {wt:.2f}")
            
            return {
                'weight_id': weight_id,
                'weight_wt': w_wt,
                'weight_cost': w_cost,
                'cost': cost,
                'waiting_time': wt,
                'losses': losses,
                'final_episode': len(losses) if losses else 0,
                'training_time': elapsed_time
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"Worker {weight_id}: 学習エラー - {str(e)[:100]}... 時間: {elapsed_time:.2f}s")
            return {
                'weight_id': weight_id,
                'weight_wt': w_wt,
                'weight_cost': w_cost,
                'cost': float('inf'),
                'waiting_time': float('inf'),
                'losses': [],
                'final_episode': 0,
                'training_time': elapsed_time,
                'error': str(e),
                'training_history': {},
                'best_wt': float('inf'),
                'best_cost': float('inf')
            }

def run_pareto_search_distributed(nb_steps: int, how_many_episodes: int, nb_jobs: int, 
                                 weight_steps: int = 10, num_workers: int = None):
    """
    Rayを使った分散パレート探索
    
    Args:
        nb_steps: シミュレーションステップ数
        how_many_episodes: 学習エピソード数
        nb_jobs: ジョブ数
        weight_steps: 重みの分割数
        num_workers: 並列ワーカー数（Noneの場合は重み数と同じ）
    """
    print(f"\n{'='*60}")
    print("分散パレート探索開始")
    print(f"開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"重み分割数: {weight_steps}")
    print(f"学習エピソード数: {how_many_episodes}")
    print(f"ジョブ数: {nb_jobs}")
    print(f"{'='*60}")
    
    # Rayの初期化
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 並列数の設定
    if num_workers is None:
        num_workers = min(weight_steps, ray.cluster_resources().get('CPU', 1))
    
    print(f"使用ワーカー数: {num_workers}")
    
    # 重みを生成
    weights = np.linspace(0, 1, weight_steps)
    print(weights)
    
    # 設定ファイルから環境パラメータを読み込み
    with open('config/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)
    
    # ワーカーを作成
    workers = [DistributedDQNWorker.remote(config) for _ in range(num_workers)]
    
    # タスクを分散実行
    futures = []
    worker_idx = 0
    
    for w_idx, w_wt in enumerate(weights):
        w_cost = 1 - w_wt
        weight_id = f"weight_{w_idx}"
        print(f"weight_id: {weight_id}, w_wt: {w_wt}, w_cost: {w_cost}")
        
        # ワーカーを循環利用
        worker = workers[worker_idx % num_workers]
        worker_idx += 1
        
        # リモートタスクを開始
        future = worker.train_with_weight.remote(
            nb_steps, how_many_episodes, nb_jobs, w_wt, w_cost, weight_id,
            max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, 
            n_job_queue_bck, penalty_not_allocate, penalty_invalid_action
        )
        futures.append(future)
    
    print(f"分散学習タスクを開始しました ({len(futures)} タスク)")
    
    # 結果を収集
    results = []
    completed = 0
    
    while futures:
        # 完了したタスクを取得
        ready, futures = ray.wait(futures, num_returns=1, timeout=10.0)
        
        for future in ready:
            try:
                result = ray.get(future)
                results.append(result)
                completed += 1
                
                print(f"完了 ({completed}/{weight_steps}): {result['weight_id']} - "
                      f"wt={result['weight_wt']:.2f}, cost={result['weight_cost']:.2f}, "
                      f"値=(cost:{result['cost']:.2f}, wt:{result['waiting_time']:.2f})")
                
            except Exception as e:
                print(f"タスク実行エラー: {e}")
                completed += 1
    
    print(f"\n全ての分散学習が完了しました。")
    
    # 結果の検証
    if not results:
        print("警告: 結果が空です。パレート最適解の計算をスキップします。")
        return []
    
    print(f"収集された結果数: {len(results)}")
    
    # 結果を重みでソート
    results.sort(key=lambda x: x['weight_wt'])
    
    # パレートフロントの計算
    costs = [r['cost'] for r in results]
    wts = [r['waiting_time'] for r in results]
    solutions = np.array(list(zip(costs, wts)))
    
    print(f"結果数: {len(results)}")
    print(f"解の形状: {solutions.shape}")
    print(f"コスト範囲: [{min(costs):.4f}, {max(costs):.4f}]")
    print(f"待ち時間範囲: [{min(wts):.4f}, {max(wts):.4f}]")
    
    # 非支配解の取得
    non_dominated_inds = get_non_dominated_inds_minimize(solutions)
    print(f"非支配解のインデックス: {non_dominated_inds}")
    print(f"非支配解の数: {len(non_dominated_inds)}")
    
    # numpy配列をリストに変換し、インデックスの範囲チェック
    if hasattr(non_dominated_inds, '__iter__'):
        non_dominated_inds = list(non_dominated_inds)
    
    # インデックスの範囲チェック
    if len(non_dominated_inds) > 0:
        max_index = max(non_dominated_inds)
        if max_index >= len(results):
            print(f"警告: 最大インデックス {max_index} が結果数 {len(results)} を超えています")
            # 範囲内のインデックスのみを使用
            valid_indices = [int(i) for i in non_dominated_inds if i < len(results)]
            print(f"有効なインデックス: {valid_indices}")
            non_dominated_inds = valid_indices
        else:
            # インデックスを整数に変換
            non_dominated_inds = [int(i) for i in non_dominated_inds]
    
    if len(non_dominated_inds) == 0:
        print("警告: 非支配解が見つかりませんでした。")
        pareto_front = np.array([])
        pareto_results = []
    else:
        pareto_front = solutions[non_dominated_inds]
        pareto_results = [results[i] for i in non_dominated_inds]
    
    # 結果の保存
    save_distributed_pareto_results(results, pareto_results, pareto_front, nb_jobs, weight_steps, how_many_episodes)
    
    # 可視化
    visualize_distributed_pareto_results(results, pareto_front)
    
    # Rayをシャットダウン
    # ray.shutdown()
    
    return [(r['cost'], r['waiting_time']) for r in results]

def create_progress_save_callback(weight_id: str, nb_jobs: int, weight_steps: int, how_many_episodes: int):
    """100エピソードごとの学習進捗を保存するコールバック関数"""
    
    # 保存ディレクトリの作成
    save_dir = "distributed_pareto_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 進捗保存用のファイル名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_filename = f"{save_dir}/progress_{weight_id}_{timestamp}.json"
    
    # 進捗データを格納するリスト
    progress_data = []
    
    def save_progress_callback(current_results: dict, episode: int):
        """進捗保存コールバック"""
        progress_entry = {
            'episode': episode,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weight_id': current_results['weight_id'],
            'weight_wt': current_results['weight_wt'],
            'weight_cost': current_results['weight_cost'],
            'cost': current_results['cost'],
            'waiting_time': current_results['waiting_time'],
            'epsilon': current_results['epsilon'],
            'memory_size': current_results['memory_size'],
            'training_loss': current_results['training_loss']
        }
        
        progress_data.append(progress_entry)
        
        # ファイルに保存
        with open(progress_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'weight_id': weight_id,
                    'nb_jobs': nb_jobs,
                    'weight_steps': weight_steps,
                    'how_many_episodes': how_many_episodes,
                    'created_at': timestamp,
                    'total_progress_entries': len(progress_data)
                },
                'progress_data': progress_data
            }, f, indent=2, ensure_ascii=False)
    
    return save_progress_callback

def save_distributed_pareto_results(all_results, pareto_results, pareto_front, nb_jobs, weight_steps, how_many_episodes):
    """分散パレート探索の結果をテキストファイルとJSONファイルに保存"""
    
    # 保存ディレクトリの作成
    save_dir = "distributed_pareto_results"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. テキストファイルに詳細結果を保存
    txt_filename = f"{save_dir}/pareto_details_distributed_{timestamp}.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"分散パレート探索結果\n")
        f.write(f"{'='*60}\n")
        f.write(f"実行時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ジョブ数: {nb_jobs}\n")
        f.write(f"重み分割数: {weight_steps}\n")
        f.write(f"学習エピソード数: {how_many_episodes}\n")
        f.write(f"パレート最適解数: {len(pareto_results)}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("全解の結果:\n")
        f.write(f"{'No.':<4} {'重み(待ち時間)':<12} {'重み(コスト)':<10} {'実際のコスト':<12} {'実際の待ち時間':<15} {'パレート最適':<10}\n")
        f.write(f"{'-'*80}\n")
        
        pareto_indices = set(i for i, result in enumerate(all_results) 
                           if any(r['weight_id'] == result['weight_id'] for r in pareto_results))
        
        for i, result in enumerate(all_results):
            is_pareto = "Yes" if i in pareto_indices else "No"
            f.write(f"{i+1:<4} {result['weight_wt']:<12.3f} {result['weight_cost']:<10.3f} "
                   f"{result['cost']:<12.2f} {result['waiting_time']:<15.2f} {is_pareto:<10}\n")
        
        f.write(f"\n\nパレート最適解詳細:\n")
        f.write(f"{'No.':<4} {'重み(待ち時間)':<12} {'重み(コスト)':<10} {'実際のコスト':<12} {'実際の待ち時間':<15}\n")
        f.write(f"{'-'*70}\n")
        
        for i, result in enumerate(pareto_results):
            f.write(f"{i+1:<4} {result['weight_wt']:<12.3f} {result['weight_cost']:<10.3f} "
                   f"{result['cost']:<12.2f} {result['waiting_time']:<15.2f}\n")
    
    # 2. JSONファイルに構造化データを保存
    json_filename = f"{save_dir}/pareto_data_distributed_{timestamp}.json"
    json_data = {
        "metadata": {
            "execution_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nb_jobs": nb_jobs,
            "weight_steps": weight_steps,
            "how_many_episodes": how_many_episodes,
            "total_solutions": len(all_results),
            "pareto_solutions": len(pareto_results)
        },
        "all_solutions": [
            {
                "solution_id": i,
                "weight_id": result['weight_id'],
                "weight_wt": result['weight_wt'],
                "weight_cost": result['weight_cost'],
                "cost": result['cost'],
                "waiting_time": result['waiting_time'],
                "final_episode": result['final_episode'],
                "is_pareto_optimal": any(r['weight_id'] == result['weight_id'] for r in pareto_results)
            }
            for i, result in enumerate(all_results)
        ],
        "pareto_front": [
            {
                "cost": float(point[0]),
                "waiting_time": float(point[1])
            }
            for point in pareto_front
        ]
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を保存しました:")
    print(f"  テキスト: {txt_filename}")
    print(f"  JSON: {json_filename}")

def visualize_distributed_pareto_results(all_results, pareto_front):
    """分散パレート探索の結果を可視化"""
    plt.figure(figsize=(15, 12))
    
    # カラーマップの設定
    cmap = plt.cm.viridis
    num_results = len(all_results)
    
    # サブプロット1: パレートフロント
    plt.subplot(2, 2, 1)
    
    # 全解をプロット
    costs = [r['cost'] for r in all_results]
    wts = [r['waiting_time'] for r in all_results]
    weights_wt = [r['weight_wt'] for r in all_results]
    
    scatter = plt.scatter(costs, wts, c=weights_wt, cmap=cmap, s=100, alpha=0.7, 
                         label='all')
    plt.colorbar(scatter, label='weight wait time')
    
    # パレートフロントを強調
    if len(pareto_front) > 0:
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', s=150, 
                   alpha=0.9, marker='s', label='pareto optimal')
        
        # パレートフロントを線で結ぶ
        sorted_front = pareto_front[pareto_front[:, 0].argsort()]
        plt.plot(sorted_front[:, 0], sorted_front[:, 1], 'r--', linewidth=2, alpha=0.8)
    
    plt.xlabel('cost', fontsize=12)
    plt.ylabel('wait time', fontsize=12)
    plt.title('distributed pareto front', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # サブプロット2: 重みと結果の関係（コスト）
    plt.subplot(2, 2, 2)
    plt.plot(weights_wt, costs, 'o-', linewidth=2, markersize=6)
    plt.xlabel('weight wait time', fontsize=12)
    plt.ylabel('cost', fontsize=12)
    plt.title('weight and cost', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # サブプロット3: 重みと結果の関係（待ち時間）
    plt.subplot(2, 2, 3)
    plt.plot(weights_wt, wts, 'o-', color='orange', linewidth=2, markersize=6)
    plt.xlabel('weight wait time', fontsize=12)
    plt.ylabel('wait time', fontsize=12)
    plt.title('weight and wait time', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # サブプロット4: 統計情報
    plt.subplot(2, 2, 4)
    
    # パレート最適解の統計
    pareto_costs = pareto_front[:, 0] if len(pareto_front) > 0 else []
    pareto_wts = pareto_front[:, 1] if len(pareto_front) > 0 else []
    
    stats_text = [
        f"total solutions: {len(all_results)}",
        f"pareto optimal solutions: {len(pareto_front)}",
        f"pareto rate: {len(pareto_front)/len(all_results)*100:.1f}%",
        "",
        "cost range:",
        f"  min: {min(costs):.2f}",
        f"  max: {max(costs):.2f}",
        "",
        "wait time range:",
        f"  min: {min(wts):.2f}",
        f"  max: {max(wts):.2f}",
    ]
    
    if len(pareto_costs) > 0:
        stats_text.extend([
            "",
            "pareto optimal range:",
            f"  cost: {min(pareto_costs):.2f} - {max(pareto_costs):.2f}",
            f"  wait time: {min(pareto_wts):.2f} - {max(pareto_wts):.2f}"
        ])
    
    plt.text(0.05, 0.95, '\n'.join(stats_text), transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.title('statistics', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "distributed_pareto_results"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/pareto_visualization_distributed_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"visualization saved: {save_dir}/pareto_visualization_distributed_{timestamp}.png")

if __name__ == "__main__":
    # コマンドライン引数の解析
    args = parse_args()

    if args.mode == 'pcn':
        # 強化学習モードのパラメータ設定と実行
        loops = 0
        lams = [0.2] * loops
        how_many_episodes = 1000000
        ob_number = 1
        nb_jobs = args.nb_jobs
        mapmap = run_PCN_mode(
            nb_steps, lams, loops, how_many_episodes, ob_number, nb_jobs, 
            use_cnn=args.use_cnn,  # CNNを使用するかどうかをコマンドライン引数から設定
            use_wandb=args.use_wandb,  # Wandbを使用するかどうかをコマンドライン引数から設定
            load_model=args.load_model,  # 保存済みモデルを読み込むかどうかをコマンドライン引数から設定
            model_path=args.model_path  # 読み込むモデルのパスをコマンドライン引数から設定
        )
        print("PCN強化学習による実行が完了しました")
        if args.use_cnn:
            print("CNNベースの拡張モデルを使用しました")
        if args.use_wandb:
            print("Wandbによるログ記録を有効にしました")
        if args.load_model:
            print(f"保存済みモデル weights/{args.model_path}の学習を引き継ぎました")

    elif args.mode == 'single':
        loops = 0
        lams = [0.2] * loops
        how_many_episodes = 50000
        ob_number = 1
        nb_jobs = args.nb_jobs
        mapmap = run_single_rl_mode(nb_steps, lams, loops, how_many_episodes, ob_number, nb_jobs)
        print("単目的強化学習による実行が完了しました")

    elif args.mode == 'pareto':     
        how_many_episodes = 10000000
        pareto_points_values= run_pareto_search(
            nb_steps,
            how_many_episodes,
            nb_jobs=args.nb_jobs,
            weight_steps=3 # 重みの分割数
        )
        print("\nPareto Front Points:")
        for wt, cost in pareto_points_values:
            print(f"Waiting Time: {wt:.2f}, Cost: {cost:.2f}")
        
    elif args.mode == 'all':
        # 全探索モードの実行
        results_reward = run_exhaustive_mode(args.nb_jobs)
        print("全探索による実行が完了しました")
        
    elif args.mode == 'all_distributed':
        # 分散処理による全探索モードの実行
        results_reward = run_exhaustive_mode_distributed(args.nb_jobs)
        if args.nb_jobs >= 20:
            print("分散処理によるサンプリング探索の実行が完了しました")
        else:
            print("分散処理による全探索の実行が完了しました")
        
    elif args.mode == 'nsga2':
        # NSGA-IIによる最適化
        results = run_nsga2_mode(
            nb_jobs=args.nb_jobs,
            pop_size=args.pop_size,
            num_generations=args.num_generations
        )
        print("NSGA-IIによる最適化が完了しました")
        
        # 結果の活用例：非支配解を取得して表示
        non_dominated_inds = get_non_dominated_inds(results)
        pareto_front = results[non_dominated_inds]
        
        print("\nNSGA-II Pareto Front (Non-dominated solutions):")
        for i, solution in enumerate(pareto_front):
            print(f"Solution {i+1}: Cost = {solution[0]:.2f}, Makespan = {solution[1]:.2f}")

    elif args.mode == 'nsga2_distributed':
        # 分散処理によるNSGA-IIの実行
        results = run_nsga2_mode_distributed(
            nb_jobs=args.nb_jobs,
            pop_size=args.pop_size,
            num_generations=args.num_generations,
            num_workers=args.num_workers,
            migration_interval=args.migration_interval
        )
        print("分散処理によるNSGA-IIの最適化が完了しました")
        
        # 結果の活用例：非支配解を取得して表示
        non_dominated_inds = get_non_dominated_inds(results)
        pareto_front = results[non_dominated_inds]
        
        print("\n分散処理版NSGA-II Pareto Front (Non-dominated solutions):")
        for i, solution in enumerate(pareto_front):
            print(f"Solution {i+1}: Cost = {solution[0]:.2f}, Makespan = {solution[1]:.2f}")

    elif args.mode == 'pareto_distributed':
        # 分散処理によるパレート探索の実行
        pareto_points_values = run_pareto_search_distributed(
            nb_steps,
            how_many_episodes=args.how_many_episodes,
            nb_jobs=args.nb_jobs,
            weight_steps=args.weight_steps,
            num_workers=args.num_workers
        )
        print("分散処理によるパレート探索が完了しました")
        print("\n分散パレート探索結果:")
        # for cost, wt in pareto_points_values:
        #     print(f"Waiting Time: {wt:.2f}, Cost: {cost:.2f}")

    elif args.mode == 'heuristic':
        # ヒューリスティックモードの実行
        results = run_heuristic_mode(
            nb_jobs=args.nb_jobs,
            base_threshold=args.base_threshold,
            width_factor=args.width_factor
        )
        print("ヒューリスティックによる実行が完了しました")




