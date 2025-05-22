import numpy as np
import yaml
import sys
import argparse
import itertools
import matplotlib.pyplot as plt
import signal  # シグナル処理のためのモジュールを追加

# グローバル終了フラグ
TERMINATE_SOFT = False  # 通常終了フラグ
TERMINATE_HARD = False  # 強制終了フラグ

# 強制終了用のシグナルハンドラを定義
def force_quit_handler(sig, frame):
    global TERMINATE_HARD
    print("\n強制終了シグナルを受信しました。安全に終了します...")
    TERMINATE_HARD = True

# 通常終了用のシグナルハンドラを定義
def normal_quit_handler(sig, frame):
    global TERMINATE_SOFT
    print("\n中断シグナルを受信しました。安全に終了するまでお待ちください...")
    print("即座に強制終了するにはCtrl+\\を押してください。")
    TERMINATE_SOFT = True

# SIGQUITシグナル（Ctrl+\）をハンドラに登録
signal.signal(signal.SIGQUIT, force_quit_handler)
# SIGINTシグナル（Ctrl+C）をハンドラに登録
signal.signal(signal.SIGINT, normal_quit_handler)

# 終了フラグをチェックする関数
def check_termination():
    if TERMINATE_HARD:
        print("強制終了を実行します。")
        sys.exit(2)
    if TERMINATE_SOFT:
        print("安全に終了しています...")
        sys.exit(0)

from src.agents.pcn_agent import PCN  
from src.envs.scheduling_env import SchedulingEnv
from src.utils.job_gen.job_generator import JobGenerator
from src.utils.map_visualizer import visualize_map
from src.agents.dqn_agent import DQNAgent
from numba import jit
from src.agents.all_agent import ExhaustiveSearchAgent
np.set_printoptions(linewidth=np.inf) 
import itertools
from morl_baselines.common.pareto import get_non_dominated_inds
from src.agents.nsga2_agent import NSGA2Agent

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

def set_and_train(nb_steps, lams, loops, how_many_episodes,ob_number,nb_jobs):
    next_init_windows = None
    values_all = []
    job_generator = JobGenerator(0,nb_steps, n_window, n_on_premise_node, n_cloud_node, config,nb_jobs,0.3,how_many_episodes)
    jobs_set = job_generator.generate_jobs_set()

    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
        next_init_windows,flag=0
    )

    # デバッグモードで1エピソードを実行
    # if DEBUG_MODE:
    #     run_debug_episode(env)
    # else:

    # print("train_and_execute")

    """NNの初期化→学習→割り当て"""
    agent = PCN(
        env,
        device="auto",
        state_dim=1,
        scaling_factor=np.array([0.1, 0.1, 1.0]),
        learning_rate=1e-3,
        batch_size=1024,
        hidden_dim=256,
        project_name="temp",
        experiment_name="PCN",
        log=True,
    )
    agent.train(
        eval_env=env,
        total_timesteps=int(how_many_episodes),
        ref_point=np.array([-1000,-1000]),
        num_er_episodes=2000,
        num_step_episodes=20,  
        num_model_updates=10,
        max_buffer_size=5000,
        known_pareto_front=[1, 1],
    )

    # e_returns = agent.get_e_returns()
    # transitions = agent.get_transitions()
    mapmap = agent.get_mapmap()
    # print("mapmap: ",mapmap)

    import matplotlib.pyplot as plt

    def visualize_map(mapmap):
        # データの取得（最初の配列を使用）
        map_data = mapmap[0]
        
        # プロットの設定
        plt.figure(figsize=(12, 8))
        
        # ヒートマップの描画
        plt.imshow(map_data, cmap='tab20', interpolation='nearest')
        
        # カラーバーの追加
        plt.colorbar(label='Region ID')
        
        # タイトルと軸ラベルの設定
        plt.title('マップの可視化')
        plt.xlabel('X座標')
        plt.ylabel('Y座標')
        
        # グリッドの表示
        plt.grid(True, which='minor', color='black', linestyle='-', alpha=0.2)
        
        # 表示
        plt.show()

    return 0

    # print("values: ",values)
    #0 = waiting time , 1 = cost
    for i in range(loops):
        job_generator = JobGenerator(i,nb_steps, n_window, n_on_premise_node, n_cloud_node, config, nb_jobs,lams[i])
        jobs_set = job_generator.generate_jobs_set()

        next_init_windows = env.get_next_init_windows()
        env = SchedulingEnv(
            max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
            weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, jobs_set,
            next_init_windows,flag=1,exe_mode=0
        )
        values = train_and_execute(env, how_many_episodes, ob_number)
        print("values: ",i,values)
        values_all.append(values)
        print("values_all: ",values_all)
    return values_all

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', 
                      choices=['single', 'pareto', 'pcn', 'all', 'nsga2'], 
                      help='実行モード（single: 単一重み, pareto: パレートフロント探索, pcn: PCN, all: 全探索, nsga2: NSGA-II）')
    # 既存の引数
    parser.add_argument('--how_many_episodes', type=int, default=1000)
    parser.add_argument('--nb_jobs', type=int, default=11)
    # CNN関連の引数
    parser.add_argument('--use_cnn', action='store_true', 
                      help='PCNモードでCNNベースの拡張モデルを使用する')
    # wandb関連の引数
    parser.add_argument('--use_wandb', action='store_true',
                      help='wandbを使用してエピソード数を定期的に送信する')
    # NSGA-II用の引数
    parser.add_argument('--pop_size', type=int, default=200, help='NSGA-IIの集団サイズ')
    parser.add_argument('--num_generations', type=int, default=200, help='NSGA-IIの世代数')
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
        target_update=10,
        weight_wt=1.0,    # 待ち時間を重視
        weight_cost=0.0   # コストの重みを小さく
    )

    losses = agent.train(
        how_many_episodes,
        early_stop_threshold=0.01,
        patience=5,
        min_episodes=50
    )
    
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
                 ob_number: int, nb_jobs: int, use_cnn: bool = True, use_wandb: bool = False) -> list:
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

    # 終了フラグのチェック
    check_termination()

    # CNN拡張モデルを使用するかどうかを設定
    agent = PCN(
        env,
        device="auto",
        state_dim=1,
        scaling_factor=np.array([0.1, 0.01, 1]),
        learning_rate=1e-2,
        batch_size=512,
        hidden_dim=256,
        project_name="temp",
        experiment_name="PCN_Enhanced" if use_cnn else "PCN",
        log=use_wandb,  # wandbによるログ記録を設定
        use_enhanced_model=use_cnn,  # CNNベースの拡張モデルを使用
        debug_mode=False,  # デバッグモードをオフに設定
    )
    
    # PCNエージェントのtrainメソッドを修正して定期的に終了フラグをチェックするようなラッパー関数
    def train_with_termination_check(agent, **kwargs):
        # オリジナルのtrainメソッドの引数を保持
        original_train = agent.train
        
        # 現在のステップ数を追跡する変数
        current_step = 0
        check_interval = 100  # 何ステップごとにチェックするか
        
        # ラップされたコールバック関数
        def wrapped_callback(*args, **callback_kwargs):
            nonlocal current_step
            current_step += 1
            
            # 定期的に終了フラグをチェック
            if current_step % check_interval == 0:
                check_termination()
            
            # 元のコールバックがあれば呼び出す
            if 'callback' in kwargs and kwargs['callback'] is not None:
                return kwargs['callback'](*args, **callback_kwargs)
            return True
        
        # コールバックを上書き
        train_kwargs = kwargs.copy()
        train_kwargs['callback'] = wrapped_callback
        
        # オリジナルのtrainメソッドを呼び出す
        return original_train(**train_kwargs)
    
    # 終了フラグのチェック
    check_termination()
    
    # 修正したtrainメソッドを使用
    train_with_termination_check(
        agent,
        eval_env=env,
        total_timesteps=int(how_many_episodes),
        ref_point=np.array([0,0]),
        num_er_episodes=5000,
        num_step_episodes=50,  
        num_model_updates=2,
        max_buffer_size=10000,
        known_pareto_front=[1, 1],
        max_return=np.array([1000, 1000]),
        use_wandb=use_wandb,  # wandbを使用するかどうかを設定
        log_episode_only=True,  # エピソード数だけをwandbに送信する（常にTrue）
    )

    # 終了フラグのチェック
    check_termination()

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
    
    print("results: ", search_results['results'])
    print("pareto_front: ", search_results['pareto_front'])
    
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
        print(f"\nTraining with weights - Waiting Time: {w_wt:.2f}, Cost: {w_cost:.2f}")
        
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
            weight_cost=w_cost
        )
        
        # 学習履歴を記録するための構造
        history = {
            'episodes': [],
            'wt_values': [],
            'cost_values': []
        }
        
        # 学習実行と履歴の記録
        losses = agent.train(
            how_many_episodes,
            early_stop_threshold=0.01,
            patience=5,
            min_episodes=50
        )
        
        # 最終的な値を記録
        cost, makespan = env.calc_objective_values()
        pareto_points_values.append((cost, makespan))
        
        # このエージェントの学習履歴を保存
        training_histories[weight_id] = {
            'weight_wt': w_wt,
            'weight_cost': w_cost,
            'history': {
                'episodes': [how_many_episodes-1],
                'wt_values': [makespan],
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
    
    # エージェントの初期化と最適化実行
    agent = NSGA2Agent(
        pop_size=pop_size,
        num_generations=num_generations,
        crossover_prob=0.9,
        mutation_prob=0.3
    )
    
    # n_jobsパラメータを追加（使用するCPUコア数、-1は全コア使用）
    result = agent.run(env, nb_jobs, n_jobs=-1)
    
    # パレートフロントの可視化（独自のグラフ）
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

if __name__ == "__main__":
    try:
        # コマンドライン引数の解析
        args = parse_args()

        if args.mode == 'pcn':
            # 強化学習モードのパラメータ設定と実行
            loops = 0
            lams = [0.2] * loops
            how_many_episodes = 100000000
            ob_number = 1
            nb_jobs = args.nb_jobs
            mapmap = run_PCN_mode(
                nb_steps, lams, loops, how_many_episodes, ob_number, nb_jobs, 
                use_cnn=args.use_cnn,  # CNNを使用するかどうかをコマンドライン引数から設定
                use_wandb=args.use_wandb  # wandbを使用するかどうかをコマンドライン引数から設定
            )
            print("PCN強化学習による実行が完了しました")
            if args.use_cnn:
                print("CNNベースの拡張モデルを使用しました")
            if args.use_wandb:
                print("wandbによるロギングを有効にしました")

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
    
    except KeyboardInterrupt:
        print("\nプログラムがユーザーによって中断されました。")
        sys.exit(0)
    except Exception as e:
        print(f"\n予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




    # env2 = SimpleEnv()
    # agent2 = PCN(
    #     env2,
    #     device = "cpu",
    #     scaling_factor=np.array([1, 1,1])
    #     learning_rate=1e-3,
    #     batch_size=256,
    #     project_name="MORL-Baselines",
    #     experiment_name="PCN",
    #     log=True,
    # )
    # agent2.train(
    # eval_env=env2,
    # total_timesteps=int(1e6),
    # ref_point=np.array([0, 0]),
    # num_er_episodes=20,
    # max_buffer_size=50,
    # num_model_updates=50,
    # # max_return=np.array([1.5, 1.5, -0.0]),
    # known_pareto_front=[1,1],
    # )



